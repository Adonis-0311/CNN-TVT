"""Losses with explicit per-component reporting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VIMDLossWeights:
    jammer: float = 0.25
    quality: float = 0.05
    mask: float = 0.50
    contrastive: float = 0.10
    orthogonality: float = 0.01
    label_smoothing: float = 0.05
    contrastive_temperature: float = 0.12


def jensen_shannon_mask_loss(target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Return a numerically stable FP32 Jensen--Shannon divergence.

    Mask probabilities can be exactly zero and the surrounding training
    forward pass may run under automatic mixed precision.  Performing the
    logarithms in float16/bfloat16 makes the hard SNR/SIR cells unnecessarily
    fragile.  The casts below remain differentiable, so the returned FP32
    scalar backpropagates to a lower-precision student mask when applicable.
    """

    if target.shape != predicted.shape:
        raise ValueError(
            "target and predicted masks must have identical shapes: "
            f"{tuple(target.shape)} != {tuple(predicted.shape)}"
        )
    if target.ndim < 2:
        raise ValueError("mask tensors must include batch and route dimensions")
    if not target.is_floating_point() or not predicted.is_floating_point():
        raise TypeError("Jensen--Shannon mask inputs must be floating point")
    if target.device != predicted.device:
        raise ValueError("target and predicted masks must share a device")

    with torch.amp.autocast(device_type=predicted.device.type, enabled=False):
        target_fp32 = target.float()
        predicted_fp32 = predicted.float()
        epsilon = torch.finfo(torch.float32).eps
        target_fp32 = target_fp32.clamp_min(0.0)
        predicted_fp32 = predicted_fp32.clamp_min(0.0)
        target_fp32 = target_fp32 / target_fp32.sum(
            dim=1, keepdim=True
        ).clamp_min(epsilon)
        predicted_fp32 = predicted_fp32 / predicted_fp32.sum(
            dim=1, keepdim=True
        ).clamp_min(epsilon)
        target_fp32 = target_fp32.clamp_min(epsilon)
        predicted_fp32 = predicted_fp32.clamp_min(epsilon)
        # Renormalize after flooring so each argument remains a probability
        # vector and the theoretical JS bound is preserved to FP32 tolerance.
        target_fp32 = target_fp32 / target_fp32.sum(dim=1, keepdim=True)
        predicted_fp32 = predicted_fp32 / predicted_fp32.sum(dim=1, keepdim=True)
        midpoint = 0.5 * (target_fp32 + predicted_fp32)
        target_kl = (
            target_fp32 * (target_fp32.log() - midpoint.log())
        ).sum(dim=1)
        predicted_kl = (
            predicted_fp32 * (predicted_fp32.log() - midpoint.log())
        ).sum(dim=1)
        loss = 0.5 * (target_kl + predicted_kl).mean()
    return loss


def supervised_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    embeddings = F.normalize(torch.cat((first, second), dim=0), dim=1)
    repeated_labels = torch.cat((labels, labels), dim=0)
    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    count = logits.shape[0]
    self_mask = torch.eye(count, device=logits.device, dtype=torch.bool)
    positive_mask = repeated_labels[:, None].eq(repeated_labels[None, :]) & ~self_mask
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positives_per_anchor = positive_mask.sum(dim=1).clamp_min(1)
    mean_positive_log_probability = (
        (positive_mask * log_probability).sum(dim=1) / positives_per_anchor
    )
    return -mean_positive_log_probability.mean()


def paired_cross_condition_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Class-aware bidirectional InfoNCE for paired physical conditions.

    The positive is only the other condition of the same immutable source
    sequence (the diagonal).  Different-modulation sources are negatives;
    other source sequences with the same modulation are ignored so they are
    not turned into false negatives or mislabeled as instance-level positives.
    """

    first = F.normalize(first, dim=1)
    second = F.normalize(second, dim=1)
    logits = first @ second.T / temperature
    batch_size = logits.shape[0]
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
    different_class = labels[:, None].ne(labels[None, :])
    allowed = different_class | diagonal

    def directional_loss(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_scores = scores.masked_fill(~mask, -torch.inf)
        log_denominator = torch.logsumexp(masked_scores, dim=1)
        positive = scores.diagonal()
        has_negative = different_class.any(dim=1)
        losses = log_denominator - positive
        if has_negative.any():
            return losses[has_negative].mean()
        return scores.sum() * 0.0

    return 0.5 * (
        directional_loss(logits, allowed)
        + directional_loss(logits.T, allowed.T)
    )


def orthogonality_loss(
    modulation_embedding: torch.Tensor,
    jammer_embedding: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if active_mask is not None:
        active_mask = active_mask.to(dtype=torch.bool, device=modulation_embedding.device)
        modulation_embedding = modulation_embedding[active_mask]
        jammer_embedding = jammer_embedding[active_mask]
    if modulation_embedding.shape[0] == 0:
        return modulation_embedding.sum() * 0.0
    if modulation_embedding.shape[0] > 1:
        modulation_embedding = (
            modulation_embedding - modulation_embedding.mean(dim=0, keepdim=True)
        )
        jammer_embedding = jammer_embedding - jammer_embedding.mean(dim=0, keepdim=True)
    modulation_norm = modulation_embedding.norm(dim=1)
    jammer_norm = jammer_embedding.norm(dim=1)
    valid = modulation_norm.gt(1e-8) & jammer_norm.gt(1e-8)
    if not valid.any():
        return modulation_embedding.sum() * 0.0
    cosine = F.cosine_similarity(
        modulation_embedding[valid],
        jammer_embedding[valid],
        dim=1,
    )
    return cosine.square().mean()


def baseline_two_view_loss(
    first_output: dict[str, torch.Tensor],
    second_output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    label_smoothing: float = 0.05,
) -> dict[str, torch.Tensor]:
    first = F.cross_entropy(
        first_output["logits"],
        labels,
        label_smoothing=label_smoothing,
    )
    second = F.cross_entropy(
        second_output["logits"],
        labels,
        label_smoothing=label_smoothing,
    )
    loss = 0.5 * (first + second)
    return {"total": loss, "modulation": loss}


def masked_jammer_binary_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    support_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Compute BCE only over jammer labels with positive training support.

    A taxonomy column that is held out or excluded from the training split is
    not a supervised family-recognition target.  Masking after an unreduced
    BCE gives those logits exactly zero gradient while retaining the fixed
    output schema needed by cached labels and diagnostics.
    """

    if logits.shape != targets.shape:
        raise ValueError(
            "jammer logits and targets must have identical shapes: "
            f"{tuple(logits.shape)} != {tuple(targets.shape)}"
        )
    if logits.ndim != 2:
        raise ValueError("jammer logits/targets must have shape [batch, class]")
    if support_mask is None:
        mask = torch.ones(
            logits.shape[1],
            dtype=logits.dtype,
            device=logits.device,
        )
    else:
        if support_mask.ndim != 1 or support_mask.numel() != logits.shape[1]:
            raise ValueError(
                "jammer support mask must have one entry per output column"
            )
        mask = support_mask.to(device=logits.device, dtype=logits.dtype)
        if not bool(torch.all(torch.isfinite(mask))):
            raise ValueError("jammer support mask contains nonfinite values")
        if not bool(torch.all((mask == 0) | (mask == 1))):
            raise ValueError("jammer support mask must be binary")
    supported = mask.sum()
    if float(supported.detach()) <= 0.0:
        raise ValueError(
            "jammer auxiliary loss requires at least one training-supported "
            "taxonomy column"
        )
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    return (elementwise * mask.unsqueeze(0)).sum() / (
        logits.shape[0] * supported
    )


def vimd_two_view_loss(
    *,
    first_output: dict[str, torch.Tensor],
    second_output: dict[str, torch.Tensor],
    first_batch: dict[str, torch.Tensor],
    second_batch: dict[str, torch.Tensor],
    labels: torch.Tensor,
    first_teacher_mask: torch.Tensor | None,
    second_teacher_mask: torch.Tensor | None,
    weights: VIMDLossWeights,
    enable_mask: bool,
    enable_contrastive: bool,
    enable_jammer: bool = True,
    enable_quality: bool = True,
    enable_orthogonality: bool = True,
    jammer_support_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    modulation = 0.5 * (
        F.cross_entropy(
            first_output["logits"],
            labels,
            label_smoothing=weights.label_smoothing,
        )
        + F.cross_entropy(
            second_output["logits"],
            labels,
            label_smoothing=weights.label_smoothing,
        )
    )
    if enable_jammer:
        jammer = 0.5 * (
            masked_jammer_binary_cross_entropy(
                first_output["jam_logits"],
                first_batch["jam_labels"],
                jammer_support_mask,
            )
            + masked_jammer_binary_cross_entropy(
                second_output["jam_logits"],
                second_batch["jam_labels"],
                jammer_support_mask,
            )
        )
    else:
        jammer = modulation.new_zeros(())
    if enable_quality:
        first_quality_error = F.smooth_l1_loss(
            first_output["quality"],
            first_batch["quality"],
            reduction="none",
        )
        second_quality_error = F.smooth_l1_loss(
            second_output["quality"],
            second_batch["quality"],
            reduction="none",
        )
        first_quality = (
            first_quality_error * first_batch["quality_mask"]
        ).sum() / first_batch["quality_mask"].sum().clamp_min(1.0)
        second_quality = (
            second_quality_error * second_batch["quality_mask"]
        ).sum() / second_batch["quality_mask"].sum().clamp_min(1.0)
        quality = 0.5 * (first_quality + second_quality)
    else:
        quality = modulation.new_zeros(())
    if enable_mask:
        if first_teacher_mask is None or second_teacher_mask is None:
            raise ValueError("teacher masks are required when mask supervision is enabled")
        mask = 0.5 * (
            jensen_shannon_mask_loss(first_teacher_mask, first_output["masks"])
            + jensen_shannon_mask_loss(second_teacher_mask, second_output["masks"])
        )
    else:
        mask = modulation.new_zeros(())
    if enable_contrastive:
        contrastive = paired_cross_condition_contrastive_loss(
            first_output["embedding"],
            second_output["embedding"],
            labels,
            weights.contrastive_temperature,
        )
    else:
        contrastive = modulation.new_zeros(())
    if enable_orthogonality:
        orthogonality = 0.5 * (
            orthogonality_loss(
                first_output["embedding"],
                first_output["jammer_embedding"],
                first_batch["quality_mask"][:, 1].gt(0.5),
            )
            + orthogonality_loss(
                second_output["embedding"],
                second_output["jammer_embedding"],
                second_batch["quality_mask"][:, 1].gt(0.5),
            )
        )
    else:
        orthogonality = modulation.new_zeros(())
    total = (
        modulation
        + weights.jammer * jammer
        + weights.quality * quality
        + weights.mask * mask
        + weights.contrastive * contrastive
        + weights.orthogonality * orthogonality
    )
    return {
        "total": total,
        "modulation": modulation,
        "jammer": jammer,
        "quality": quality,
        "mask": mask,
        "contrastive": contrastive,
        "orthogonality": orthogonality,
    }
