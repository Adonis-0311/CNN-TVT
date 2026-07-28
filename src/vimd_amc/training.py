"""Deterministic two-view training with staged VIMD objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import random
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .losses import (
    VIMDLossWeights,
    baseline_two_view_loss,
    vimd_two_view_loss,
)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    mask_start_epoch: int = 2
    contrastive_start_epoch: int = 5
    mask_ramp_epochs: int = 3
    contrastive_ramp_epochs: int = 3
    minimum_full_stage_epochs: int = 3
    gradient_clip: float = 5.0
    patience: int = 5
    num_workers: int = 0
    use_amp: bool = True


@dataclass(frozen=True)
class TrainingObjective:
    """Explicit loss protocol used by baselines and A0--A7 ablations.

    Keeping this separate from the model class prevents architecture type
    checks from silently switching objectives.  In particular, the exact same
    tri-mask network can be trained with teacher, auxiliary, or XCC terms
    independently disabled.
    """

    name: str = "classification_only"
    loss_family: Literal["classification", "vimd"] = "classification"
    use_mask_supervision: bool = False
    use_jammer_auxiliary: bool = False
    use_quality_auxiliary: bool = False
    use_cross_condition_contrastive: bool = False
    use_orthogonality: bool = False

    def __post_init__(self) -> None:
        if self.loss_family not in {"classification", "vimd"}:
            raise ValueError(f"unknown loss family: {self.loss_family}")
        advanced = (
            self.use_mask_supervision
            or self.use_jammer_auxiliary
            or self.use_quality_auxiliary
            or self.use_cross_condition_contrastive
            or self.use_orthogonality
        )
        if self.loss_family == "classification" and advanced:
            raise ValueError("classification loss family cannot enable VIMD objectives")


def _physical_teacher_target(
    teacher: nn.Module,
    view: dict[str, torch.Tensor],
    *,
    output_dtype: torch.dtype,
    device_type: str,
) -> torch.Tensor:
    """Evaluate the fixed physical teacher in float32 outside autocast.

    Component-power ratios can underflow in float16 at the hard SNR/SIR edge.
    The teacher is therefore always evaluated in physical float32 units and
    only its normalized mask target is cast to the student's output dtype.
    """

    with torch.amp.autocast(device_type=device_type, enabled=False):
        target = teacher(
            view["clean"].float(),
            view["jammer"].float(),
            view["unexplained"].float(),
        )
    return target.to(dtype=output_dtype)


def _staged_cosine_learning_rate_factor(
    completed_epochs: int,
    *,
    total_epochs: int,
    selection_start_epoch: int,
) -> float:
    """Hold LR through the minimum full-objective checkpoint stage.

    ``selection_start_epoch`` is zero-based.  The first decayed learning rate
    is installed after that epoch completes and is used by the next epoch.
    """

    if completed_epochs <= selection_start_epoch:
        return 1.0
    decay_span = max(1, total_epochs - selection_start_epoch)
    progress = min(
        1.0,
        (completed_epochs - selection_start_epoch) / decay_span,
    )
    return float(0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * progress)))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _move_view(view: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in view.items()}


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_total, correct, count = 0.0, 0, 0
    for batch in loader:
        view = _move_view(batch["view1"], device)
        labels = batch["label"].to(device)
        output = model(view["x"])
        loss = nn.functional.cross_entropy(output["logits"], labels)
        loss_total += float(loss) * len(labels)
        correct += int(output["logits"].argmax(dim=1).eq(labels).sum())
        count += len(labels)
    return loss_total / max(1, count), correct / max(1, count)


def checkpoint_selection_protocol(config: TrainingConfig) -> dict[str, Any]:
    """Return the exact, serializable validation-checkpoint rule."""

    full_objective_index = max(
        config.mask_start_epoch + max(1, config.mask_ramp_epochs) - 1,
        config.contrastive_start_epoch
        + max(1, config.contrastive_ramp_epochs)
        - 1,
    )
    selection_start_index = (
        full_objective_index
        + max(1, config.minimum_full_stage_epochs)
        - 1
    )
    return {
        "full_objective_epoch": full_objective_index + 1,
        "selection_start_epoch": selection_start_index + 1,
        "configured_eligible_epoch_count": max(
            0,
            int(config.epochs) - selection_start_index,
        ),
        "criterion": {
            "view": "view1",
            "loss": "modulation_cross_entropy",
            "label_smoothing": 0.0,
            "direction": "minimize",
            "strict_improvement_tolerance": 1e-5,
            "patience_eligible_epochs": int(config.patience),
            "auxiliary_losses_included": False,
            "validation_loader_shuffle": False,
        },
    }


def train_model(
    *,
    model: nn.Module,
    teacher: nn.Module | None,
    train_dataset,
    validation_dataset,
    device: torch.device,
    seed: int,
    config: TrainingConfig,
    objective: TrainingObjective | None = None,
    loss_weights: VIMDLossWeights | None = None,
    jammer_support_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    seed_everything(seed)
    model.to(device)
    if teacher is not None:
        teacher.to(device)
    objective = objective or TrainingObjective()
    if objective.use_mask_supervision and teacher is None:
        raise ValueError("mask-supervised objective requires a physical teacher")
    weights = loss_weights or VIMDLossWeights()
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    selection_protocol = checkpoint_selection_protocol(config)
    full_objective_epoch = int(
        selection_protocol["full_objective_epoch"]
    ) - 1
    selection_start_epoch = int(
        selection_protocol["selection_start_epoch"]
    ) - 1
    if jammer_support_mask is not None:
        jammer_support_mask = jammer_support_mask.detach().to(
            device=device,
            dtype=torch.float32,
        )
    if objective.use_jammer_auxiliary and jammer_support_mask is not None:
        if jammer_support_mask.ndim != 1:
            raise ValueError("jammer_support_mask must be one-dimensional")
        if not bool(torch.any(jammer_support_mask > 0.5)):
            raise ValueError(
                "jammer auxiliary objective has no training-supported labels"
            )

    def learning_rate_factor(completed_epochs: int) -> float:
        return _staged_cosine_learning_rate_factor(
            completed_epochs,
            total_epochs=config.epochs,
            selection_start_epoch=selection_start_epoch,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=learning_rate_factor,
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, float]] = []
    best_state = None
    best_teacher_state = None
    best_epoch: int | None = None
    fallback_state = copy.deepcopy(model.state_dict())
    fallback_teacher_state = copy.deepcopy(teacher.state_dict()) if teacher is not None else None
    best_validation = float("inf")
    preselection_best_validation = float("inf")
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        model.train()
        running: dict[str, float] = {}
        sample_count = 0
        mask_factor = float(
            np.clip(
                (epoch - config.mask_start_epoch + 1) / max(1, config.mask_ramp_epochs),
                0.0,
                1.0,
            )
        )
        contrastive_factor = float(
            np.clip(
                (epoch - config.contrastive_start_epoch + 1)
                / max(1, config.contrastive_ramp_epochs),
                0.0,
                1.0,
            )
        )
        enable_mask = objective.use_mask_supervision and mask_factor > 0.0
        enable_contrastive = (
            objective.use_cross_condition_contrastive
            and contrastive_factor > 0.0
        )
        epoch_weights = replace(
            weights,
            mask=weights.mask * mask_factor,
            contrastive=weights.contrastive * contrastive_factor,
        )
        for batch in train_loader:
            first = _move_view(batch["view1"], device)
            second = _move_view(batch["view2"], device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                first_output = model(first["x"])
                second_output = model(second["x"])
                if objective.loss_family == "vimd":
                    first_target = None
                    second_target = None
                    if enable_mask:
                        assert teacher is not None
                        first_target = _physical_teacher_target(
                            teacher,
                            first,
                            output_dtype=first_output["masks"].dtype,
                            device_type=device.type,
                        )
                        second_target = _physical_teacher_target(
                            teacher,
                            second,
                            output_dtype=second_output["masks"].dtype,
                            device_type=device.type,
                        )
                    losses = vimd_two_view_loss(
                        first_output=first_output,
                        second_output=second_output,
                        first_batch=first,
                        second_batch=second,
                        labels=labels,
                        first_teacher_mask=first_target,
                        second_teacher_mask=second_target,
                        weights=epoch_weights,
                        enable_mask=enable_mask,
                        enable_contrastive=enable_contrastive,
                        enable_jammer=objective.use_jammer_auxiliary,
                        enable_quality=objective.use_quality_auxiliary,
                        enable_orthogonality=objective.use_orthogonality,
                        jammer_support_mask=jammer_support_mask,
                    )
                else:
                    losses = baseline_two_view_loss(
                        first_output,
                        second_output,
                        labels,
                        label_smoothing=weights.label_smoothing,
                    )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(labels)
            sample_count += batch_size
            for name, value in losses.items():
                running[name] = running.get(name, 0.0) + float(value.detach()) * batch_size

        scheduler.step()
        val_loss, val_accuracy = validation_loss(model, validation_loader, device)
        record = {
            "epoch": float(epoch + 1),
            **{name: value / max(1, sample_count) for name, value in running.items()},
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "mask_enabled": float(enable_mask),
            "contrastive_enabled": float(enable_contrastive),
            "mask_weight_factor": mask_factor,
            "contrastive_weight_factor": contrastive_factor,
            "checkpoint_selection_eligible": float(epoch >= selection_start_epoch),
        }
        history.append(record)
        fallback_state = copy.deepcopy(model.state_dict())
        fallback_teacher_state = (
            copy.deepcopy(teacher.state_dict()) if teacher is not None else None
        )
        if epoch < selection_start_epoch:
            preselection_best_validation = min(preselection_best_validation, val_loss)
        elif val_loss < best_validation - 1e-5:
            best_validation = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            best_teacher_state = (
                copy.deepcopy(teacher.state_dict()) if teacher is not None else None
            )
            epochs_without_improvement = 0
        elif epoch >= selection_start_epoch:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    fallback_used = best_state is None
    selected_epoch = (
        best_epoch
        if best_epoch is not None
        else int(history[-1]["epoch"])
    )
    selected_validation_loss = (
        best_validation
        if np.isfinite(best_validation)
        else history[-1]["validation_loss"]
    )
    eligible_checkpoint_count = sum(
        int(record["checkpoint_selection_eligible"] > 0.5)
        for record in history
    )
    model.load_state_dict(
        best_state if best_state is not None else fallback_state
    )
    teacher_state = (
        best_teacher_state if best_teacher_state is not None else fallback_teacher_state
    )
    if teacher is not None and teacher_state is not None:
        teacher.load_state_dict(teacher_state)
    return {
        "history": history,
        "best_validation_loss": selected_validation_loss,
        "preselection_best_validation_loss": preselection_best_validation,
        "checkpoint_selection_start_epoch": selection_start_epoch + 1,
        "full_objective_epoch": full_objective_epoch + 1,
        "learning_rate_hold_through_epoch": selection_start_epoch + 1,
        "learning_rate_decay_first_applied_epoch": (
            selection_start_epoch + 2
            if selection_start_epoch + 1 < config.epochs
            else None
        ),
        "epochs_completed": len(history),
        "checkpoint_selection": {
            "status": (
                "eligible_validation_checkpoint_selected"
                if not fallback_used
                else "fallback_final_state_no_eligible_checkpoint_selected"
            ),
            "selected_checkpoint_eligible": not fallback_used,
            "fallback_used": fallback_used,
            "selected_epoch": selected_epoch,
            "selected_validation_loss": selected_validation_loss,
            "eligible_checkpoint_count": eligible_checkpoint_count,
            "criterion": selection_protocol["criterion"],
        },
        "selected_epoch": selected_epoch,
        "selection_criterion": selection_protocol["criterion"],
        "checkpoint_fallback_used": fallback_used,
        "training_config": asdict(config),
        "objective": asdict(objective),
        "loss_weights": asdict(weights),
        "jammer_auxiliary_training": {
            "enabled": bool(objective.use_jammer_auxiliary),
            "support_mask": (
                [
                    bool(value)
                    for value in jammer_support_mask.gt(0.5).cpu().tolist()
                ]
                if jammer_support_mask is not None
                else None
            ),
            "support_mask_source": (
                "explicit_runner_training_support_contract"
                if jammer_support_mask is not None
                else "legacy_unmasked_all_columns"
            ),
            "loss": (
                "binary_cross_entropy_mean_over_batch_and_supported_columns"
                if objective.use_jammer_auxiliary
                else None
            ),
            "unsupported_columns_receive_loss_or_gradient": (
                False
                if objective.use_jammer_auxiliary
                and jammer_support_mask is not None
                else None
            ),
            "unsupported_logit_columns_receive_direct_bce_gradient": (
                False
                if objective.use_jammer_auxiliary
                and jammer_support_mask is not None
                else None
            ),
            "shared_backbone_gradient_scope": (
                "shared features update through supported jammer columns; "
                "unsupported jammer logits have exact zero BCE derivative"
                if objective.use_jammer_auxiliary
                and jammer_support_mask is not None
                else None
            ),
        },
        "physical_teacher_precision": (
            "float32 with autocast disabled; normalized target cast to student dtype"
            if objective.use_mask_supervision
            else None
        ),
    }
