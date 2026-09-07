"""Auditable AMC baselines and fair complex-STFT ablations.

Two externally proposed architectures are exposed here with deliberately
different labels:

* :class:`MCLDNNReimplementation` is a framework-port of the authors' public
  three-stream architecture.  Layer topology, widths, padding semantics, LSTM
  depth, activations, and dropout follow the official implementation, but
  TensorFlow/Keras checkpoints are not binary-compatible with this PyTorch
  port.
* :class:`IQFormerInspiredClassifier` is *not* claimed as an exact
  reproduction.  It follows the public RML2016 configuration and core
  IQ/STFT-fusion blocks, while computing its differentiable STFT internally
  and removing the ``timm``/``einops`` dependencies.

Primary sources and pinned revisions are recorded next to each implementation
so that later paper tables cannot silently relabel an approximation as an
official result.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .common import ModelConfig, _group_count
from .spectral import (
    ComplexSTFT,
    SpectralBranchEncoder,
    SpectralContextEncoder,
    SpectralEnvironmentEncoder,
)

MCLDNN_PAPER_URL = "https://doi.org/10.1109/LWC.2020.2999453"
MCLDNN_OFFICIAL_CODE_URL = "https://github.com/wzjialang/MCLDNN"
MCLDNN_AUDITED_COMMIT = "f1093eea5a04ba6f7fc5297171ffbae5c9853f93"

IQFORMER_PAPER_URL = "https://doi.org/10.1109/TCCN.2024.3485118"
IQFORMER_OFFICIAL_CODE_URL = "https://github.com/WestdoorSad/IQFormer"
IQFORMER_AUDITED_COMMIT = "7ee6ac949551b24d45f218762cab919e0cb6b4f9"

CSSL_AMC_PAPER_URL = "https://doi.org/10.1109/TWC.2025.3532438"
CSSL_AMC_OFFICIAL_CODE_URL = (
    "https://github.com/dumingyang20/CSSL-AMC-Pytorch"
)
CSSL_AMC_AUDITED_COMMIT = (
    "2fbc5b3e12f780b0b26eb0ee2c33d592739aa24f"
)
CSSL_AMC_LICENSE = "Apache-2.0"


class BackboneClassifier(nn.Module):
    """A0: the same spectral modulation branch without decomposition."""

    def __init__(self, num_classes: int, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.front_end = ComplexSTFT(self.config.n_fft, self.config.hop_length)
        self.branch = SpectralBranchEncoder(self.config)
        self.classifier = nn.Linear(self.config.embedding_dim, num_classes)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.front_end(values)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        spectrum = self.front_end(values)
        embedding = self.branch(spectrum)
        return {
            "logits": self.classifier(embedding),
            "embedding": embedding,
            "spectrum": spectrum,
            "mod_spectrum": spectrum,
        }


class SingleMaskClassifier(nn.Module):
    """A1: one environment-conditioned mask, with no residual bypass."""

    def __init__(self, num_classes: int, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        channels = self.config.spectral_channels
        self.front_end = ComplexSTFT(self.config.n_fft, self.config.hop_length)
        self.context_encoder = SpectralContextEncoder(self.config)
        self.environment_encoder = SpectralEnvironmentEncoder(self.config)
        self.film = nn.Linear(self.config.environment_dim, channels * 2)
        self.mask_generator = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.branch = SpectralBranchEncoder(self.config)
        self.classifier = nn.Linear(self.config.embedding_dim, num_classes)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.front_end(values)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        spectrum = self.front_end(values)
        context_features = self.context_encoder(spectrum)
        condition = self.environment_encoder(context_features)
        gamma, beta = self.film(condition).chunk(2, dim=1)
        conditioned = (
            (1.0 + 0.5 * torch.tanh(gamma))[:, :, None, None] * context_features
            + beta[:, :, None, None]
        )
        mask = torch.sigmoid(self.mask_generator(conditioned)).squeeze(1)
        mod_spectrum = mask * spectrum
        embedding = self.branch(mod_spectrum)
        return {
            "logits": self.classifier(embedding),
            "embedding": embedding,
            "spectrum": spectrum,
            "features": context_features,
            "mod_spectrum": mod_spectrum,
            "mask": mask.unsqueeze(1),
            "condition": condition,
        }


class DiagnosticDualMaskCEClassifier(nn.Module):
    """Diagnostic CE-only two-mask classifier, outside paper A0--A7.

    The target mask drives a single modulation branch and there is no teacher,
    jammer task branch, or residual bypass.  It is intentionally not
    registered by either paper runner; paper A6 is :class:`DualMaskVIMDNet`.
    """

    def __init__(self, num_classes: int, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        channels = self.config.spectral_channels
        self.front_end = ComplexSTFT(self.config.n_fft, self.config.hop_length)
        self.context_encoder = SpectralContextEncoder(self.config)
        self.environment_encoder = SpectralEnvironmentEncoder(self.config)
        self.film = nn.Linear(self.config.environment_dim, channels * 2)
        self.mask_generator = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, 2, kernel_size=1),
        )
        self.branch = SpectralBranchEncoder(self.config)
        self.classifier = nn.Linear(self.config.embedding_dim, num_classes)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.front_end(values)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        spectrum = self.front_end(values)
        context_features = self.context_encoder(spectrum)
        condition = self.environment_encoder(context_features)
        gamma, beta = self.film(condition).chunk(2, dim=1)
        conditioned = (
            (1.0 + 0.5 * torch.tanh(gamma))[:, :, None, None] * context_features
            + beta[:, :, None, None]
        )
        masks = torch.softmax(self.mask_generator(conditioned), dim=1)
        modulation_spectrum = masks[:, 0] * spectrum
        embedding = self.branch(modulation_spectrum)
        return {
            "logits": self.classifier(embedding),
            "embedding": embedding,
            "spectrum": spectrum,
            "features": context_features,
            "mod_spectrum": modulation_spectrum,
            "masks": masks,
            "condition": condition,
        }


class _TensorFlowSamePad2d(nn.Module):
    """Stride-one TensorFlow ``padding='same'`` for even-sized kernels."""

    def __init__(self, kernel_size: tuple[int, int]):
        super().__init__()
        height, width = kernel_size
        self.padding = (
            (width - 1) // 2,
            width - 1 - (width - 1) // 2,
            (height - 1) // 2,
            height - 1 - (height - 1) // 2,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.pad(values, self.padding)


class MCLDNNReimplementation(nn.Module):
    """Literature-faithful PyTorch reimplementation of MCLDNN.

    Audited against the authors' public Keras implementation at
    ``MCLDNN_AUDITED_COMMIT``.  The three streams, 50/50/100 convolutional
    widths, two 128-unit LSTMs, two 128-unit SELU layers, and 0.5 dropout are
    retained.  Framework-specific initialization and checkpoint formats differ,
    so results from this class must be labeled "reimplementation", never
    "official implementation".
    """

    provenance = {
        "display_name": "MCLDNN literature-faithful reimplementation",
        "claim_level": "literature-faithful PyTorch reimplementation; not official weights",
        "paper_url": MCLDNN_PAPER_URL,
        "official_code_url": MCLDNN_OFFICIAL_CODE_URL,
        "audited_commit": MCLDNN_AUDITED_COMMIT,
        "material_differences": [
            "PyTorch implementation; official public code is Keras/TensorFlow",
            "randomly initialized; official Keras checkpoints are not loaded",
            "accepts any frame length of at least five samples",
        ],
    }

    def __init__(self, num_classes: int):
        super().__init__()
        self.joint_conv = nn.Sequential(
            _TensorFlowSamePad2d((2, 8)),
            nn.Conv2d(1, 50, kernel_size=(2, 8)),
            nn.ReLU(),
        )
        self.i_conv = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0.0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(),
        )
        self.q_conv = nn.Sequential(
            nn.ConstantPad1d((7, 0), 0.0),
            nn.Conv1d(1, 50, kernel_size=8),
            nn.ReLU(),
        )
        self.separate_fusion = nn.Sequential(
            _TensorFlowSamePad2d((1, 8)),
            nn.Conv2d(50, 50, kernel_size=(1, 8)),
            nn.ReLU(),
        )
        self.stream_fusion = nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=(2, 5)),
            nn.ReLU(),
        )
        self.temporal = nn.LSTM(
            input_size=100,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
        )
        self.dense1 = nn.Linear(128, 128)
        self.dense2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, num_classes)
        self.apply(self._initialize_like_keras)

    @staticmethod
    def _initialize_like_keras(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, parameter in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError("MCLDNN expects [batch, 2, time] I/Q tensors")
        if values.shape[-1] < 5:
            raise ValueError("MCLDNN requires at least five time samples")
        joint = self.joint_conv(values.unsqueeze(1))
        i_features = self.i_conv(values[:, 0:1])
        q_features = self.q_conv(values[:, 1:2])
        separate = self.separate_fusion(
            torch.stack((i_features, q_features), dim=2)
        )
        fused = self.stream_fusion(torch.cat((joint, separate), dim=1))
        sequence = fused.squeeze(2).transpose(1, 2)
        sequence, _ = self.temporal(sequence)
        embedding = self.dropout(F.selu(self.dense1(sequence[:, -1])))
        embedding = self.dropout(F.selu(self.dense2(embedding)))
        return {
            "logits": self.classifier(embedding),
            "embedding": embedding,
        }


class _CSSLNoiseLevelEstimator(nn.Module):
    """Three-layer FCN copied structurally from the official CSSL encoder."""

    def __init__(self):
        super().__init__()
        self.input_convolution = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.hidden_convolution = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.output_convolution = nn.Sequential(
            nn.Conv1d(32, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.input_convolution(values)
        values = self.hidden_convolution(values)
        return self.output_convolution(values)


class _CSSLResidualBlock(nn.Module):
    """Official two-convolution 1-D CSSL residual block."""

    def __init__(self, input_channels: int, channels: int, stride: int = 1):
        super().__init__()
        self.first = nn.Conv1d(
            input_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.second = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        if stride != 1 or input_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    input_channels,
                    channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(values)
        values = F.relu(self.first(values))
        values = self.second(values)
        return F.relu(values + residual)


class _CSSLOfficialEncoder1024(nn.Module):
    """Official CSSL-AMC encoder topology for 1024-sample RadioML frames."""

    def __init__(self):
        super().__init__()
        self.noise_level_estimator = _CSSLNoiseLevelEstimator()
        self.input_convolution = nn.Conv1d(
            4,
            64,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        # The official source defines this layer but does not call it in
        # ``ResNet.forward``.  Retaining the unused parameters preserves the
        # audited state-dict topology and parameter count.
        self.unused_input_norm = nn.BatchNorm1d(64)
        self.stage1 = nn.Sequential(
            _CSSLResidualBlock(64, 64),
            _CSSLResidualBlock(64, 64),
        )
        self.stage2 = nn.Sequential(
            _CSSLResidualBlock(64, 128, stride=2),
            _CSSLResidualBlock(128, 128),
        )
        self.readout = nn.Linear(128 * 512, 128)
        self.readout_norm = nn.BatchNorm1d(128)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        estimated_noise = self.noise_level_estimator(values)
        values = torch.cat((values, estimated_noise), dim=1)
        values = F.relu(self.input_convolution(values))
        values = self.stage1(values)
        values = self.stage2(values)
        values = values.flatten(start_dim=1)
        return F.relu(self.readout_norm(self.readout(values)))


class CSSLAMCSupervisedAdaptation(nn.Module):
    """Unified-budget supervised adaptation of the 2025 CSSL-AMC topology.

    The encoder and two-linear-layer classifier follow the authors' official
    PyTorch source at ``CSSL_AMC_AUDITED_COMMIT``.  The formal TVT comparison
    deliberately does not import checkpoints and does not execute CSSL's
    separate momentum-encoder contrastive pretraining stage.  Consequently,
    this class is an architecture adaptation under the common supervised
    budget, not a reproduction of the complete published training method.
    """

    provenance = {
        "display_name": "CSSL-AMC supervised adaptation",
        "claim_level": (
            "official-architecture supervised adaptation; not a reproduction "
            "of the complete two-stage CSSL method"
        ),
        "paper_url": CSSL_AMC_PAPER_URL,
        "official_code_url": CSSL_AMC_OFFICIAL_CODE_URL,
        "audited_commit": CSSL_AMC_AUDITED_COMMIT,
        "license": CSSL_AMC_LICENSE,
        "native_input": {
            "representation": "raw complex IQ represented as [batch, 2, 1024]",
            "sample_length": 1024,
            "native_classes": 24,
        },
        "retained_from_official_source": [
            "three-convolution FCN noise-level estimator",
            "four-channel concatenation of raw IQ and estimated noise",
            "two-stage 1-D residual encoder with [2, 2] blocks",
            "fixed 128x512 flattening readout for 1024-sample frames",
            "128-to-64-to-class two-linear-layer classifier",
            "officially defined but forward-unused input BatchNorm parameters",
        ],
        "material_differences": [
            "ten output classes replace the native 24-class RadioML head",
            "random initialization; no external checkpoint or weight import",
            (
                "single-stage paired-view supervised cross-entropy replaces "
                "momentum-encoder contrastive pretraining plus fine-tuning"
            ),
            "common optimizer, source split, epoch budget, and checkpoint rule",
            "dictionary output adapts the model to the local evaluation API",
        ],
    }

    def __init__(self, num_classes: int):
        super().__init__()
        if num_classes <= 1:
            raise ValueError("CSSL-AMC requires at least two output classes")
        self.encoder = _CSSLOfficialEncoder1024()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64, bias=True),
            nn.Linear(64, num_classes, bias=True),
        )

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError(
                "CSSL-AMC supervised adaptation expects [batch, 2, time]"
            )
        if values.shape[-1] != 1024:
            raise ValueError(
                "CSSL-AMC official RadioML topology requires 1024 samples"
            )
        return self.encoder(values)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encode(values)
        return {
            "logits": self.classifier(embedding),
            "embedding": embedding,
        }


class _IQFormerSTFT(nn.Module):
    """Official-shape Blackman STFT of the in-phase component.

    SciPy's default boundary extension with a 31-sample window and unit hop
    yields exactly one frame per input sample.  Explicit framing reproduces
    that alignment without adding SciPy to the model's forward pass.
    """

    def __init__(self):
        super().__init__()
        self.window_length = 31
        self.n_fft = 128
        self.frequency_bins = 32
        self.register_buffer(
            "window",
            torch.blackman_window(self.window_length, periodic=True),
            persistent=True,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        in_phase = values[:, 0]
        half = self.window_length // 2
        frames = F.pad(in_phase, (half, half)).unfold(
            dimension=-1,
            size=self.window_length,
            step=1,
        )
        frames = frames * self.window.to(device=values.device, dtype=values.dtype)
        spectrum = torch.fft.rfft(frames, n=self.n_fft, dim=-1)
        spectrum = spectrum / self.window.to(
            device=values.device,
            dtype=values.dtype,
        ).sum()
        # The public dataset wrapper casts a complex NumPy STFT via
        # ``torch.Tensor``, which retains its real component.  We make that
        # behavior explicit and avoid the associated warning.
        return spectrum[..., : self.frequency_bins].real.transpose(1, 2).unsqueeze(1)

    def estimated_real_operations(self, sample_length: int) -> float:
        fft = 2.5 * self.n_fft * math.log2(self.n_fft)
        windowing = 2.0 * self.window_length
        return float(sample_length * (fft + windowing))


class _IQConvEncoder(nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise1 = nn.Conv1d(channels, expansion, kernel_size=1)
        self.pointwise2 = nn.Conv1d(expansion, channels, kernel_size=1)
        self.layer_scale = nn.Parameter(torch.ones(channels, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(values)
        values = self.norm(values)
        values = F.gelu(self.pointwise1(values))
        return residual + self.layer_scale * self.pointwise2(values)


class _IQLocalRepresentation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.pointwise2 = nn.Conv1d(channels, channels, kernel_size=1)
        # The official local block keeps layer scaling enabled even when the
        # outer encoder is constructed with ``use_layer_scale=False``.
        self.layer_scale = nn.Parameter(torch.ones(channels, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(values)
        values = self.norm(values)
        values = F.gelu(self.pointwise1(values))
        return residual + self.layer_scale * self.pointwise2(values)


class _EfficientAdditiveAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.to_query = nn.Linear(channels, channels)
        self.to_key = nn.Linear(channels, channels)
        self.global_weight = nn.Parameter(torch.randn(channels, 1))
        self.scale = channels**-0.5
        self.project = nn.Linear(channels, channels)
        self.final = nn.Linear(channels, channels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = F.normalize(self.to_query(tokens), dim=-1)
        key = F.normalize(self.to_key(tokens), dim=-1)
        attention = F.normalize((query @ self.global_weight) * self.scale, dim=1)
        global_query = torch.sum(attention * query, dim=1, keepdim=True)
        return self.final(self.project(global_query * key) + query)


class _IQFeedForward(nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)
        self.first = nn.Conv1d(channels, expansion, kernel_size=1)
        self.second = nn.Conv1d(expansion, channels, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.second(F.gelu(self.first(self.norm(values))))


class _IQFormerEncoder(nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.local = _IQLocalRepresentation(channels)
        self.attention = _EfficientAdditiveAttention(channels)
        self.feed_forward = _IQFeedForward(channels, expansion)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.local(values)
        attended = self.attention(values.transpose(1, 2)).transpose(1, 2)
        values = values + attended
        return values + self.feed_forward(values)


class IQFormerInspiredClassifier(nn.Module):
    """Dependency-minimal IQFormer-inspired strong baseline.

    The public RML2016 setting is audited at ``IQFORMER_AUDITED_COMMIT``:
    stages ``[2, 3, 2]``, width 64, MLP ratio 4, two-layer bidirectional LSTM,
    one additive-attention block per stage, and fusion/LSTM dropout 0.2.

    This class is intentionally named ``Inspired`` because the STFT is
    calculated inside the PyTorch graph, the one-sample ``squeeze`` bug is
    corrected, and no official checkpoint equivalence test is available.
    """

    provenance = {
        "display_name": "IQFormer-inspired local baseline",
        "claim_level": "architecture-inspired local implementation; not an exact reproduction",
        "paper_url": IQFORMER_PAPER_URL,
        "official_code_url": IQFORMER_OFFICIAL_CODE_URL,
        "audited_commit": IQFORMER_AUDITED_COMMIT,
        "audited_rml2016_configuration": {
            "layers": [2, 3, 2],
            "embed_dims": [64, 64, 64],
            "mlp_ratio": 4,
            "drop_rate": 0.2,
            "drop_path_rate": 0.0,
            "use_layer_scale": False,
            "vit_num": 1,
        },
        "material_differences": [
            "internal differentiable PyTorch STFT replaces per-sample SciPy preprocessing",
            "dependency-free local blocks replace timm DropPath/trunc_normal helpers",
            "batch-safe squeeze replaces the public batch-size-one squeeze behavior",
            "randomly initialized; no official checkpoint equivalence test",
        ],
    }

    def __init__(self, num_classes: int):
        super().__init__()
        width = 64
        stem_width = width // 8
        fusion_width = width // 2
        self.iq_norm = nn.BatchNorm1d(2)
        self.stft = _IQFormerSTFT()
        self.stft_norm = nn.BatchNorm2d(1)
        self.iq_stem = nn.Sequential(
            nn.Conv1d(2, stem_width, kernel_size=5, padding=2, groups=2),
            nn.BatchNorm1d(stem_width),
        )
        self.stft_stem = nn.Sequential(
            nn.Conv2d(1, stem_width, kernel_size=(32, 1)),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(stem_width * 2, fusion_width, kernel_size=1),
            nn.BatchNorm1d(fusion_width),
            nn.GELU(),
            nn.Conv1d(fusion_width, fusion_width, kernel_size=1),
            nn.Dropout(0.2),
        )
        self.temporal = nn.LSTM(
            input_size=fusion_width,
            hidden_size=fusion_width,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )
        blocks: list[nn.Module] = []
        for stage_depth in (2, 3, 2):
            for block_index in range(stage_depth):
                if block_index == stage_depth - 1:
                    blocks.append(_IQFormerEncoder(width, width * 4))
                else:
                    blocks.append(_IQConvEncoder(width, width * 4))
        self.network = nn.Sequential(*blocks)
        self.norm = nn.BatchNorm1d(width)
        self.classifier = nn.Linear(width, num_classes)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def estimated_frontend_real_operations(self, sample_length: int) -> float:
        return self.stft.estimated_real_operations(sample_length)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """Return the audited 64-D representation before classification."""

        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError("IQFormer-inspired model expects [batch, 2, time]")
        iq = self.iq_stem(self.iq_norm(values))
        stft = self.stft_stem(self.stft_norm(self.stft(values))).squeeze(2)
        fused = self.fusion(torch.cat((iq, stft), dim=1))
        fused, _ = self.temporal(fused.transpose(1, 2))
        encoded = self.network(fused.transpose(1, 2))
        return self.norm(encoded).mean(dim=-1)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encode(values)
        return {
            "logits": self.classifier(embedding),
            "embedding": embedding,
        }


__all__ = [
    "BackboneClassifier",
    "SingleMaskClassifier",
    "DiagnosticDualMaskCEClassifier",
    "MCLDNNReimplementation",
    "CSSLAMCSupervisedAdaptation",
    "IQFormerInspiredClassifier",
    "MCLDNN_PAPER_URL",
    "MCLDNN_OFFICIAL_CODE_URL",
    "CSSL_AMC_PAPER_URL",
    "CSSL_AMC_OFFICIAL_CODE_URL",
    "CSSL_AMC_AUDITED_COMMIT",
    "CSSL_AMC_LICENSE",
    "IQFORMER_PAPER_URL",
    "IQFORMER_OFFICIAL_CODE_URL",
]
