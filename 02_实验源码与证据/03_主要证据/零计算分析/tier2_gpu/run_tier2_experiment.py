"""Independent Tier-2 launcher for the prospective TVT expansion arms.

This wrapper deliberately leaves the sealed cache, V4/V4R runner, checkpoints,
and composite untouched.  It reuses the standard training/evaluation engine but
marks the cache contract exploratory in memory, optionally wraps only the
training dataset with a deterministic 10% jammer-free component re-mix, and
registers the lightweight received-I/Q sidecar candidate.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


class CounterfactualCleanCoverageDataset:
    """Replace a fixed fraction of training sources by audited clean re-mixes."""

    def __init__(self, base, fraction: float, *, selection_seed: int = 20260812):
        if not 0.0 < fraction < 1.0:
            raise ValueError("counterfactual fraction must lie strictly between zero and one")
        reciprocal = round(1.0 / fraction)
        if abs(fraction - 1.0 / reciprocal) > 1e-12:
            raise ValueError("fraction must be an exact reciprocal for deterministic sampling")
        self.base = base
        self.fraction = float(fraction)
        self.reciprocal = int(reciprocal)
        self.modulations = base.modulations
        labels = np.asarray(base._arrays["label"], dtype=np.int64)
        rng = np.random.default_rng(selection_seed)
        self.selected = np.zeros(len(base), dtype=bool)
        self.selected_counts: dict[int, int] = {}
        for class_index in np.unique(labels):
            indices = np.flatnonzero(labels == class_index)
            count = int(round(len(indices) * fraction))
            chosen = rng.permutation(indices)[:count]
            self.selected[chosen] = True
            self.selected_counts[int(class_index)] = count

    def __len__(self):
        return len(self.base)

    @staticmethod
    def _remix(view: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        clean = view["clean"].float()
        noise = view["noise"].float()
        artifact = view["receiver_artifact"].float()
        total = clean + noise + artifact
        power = total.square().sum(dim=0).mean().clamp_min(1e-12)
        scale = power.sqrt()
        remixed = dict(view)
        remixed["x"] = total / scale
        remixed["clean"] = clean / scale
        remixed["noise"] = noise / scale
        remixed["receiver_artifact"] = artifact / scale
        remixed["jammer"] = torch.zeros_like(view["jammer"])
        remixed["unexplained"] = (noise + artifact) / scale
        remixed["jam_labels"] = torch.zeros_like(view["jam_labels"])
        remixed["quality"] = view["quality"].clone()
        remixed["quality"][1] = 0.0
        remixed["quality_mask"] = view["quality_mask"].clone()
        remixed["quality_mask"][1] = 0.0
        remixed["sir_db"] = torch.zeros_like(view["sir_db"])
        remixed["overlap"] = torch.zeros_like(view["overlap"])
        return remixed

    def __getitem__(self, index: int):
        item = self.base[index]
        if not self.selected[index]:
            return item
        return {
            **item,
            "view1": self._remix(item["view1"]),
            "view2": self._remix(item["view2"]),
        }

    def source_ids(self):
        return self.base.source_ids()

    def close(self):
        return self.base.close()

    def __getattr__(self, name):
        return getattr(self.base, name)


class LightweightIQSidecarVIMD(nn.Module):
    """A5 plus a small raw-I/Q temporal branch; inference still uses mixture only."""

    supports_tri_mechanism = True
    provenance = {
        "display_name": "A5 with lightweight received-I/Q sidecar",
        "claim_level": "exploratory Tier-2 front-end remediation",
        "derived_from": "a5_vimd_full",
        "frozen_family_member": False,
        "inference_inputs": ["received_iq_mixture"],
    }

    def __init__(self, num_classes: int, num_jammers: int, config):
        from vimd_amc.models.vimd import VIMDNet

        super().__init__()
        self.base = VIMDNet(num_classes, num_jammers, config)
        side = 24
        self.iq_branch = nn.Sequential(
            nn.Conv1d(2, 12, kernel_size=9, padding=4, bias=False),
            nn.GroupNorm(3, 12),
            nn.SiLU(),
            nn.Conv1d(12, 12, kernel_size=7, padding=3, groups=12, bias=False),
            nn.Conv1d(12, side, kernel_size=1, bias=False),
            nn.GroupNorm(6, side),
            nn.SiLU(),
            nn.Conv1d(side, side, kernel_size=5, padding=2, groups=side, bias=False),
            nn.Conv1d(side, side, kernel_size=1, bias=False),
            nn.GroupNorm(6, side),
            nn.SiLU(),
        )
        self.iq_projector = nn.Sequential(
            nn.Linear(side * 3, side), nn.LayerNorm(side), nn.SiLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.embedding_dim + side, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(config.embedding_dim, num_classes)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """Preserve the VIMD mechanism-evaluation interface."""
        return self.base.encode(values)

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.base(values)
        sequence = self.iq_branch(values)
        pooled = torch.cat(
            (sequence.mean(-1), sequence.var(-1, unbiased=False).add(1e-6).sqrt(), sequence.amax(-1)),
            dim=1,
        )
        iq_embedding = self.iq_projector(pooled)
        embedding = self.fusion(torch.cat((output["embedding"], iq_embedding), dim=1))
        return {
            **output,
            "logits": output["logits"] + self.classifier(embedding),
            "embedding": embedding,
            "iq_embedding": iq_embedding,
        }


def _parse_wrapper_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--tier2-arm",
        choices=("coverage", "iq_sidecar", "capacity", "presence_gate"),
        required=True,
    )
    parser.add_argument("--counterfactual-fraction", type=float, default=0.10)
    return parser.parse_known_args(argv)


def main() -> None:
    wrapper, forwarded = _parse_wrapper_arguments(sys.argv[1:])
    import experiments.run_standard_experiment as standard

    original_inspect = standard.inspect_cache_contract
    original_factories = standard.available_model_factories

    def inspect_exploratory(path):
        contract = original_inspect(path)
        # The frozen manifest is about 2 GB.  A deep copy expands it to tens of
        # GB in memory, even though Tier-2 only changes one top-level marker.
        # Copy only the two dictionaries that are mutated and share every
        # immutable records/files/source_ids payload with the inspected
        # contract.
        manifest = dict(contract.manifest)
        manifest["configuration"] = dict(contract.manifest["configuration"])
        manifest["configuration"]["evidence_designation"] = (
            "exploratory_tier2_outside_frozen_confirmatory_family"
        )
        return replace(contract, manifest=manifest)

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_shared_manifest_datasets(contract, *, verify_checksums):
        """Memory-map every split while retaining one shared manifest object.

        CachedPairedAMCDataset normally reparses and retains the full root
        manifest once per split.  That is harmless for ordinary manifests but
        this cache's per-window audit records make its manifest about 2 GB.
        The contract has already performed all structural checks, so construct
        the same read-only datasets from its shared, inspected manifest.
        """
        from vimd_amc.standards.cache import CachedPairedAMCDataset

        datasets = {}
        try:
            for split, expected_size in contract.split_sizes.items():
                dataset = CachedPairedAMCDataset.__new__(CachedPairedAMCDataset)
                dataset.cache_root = contract.cache_root
                dataset.split = split
                dataset._manifest = contract.manifest
                dataset._arrays = {}
                for name, specification in contract.manifest["files"][split].items():
                    path = contract.cache_root / specification["path"]
                    if verify_checksums:
                        actual = sha256_file(path)
                        if actual != specification["sha256"]:
                            raise RuntimeError(
                                f"checksum mismatch for {path}: "
                                f"{actual} != {specification['sha256']}"
                            )
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    if list(array.shape) != specification["shape"]:
                        raise RuntimeError(f"shape mismatch for cache array: {path}")
                    dataset._arrays[name] = array
                dataset.size = int(dataset._arrays["source_id"].shape[0])
                dataset.modulations = contract.modulations
                if dataset.size != expected_size:
                    raise ValueError(f"{split} dataset length disagrees with manifest")
                datasets[split] = dataset
        except Exception:
            for dataset in datasets.values():
                dataset.close()
            raise
        standard.assert_disjoint_source_ids(
            *(datasets[split].source_ids() for split in contract.split_sizes)
        )
        return datasets

    def load_exploratory(contract, *, verify_checksums):
        datasets = load_shared_manifest_datasets(
            contract, verify_checksums=verify_checksums
        )
        if wrapper.tier2_arm in {"coverage", "iq_sidecar"}:
            for split in ("train", "validation"):
                datasets[split] = CounterfactualCleanCoverageDataset(
                    datasets[split], wrapper.counterfactual_fraction
                )
        return datasets

    def factories_exploratory():
        factories = original_factories()
        if wrapper.tier2_arm == "presence_gate":
            from analysis_zero_compute.tier2_gpu.presence_gated_vimd import register

            factories = register(factories)
        if wrapper.tier2_arm == "iq_sidecar":
            from vimd_amc.losses import VIMDLossWeights
            from vimd_amc.models.spectral import PhysicalTriMaskTeacher
            from vimd_amc.training import TrainingObjective

            def build(classes, jammers, config):
                return standard.BuiltStandardModel(
                    model=LightweightIQSidecarVIMD(classes, jammers, config),
                    teacher=PhysicalTriMaskTeacher(config),
                    objective=TrainingObjective(
                        name="tier2_h2_f2_coverage_plus_iq_sidecar",
                        loss_family="vimd",
                        use_mask_supervision=True,
                        use_jammer_auxiliary=True,
                        use_quality_auxiliary=True,
                        use_cross_condition_contrastive=True,
                        use_orthogonality=True,
                    ),
                    loss_weights=VIMDLossWeights(),
                )

            factories["tier2_h2_f2_iq_sidecar"] = build
        return factories

    standard.inspect_cache_contract = inspect_exploratory
    standard.load_cache_datasets = load_exploratory
    standard.available_model_factories = factories_exploratory
    sys.argv = [sys.argv[0], *forwarded]
    standard.main()

    # Add an explicit wrapper record without changing any model result.
    run_id = forwarded[forwarded.index("--run-id") + 1]
    output = Path(forwarded[forwarded.index("--output") + 1]).resolve()
    record = {
        "schema": "tvt_tier2_execution_context_v1",
        "evidence_class": "exploratory_tier2_outside_frozen_confirmatory_family",
        "tier2_arm": wrapper.tier2_arm,
        "counterfactual_fraction": (
            wrapper.counterfactual_fraction
            if wrapper.tier2_arm in {"coverage", "iq_sidecar"}
            else 0.0
        ),
        "counterfactual_definition": "renormalize(clean + noise + receiver_artifact)",
        "counterfactual_selection": "exact per-class fraction, fixed RNG seed 20260812",
        "checkpoint_validation_distribution": "same original/counterfactual ratio as training",
        "sealed_cache_modified": False,
        "sealed_composite_modified": False,
        "shared_manifest_loader": True,
    }
    path = output / run_id / "tier2_execution_context.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
