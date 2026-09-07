"""S7 launcher: retrained permanently-zeroed-I/Q sidecar ablation.

Preregistration: docs/S7_IQZERO_SIDECAR_RETRAIN_PREREG.md (FROZEN 2026-08-21).

Identical to the tier2 iq_sidecar arm in cache, counterfactual re-mix,
objective, loss weights, and every training hyperparameter; the only change
is that the I/Q branch input is permanently zeroed at training AND
inference. This keeps capacity, fusion structure, and data identical while
removing the received-I/Q information channel, so the severe-held contrast
against A5 isolates the information contribution of the I/Q branch.

Usage:
    python run_tier2_iqzero_experiment.py --cache-root standards\\cache_factor_headline_1024_v2 \
        --models tier2_h2_f2_iq_sidecar_iqzero --seeds 17,29,43,71,101 \
        --epochs 30 --batch-size 16 --learning-rate 3e-4 --weight-decay 0.01 \
        --patience 8 --use-amp --device cuda \
        --output artifacts --run-id tier2_iq_sidecar_iqzero_v1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import torch
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis_zero_compute" / "tier2_gpu"))

import run_tier2_experiment as t2  # noqa: E402


class IQZeroLightweightIQSidecarVIMD(t2.LightweightIQSidecarVIMD):
    """Sidecar whose I/Q branch is permanently fed zeros.

    Architecture and parameter count (46,794) are identical to the parent;
    only the I/Q branch input is replaced, at training and inference alike.
    """

    provenance = {
        "display_name": "A5 with lightweight received-I/Q sidecar (I/Q permanently zeroed)",
        "claim_level": "exploratory retrained ablation outside the frozen family",
        "derived_from": "tier2_h2_f2_iq_sidecar",
        "frozen_family_member": False,
        "inference_inputs": ["spectral_mixture_only_iq_zeroed"],
    }

    def forward(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.base(values)
        sequence = self.iq_branch(torch.zeros_like(values))
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
    parser.add_argument("--counterfactual-fraction", type=float, default=0.10)
    return parser.parse_known_args(argv)


def main() -> None:
    wrapper, forwarded = _parse_wrapper_arguments(sys.argv[1:])
    import experiments.run_standard_experiment as standard

    original_inspect = standard.inspect_cache_contract
    original_factories = standard.available_model_factories

    def inspect_exploratory(path):
        contract = original_inspect(path)
        manifest = dict(contract.manifest)
        manifest["configuration"] = dict(contract.manifest["configuration"])
        manifest["configuration"]["evidence_designation"] = (
            "exploratory_retrained_ablation_outside_frozen_confirmatory_family"
        )
        return replace(contract, manifest=manifest)

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load_shared_manifest_datasets(contract, *, verify_checksums):
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
        for split in ("train", "validation"):
            datasets[split] = t2.CounterfactualCleanCoverageDataset(
                datasets[split], wrapper.counterfactual_fraction
            )
        return datasets

    def factories_exploratory():
        factories = original_factories()
        from vimd_amc.losses import VIMDLossWeights
        from vimd_amc.models.spectral import PhysicalTriMaskTeacher
        from vimd_amc.training import TrainingObjective

        def build(classes, jammers, config):
            return standard.BuiltStandardModel(
                model=IQZeroLightweightIQSidecarVIMD(classes, jammers, config),
                teacher=PhysicalTriMaskTeacher(config),
                objective=TrainingObjective(
                    name="tier2_h2_f2_coverage_plus_iq_sidecar_iqzero",
                    loss_family="vimd",
                    use_mask_supervision=True,
                    use_jammer_auxiliary=True,
                    use_quality_auxiliary=True,
                    use_cross_condition_contrastive=True,
                    use_orthogonality=True,
                ),
                loss_weights=VIMDLossWeights(),
            )

        factories["tier2_h2_f2_iq_sidecar_iqzero"] = build
        return factories

    standard.inspect_cache_contract = inspect_exploratory
    standard.load_cache_datasets = load_exploratory
    standard.available_model_factories = factories_exploratory
    sys.argv = [sys.argv[0], *forwarded]
    standard.main()

    run_id = forwarded[forwarded.index("--run-id") + 1]
    output = Path(forwarded[forwarded.index("--output") + 1]).resolve()
    record = {
        "schema": "s7_iqzero_sidecar_execution_context_v1",
        "preregistration": "docs/S7_IQZERO_SIDECAR_RETRAIN_PREREG.md",
        "evidence_class": "exploratory_retrained_ablation_outside_frozen_confirmatory_family",
        "variant": "tier2_h2_f2_iq_sidecar_iqzero",
        "iq_branch_input": "permanently zeroed at training and inference",
        "architecture_parent": "LightweightIQSidecarVIMD (46,794 parameters)",
        "counterfactual_fraction": wrapper.counterfactual_fraction,
        "counterfactual_definition": "renormalize(clean + noise + receiver_artifact)",
        "counterfactual_selection": "exact per-class fraction, fixed RNG seed 20260812",
        "sealed_cache_modified": False,
        "sealed_composite_modified": False,
        "shared_manifest_loader": True,
    }
    path = output / run_id / "s7_execution_context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
