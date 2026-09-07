"""Contract tests for the prospective M4--M7 TVT v2 protocol."""

from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stderr
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from typing import Any
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tvt_submission.formal_v2_contract import (
    EXPECTED_DEFAULT_FREEZE_SHA256,
    EXPECTED_GAIN_DEFINITION,
    EXPECTED_JUSTIFICATION_PATH,
    EXPECTED_JUSTIFICATION_SHA256,
    EXPECTED_OCCUPANCY_DEFINITION,
    EXPECTED_V3_8GB_CUDA_EXECUTION,
    EXPECTED_V3_8GB_FREEZE_SHA256,
    EXPECTED_V4_8GB_DUAL_CUDA_EXECUTION,
    EXPECTED_V4_8GB_DUAL_FREEZE_SHA256,
    FormalV2ContractError,
    V3_8GB_FREEZE,
    V4_8GB_DUAL_FREEZE,
    freeze_sha256,
    load_contract,
    loaded_freeze_identity,
    loaded_justification_identity,
    validate_contract,
    validate_justification_document,
)
from experiments.run_learning_curve_v2 import (
    _manifest_cache_digest,
    build_evidence,
    validate_learning_caches,
)
from experiments.run_formal_tvt_v2 import (
    validate_learning_prerequisite,
    write_bound_scientific_gate,
)
from vimd_amc.teacher_audit import OCCUPANCY_GAIN_MECHANISM_PROTOCOL
from vimd_amc.data.synthesis import SignalSynthesizer, SynthesisConfig
from vimd_amc.metrics import (
    PredictionBundle,
    ood_axis_calibrated_bootstrap,
)
from vimd_amc.reproducibility import (
    TVT_V2_SOURCE_PROFILE,
    source_tree_record,
)
from vimd_amc.standards import (
    TVT_V2_FACTOR_SPLITS,
    factor_isolated_split_policies_v2,
)
from tvt_submission.validate_v2_release import (
    _clean_retention_tests,
    _fit_grid_reasons,
    _prediction_grid_reasons,
    _spectral_support_occupancy,
    _view1_jammer_families_and_occupancy,
)
from standards.build_factor_cache import parse_args as parse_cache_args


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _mini_learning_fixture(
    root: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    Path,
    dict[str, Any],
    Path,
    Path,
]:
    contract = load_contract()
    contract["learning_curve"]["training_source_counts"] = [10]
    contract["learning_curve"]["models"] = ["a0_backbone"]
    contract["learning_curve"]["algorithm_seeds"] = [17]
    contract["learning_curve"]["fixed_formal_scale_regardless_of_curve"] = 10
    contract["cache"]["expected_split_source_counts"] = {
        "train": 10,
        "validation": 1,
        "id_test": 1,
        "hard_interference": 1,
    }
    cache_root = root / "cache"
    cache_root.mkdir(parents=True)
    split_counts = contract["cache"]["expected_split_source_counts"]
    next_source = 100
    source_ids: dict[str, list[int]] = {}
    for split, count in split_counts.items():
        source_ids[split] = list(
            range(next_source, next_source + count)
        )
        next_source += count
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "configuration": {
            "split_sizes": [
                [split, count] for split, count in split_counts.items()
            ],
            "master_seed": contract["cache"]["master_seed"],
            "sample_length": contract["cache"]["sample_length"],
            "guard_samples": contract["cache"]["guard_samples"],
            "evidence_designation": contract["cache"][
                "expected_designation"
            ],
        },
        "source_ids": source_ids,
        "files": {split: {} for split in split_counts},
    }
    manifest["cache_digest"] = _manifest_cache_digest(manifest)
    _write_json(cache_root / "manifest.json", manifest)
    bindings = validate_learning_caches([cache_root], contract)

    run_directory = root / "tvt_learning_curve_10_v2"
    model_root = run_directory / "models" / "a0_backbone_seed17"
    model_root.mkdir(parents=True)
    checkpoint_path = model_root / "model.pt"
    checkpoint_path.write_bytes(b"nonempty-checkpoint")
    result: dict[str, Any] = {
        "model": "a0_backbone",
        "seed": 17,
        "checkpoint": str(
            checkpoint_path.relative_to(run_directory)
        ),
        "training": {
            "checkpoint_fallback_used": False,
            "history": [
                {
                    "epoch": 1,
                    "training_macro_f1": 0.5,
                    "validation_macro_f1": 0.6,
                    "training_total_loss": 1.2,
                    "validation_loss": 1.0,
                    "checkpoint_selection_eligible": 1.0,
                }
            ],
            "checkpoint_selection": {
                "status": "eligible_validation_checkpoint_selected",
                "selected_checkpoint_eligible": True,
                "fallback_used": False,
                "selected_epoch": 1,
                "selected_validation_loss": 1.0,
                "eligible_checkpoint_count": 1,
            },
        },
        "regimes": {
            "id_test": {"macro_f1": 0.61},
            "hard_interference": {"macro_f1": 0.52},
        },
    }
    result_path = model_root / "result.json"
    _write_json(result_path, result)
    source_binding = source_tree_record(
        ROOT,
        ROOT / "experiments" / "run_standard_experiment.py",
        profile=TVT_V2_SOURCE_PROFILE,
    )
    run: dict[str, Any] = {
        "run_id": "tvt_learning_curve_10_v2",
        "runner": "experiments/run_standard_experiment.py",
        "status": "complete",
        "execution_status": "complete",
        "models": ["a0_backbone"],
        "seeds": [17],
        "cache_root": str(cache_root.resolve()),
        "cache_digest": bindings[0]["cache_digest"],
        "splits": split_counts,
        "checksums_verified": True,
        "component_validation": {
            split: {"maximum_residual": 0.0} for split in split_counts
        },
        "source_tree_execution_audit": {
            "unchanged": True,
            "reason": "unchanged",
            "start": source_binding,
            "end": source_binding,
        },
        "environment": {
            "python": "3.12-test",
            "torch": "2-test",
            "numpy": "2-test",
            "pandas": "2-test",
            "scipy": "1-test",
            "device": "cuda",
            "device_type": "cuda",
            "device_index": None,
            "cuda_available": True,
            "cuda_runtime": "12-test",
            "gpu": "fixture-gpu",
            "source_tree": source_binding,
        },
        "results": [result],
    }
    _write_json(run_directory / "run.json", run)
    return (
        contract,
        bindings,
        run_directory,
        run,
        result_path,
        checkpoint_path,
    )


class FormalV2ProtocolTest(unittest.TestCase):
    def test_v3_8gb_amendment_is_hash_bound_and_only_changes_execution(self) -> None:
        base = load_contract()
        amended = load_contract(V3_8GB_FREEZE)
        freeze_path, digest = loaded_freeze_identity(amended)
        self.assertEqual(freeze_path, V3_8GB_FREEZE.resolve())
        self.assertEqual(digest, EXPECTED_V3_8GB_FREEZE_SHA256)
        self.assertEqual(
            amended["schema_version"],
            "vimd_amc.tvt.formal_freeze.v3_8gb",
        )
        self.assertEqual(
            amended["experiment"]["training"]["batch_size"],
            EXPECTED_V3_8GB_CUDA_EXECUTION["per_device_batch_size"],
        )
        self.assertEqual(
            amended["execution_amendment"]["cuda_execution"],
            EXPECTED_V3_8GB_CUDA_EXECUTION,
        )
        self.assertEqual(
            amended["experiment"]["models"], base["experiment"]["models"]
        )
        self.assertEqual(
            amended["experiment"]["seeds"], base["experiment"]["seeds"]
        )
        self.assertEqual(
            amended["cache"], base["cache"]
        )

    def test_v4_8gb_dual_amendment_is_hash_bound_and_only_changes_execution(
        self,
    ) -> None:
        base = load_contract()
        amended = load_contract(V4_8GB_DUAL_FREEZE)
        freeze_path, digest = loaded_freeze_identity(amended)
        self.assertEqual(freeze_path, V4_8GB_DUAL_FREEZE.resolve())
        self.assertEqual(digest, EXPECTED_V4_8GB_DUAL_FREEZE_SHA256)
        self.assertEqual(
            amended["schema_version"],
            "vimd_amc.tvt.formal_freeze.v4_8gb_dual",
        )
        self.assertEqual(
            amended["experiment"]["training"]["batch_size"],
            EXPECTED_V4_8GB_DUAL_CUDA_EXECUTION["per_device_batch_size"],
        )
        self.assertEqual(
            amended["execution_amendment"]["cuda_execution"],
            EXPECTED_V4_8GB_DUAL_CUDA_EXECUTION,
        )
        self.assertEqual(
            amended["execution_amendment"]["cuda_execution"][
                "cuda_fit_concurrency"
            ],
            2,
        )
        self.assertEqual(
            amended["experiment"]["run_id"],
            "tvt_headline_1024_10seed_v4_8gb_dual",
        )
        self.assertEqual(
            amended["learning_curve"]["protocol_tag"], "v4_8gb_dual"
        )
        self.assertEqual(
            amended["experiment"]["models"], base["experiment"]["models"]
        )
        self.assertEqual(
            amended["experiment"]["seeds"], base["experiment"]["seeds"]
        )
        self.assertEqual(amended["cache"], base["cache"])

    def test_freeze_closes_m4_m5_m6_m7_contracts(self) -> None:
        contract = load_contract()
        self.assertEqual(
            contract["learning_curve"]["training_source_counts"],
            [10_000, 30_000, 100_000],
        )
        self.assertEqual(len(contract["experiment"]["seeds"]), 10)
        self.assertEqual(
            len(
                contract["experiment"]["confirmatory_family"][
                    "contrasts"
                ]
            ),
            3,
        )
        self.assertNotEqual(
            contract["cache"]["master_seed"],
            contract["experiment"]["statistics"]["bootstrap_seed"],
        )
        design = contract["prospective_statistical_design"]
        self.assertEqual(
            design["power_claim"],
            "none_no_eligible_pilot_variance_or_paired_dependence_estimates",
        )
        self.assertEqual(
            design["smallest_effect_size_of_interest"][
                "absolute_macro_f1_gain"
            ],
            0.01,
        )
        self.assertEqual(
            design["fixed_design"]["required_model_seed_fit_count"],
            120,
        )
        self.assertEqual(
            design["fixed_design"]["hard_interference_source_cluster_count"],
            5_000,
        )
        self.assertFalse(
            design["fixed_design"][
                "posthoc_seed_or_source_extension_permitted"
            ]
        )
        self.assertFalse(
            design["stopping_rule"][
                "test_metrics_used_for_checkpoint_or_stopping"
            ]
        )
        self.assertEqual(
            contract["experiment"]["scientific_release_gates"][
                "receiver_robustness"
            ]["nominal_reference_regime"],
            "hard_interference",
        )
        self.assertEqual(
            contract["experiment"]["scientific_release_gates"][
                "clean_retention"
            ]["profile_strata"],
            {
                "clean_retention_seen_acd": [0, 2, 3],
                "clean_retention_held_be": [1, 4],
            },
        )
        self.assertEqual(
            contract["experiment"]["scientific_release_gates"][
                "jammer_occupancy_directional_test"
            ]["occupancy_definition"],
            EXPECTED_OCCUPANCY_DEFINITION,
        )
        self.assertEqual(
            EXPECTED_OCCUPANCY_DEFINITION,
            OCCUPANCY_GAIN_MECHANISM_PROTOCOL["occupancy_definition"],
        )
        self.assertEqual(
            contract["experiment"]["scientific_release_gates"][
                "jammer_occupancy_directional_test"
            ]["gain_definition"],
            EXPECTED_GAIN_DEFINITION,
        )
        self.assertEqual(
            EXPECTED_GAIN_DEFINITION,
            OCCUPANCY_GAIN_MECHANISM_PROTOCOL["gain_definition"],
        )
        self.assertIn("inference view (view1)", EXPECTED_OCCUPANCY_DEFINITION)
        self.assertIn("mean over sources", EXPECTED_OCCUPANCY_DEFINITION)
        self.assertNotIn("mean over views", EXPECTED_OCCUPANCY_DEFINITION)

    def test_default_freeze_sha_and_exact_execution_grid_are_locked(
        self,
    ) -> None:
        contract = load_contract()
        freeze_path, digest = loaded_freeze_identity(contract)
        justification_path, justification_digest = (
            loaded_justification_identity(contract)
        )
        self.assertEqual(digest, EXPECTED_DEFAULT_FREEZE_SHA256)
        self.assertEqual(freeze_sha256(freeze_path), digest)
        self.assertEqual(
            justification_path,
            ROOT / EXPECTED_JUSTIFICATION_PATH,
        )
        self.assertEqual(
            justification_digest,
            EXPECTED_JUSTIFICATION_SHA256,
        )
        self.assertEqual(
            freeze_sha256(justification_path),
            justification_digest,
        )

        model_drift = deepcopy(contract)
        model_drift["experiment"]["models"][:2] = reversed(
            model_drift["experiment"]["models"][:2]
        )
        seed_drift = deepcopy(contract)
        seed_drift["experiment"]["seeds"] = list(
            reversed(seed_drift["experiment"]["seeds"])
        )
        split_drift = deepcopy(contract)
        split_drift["cache"]["expected_split_source_counts"][
            "combined_ood"
        ] = 4_999
        designation_drift = deepcopy(contract)
        designation_drift["cache"]["expected_designation"] = "drifted"
        run_id_drift = deepcopy(contract)
        run_id_drift["experiment"]["run_id"] = "posthoc_run"
        training_drift = deepcopy(contract)
        training_drift["experiment"]["training"]["epochs"] = 31
        gain_drift = deepcopy(contract)
        gain_drift["experiment"]["scientific_release_gates"][
            "jammer_occupancy_directional_test"
        ]["gain_definition"] = "posthoc source averaging"
        sesoi_drift = deepcopy(contract)
        sesoi_drift["prospective_statistical_design"][
            "smallest_effect_size_of_interest"
        ]["absolute_macro_f1_gain"] = 0.005
        stopping_drift = deepcopy(contract)
        stopping_drift["prospective_statistical_design"]["stopping_rule"][
            "outcome_dependent_amendment_permitted"
        ] = True
        justification_identity_drift = deepcopy(contract)
        justification_identity_drift["prospective_statistical_design"][
            "justification_sha256"
        ] = "0" * 64

        for candidate, message in (
            (model_drift, "experiment.models drifted"),
            (seed_drift, "experiment.seeds drifted"),
            (split_drift, "expected_split_source_counts drifted"),
            (designation_drift, "cache identity"),
            (run_id_drift, "runner, run_id"),
            (training_drift, "experiment.training drifted"),
            (gain_drift, "directional mechanism contract"),
            (sesoi_drift, "prospective statistical design"),
            (stopping_drift, "prospective statistical design"),
            (
                justification_identity_drift,
                "prospective statistical design",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    FormalV2ContractError,
                    message,
                ):
                    validate_contract(candidate)

    def test_prospective_justification_missing_or_byte_drift_fails_closed(
        self,
    ) -> None:
        contract = load_contract()
        source = ROOT / EXPECTED_JUSTIFICATION_PATH
        with tempfile.TemporaryDirectory(
            prefix="vimd_v2_prospective_justification_"
        ) as temporary:
            root = Path(temporary)
            copied = root / EXPECTED_JUSTIFICATION_PATH
            copied.parent.mkdir(parents=True)
            copied.write_bytes(source.read_bytes())
            path, digest = validate_justification_document(
                contract,
                root=root,
            )
            self.assertEqual(path, copied.resolve())
            self.assertEqual(digest, EXPECTED_JUSTIFICATION_SHA256)

            copied.write_bytes(copied.read_bytes() + b"\nbyte drift\n")
            with self.assertRaisesRegex(
                FormalV2ContractError,
                "justification SHA-256 drifted",
            ):
                validate_justification_document(contract, root=root)

            copied.unlink()
            with self.assertRaisesRegex(
                FormalV2ContractError,
                "missing or unreadable",
            ):
                validate_justification_document(contract, root=root)

    def test_learning_evidence_rejects_duplicate_fallback_and_missing_fits(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_v2_learning_negative_"
        ) as temporary:
            root = Path(temporary)
            for case, message in (
                ("duplicate", "duplicate learning-curve fit"),
                ("fallback", "fallback or ineligible checkpoint"),
                ("missing_result", "result.json is missing"),
                ("missing_checkpoint", "checkpoint.*file is missing"),
            ):
                with self.subTest(case=case):
                    case_root = root / case
                    (
                        contract,
                        bindings,
                        run_directory,
                        run,
                        result_path,
                        checkpoint_path,
                    ) = _mini_learning_fixture(case_root)
                    result = run["results"][0]
                    if case == "duplicate":
                        run["results"].append(deepcopy(result))
                        _write_json(run_directory / "run.json", run)
                    elif case == "fallback":
                        selection = result["training"][
                            "checkpoint_selection"
                        ]
                        selection["status"] = (
                            "fallback_final_state_no_eligible_"
                            "checkpoint_selected"
                        )
                        selection["selected_checkpoint_eligible"] = False
                        selection["fallback_used"] = True
                        result["training"]["checkpoint_fallback_used"] = True
                        _write_json(result_path, result)
                        _write_json(run_directory / "run.json", run)
                    elif case == "missing_result":
                        result_path.unlink()
                    else:
                        checkpoint_path.unlink()
                    with self.assertRaisesRegex(ValueError, message):
                        build_evidence(
                            contract=contract,
                            run_directories=[run_directory],
                            cache_bindings=bindings,
                            output=case_root / "evidence.json",
                        )

    def test_learning_evidence_binds_freeze_cache_and_scientific_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_v2_learning_binding_"
        ) as temporary:
            root = Path(temporary)
            (
                contract,
                bindings,
                run_directory,
                _,
                _,
                _,
            ) = _mini_learning_fixture(root)
            evidence_path = root / "learning_evidence.json"
            evidence = build_evidence(
                contract=contract,
                run_directories=[run_directory],
                cache_bindings=bindings,
                output=evidence_path,
            )
            validated = validate_learning_prerequisite(
                evidence_path,
                contract,
            )
            self.assertEqual(
                validated["freeze_sha256"],
                EXPECTED_DEFAULT_FREEZE_SHA256,
            )
            self.assertIn("10", validated["cache_bindings"])
            self.assertIn(
                "a0_backbone/seed17",
                validated["fit_artifacts"]["10"],
            )

            freeze_path, freeze_digest = loaded_freeze_identity(contract)
            gate_path = root / "v2_scientific_release_gate.json"
            gate = write_bound_scientific_gate(
                gate={"passed": False, "reasons": ["test_sentinel"]},
                output=gate_path,
                freeze_path=freeze_path,
                freeze_digest=freeze_digest,
                learning_evidence=evidence,
            )
            self.assertTrue(gate["freeze_identity_verified"])
            self.assertEqual(gate["freeze_sha256"], freeze_digest)
            self.assertEqual(
                json.loads(gate_path.read_text(encoding="utf-8")),
                gate,
            )

            tampered_evidence = deepcopy(evidence)
            tampered_evidence["freeze_sha256"] = "b" * 64
            _write_json(evidence_path, tampered_evidence)
            with self.assertRaisesRegex(
                ValueError,
                "incomplete or drifted",
            ):
                validate_learning_prerequisite(evidence_path, contract)

            tampered_curve = deepcopy(evidence)
            tampered_curve["curves"]["10"]["a0_backbone/seed17"][
                "id_test_macro_f1"
            ] = 0.99
            _write_json(evidence_path, tampered_curve)
            with self.assertRaisesRegex(
                ValueError,
                "curve binding failed",
            ):
                validate_learning_prerequisite(evidence_path, contract)

            _write_json(evidence_path, evidence)
            manifest_path = Path(bindings[0]["cache_manifest"])
            manifest_path.write_text(
                manifest_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "manifest 10 SHA-256 binding failed",
            ):
                validate_learning_prerequisite(evidence_path, contract)

    def test_learning_cache_rejects_self_digest_drift(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="vimd_v2_learning_cache_digest_"
        ) as temporary:
            root = Path(temporary)
            contract, _, _, _, _, _ = _mini_learning_fixture(root)
            manifest_path = root / "cache" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cache_digest"] = "0" * 64
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                ValueError,
                "cache manifest digest is invalid",
            ):
                validate_learning_caches([root / "cache"], contract)

    def test_freeze_rejects_seed_reuse_or_family_expansion(self) -> None:
        contract = load_contract()
        reused = deepcopy(contract)
        reused["experiment"]["statistics"]["bootstrap_seed"] = reused[
            "cache"
        ]["master_seed"]
        with self.assertRaisesRegex(
            FormalV2ContractError,
            "master seed and bootstrap seed",
        ):
            validate_contract(reused)
        expanded = deepcopy(contract)
        expanded["experiment"]["confirmatory_family"]["contrasts"].append(
            {
                "contrast_id": "posthoc",
                "reference": "a1_single_mask",
                "candidate": "a5_vimd_full",
            }
        )
        with self.assertRaisesRegex(
            FormalV2ContractError,
            "exactly A5-A0",
        ):
            validate_contract(expanded)

    def test_v2_cache_policy_has_exact_receiver_stress_axes(self) -> None:
        policies = factor_isolated_split_policies_v2()
        self.assertEqual(
            tuple(policy.split for policy in policies),
            TVT_V2_FACTOR_SPLITS,
        )
        by_split = {policy.split: policy for policy in policies}
        self.assertEqual(
            getattr(by_split["adc_10bit_agc"], "adc_bits"),
            10,
        )
        self.assertEqual(
            getattr(by_split["adc_12bit_agc"], "adc_bits"),
            12,
        )
        self.assertGreater(
            getattr(
                by_split["per_emitter_sync"],
                "per_emitter_cfo_norm_max",
            ),
            0.0,
        )
        self.assertGreater(
            getattr(
                by_split["per_emitter_sync"],
                "per_emitter_timing_offset_max_samples",
            ),
            0,
        )

    def test_non_v2_headline_rejects_receiver_stress_size_overrides(
        self,
    ) -> None:
        presets = (
            "micro",
            "screening",
            "headline",
            "learning_10k_v2",
            "learning_30k_v2",
        )
        receiver_stress_splits = (
            "adc_10bit_agc",
            "adc_12bit_agc",
            "per_emitter_sync",
        )
        for preset in presets:
            for split in receiver_stress_splits:
                with self.subTest(preset=preset, split=split):
                    stderr = StringIO()
                    with redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as raised:
                            parse_cache_args(
                                [
                                    "--output",
                                    "unused",
                                    "--preset",
                                    preset,
                                    "--split-size",
                                    f"{split}=7",
                                ]
                            )
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(
                        "require --preset headline_v2",
                        stderr.getvalue(),
                    )
        accepted = parse_cache_args(
            [
                "--output",
                "unused",
                "--preset",
                "headline_v2",
                "--split-size",
                "adc_10bit_agc=7",
            ]
        )
        self.assertIn(("adc_10bit_agc", 7), accepted.split_size)

    def test_occupancy_analysis_uses_only_view1_family_and_jammer(
        self,
    ) -> None:
        sample_count = 64
        time = np.arange(sample_count, dtype=np.float64)
        tone = np.exp(2j * np.pi * 5.0 * time / sample_count)
        impulse = np.zeros(sample_count, dtype=np.complex128)
        impulse[sample_count // 2] = 1.0

        def iq(values: np.ndarray) -> np.ndarray:
            return np.stack((values.real, values.imag), axis=0)

        jammer = np.stack(
            (
                np.stack((iq(impulse), iq(tone)), axis=0),
                np.stack((iq(tone), iq(impulse)), axis=0),
            ),
            axis=0,
        )
        records = [
            {
                "views": [
                    {"jammer_name": "tone"},
                    {"jammer_name": "chirp"},
                ]
            },
            {
                "views": [
                    {"jammer_name": "comb"},
                    {"jammer_name": "sweep"},
                ]
            },
        ]
        families, occupancy = _view1_jammer_families_and_occupancy(
            records,
            jammer,
        )
        expected_view1 = np.asarray(
            [
                _spectral_support_occupancy(jammer[0, 0]),
                _spectral_support_occupancy(jammer[1, 0]),
            ]
        )
        view2 = np.asarray(
            [
                _spectral_support_occupancy(jammer[0, 1]),
                _spectral_support_occupancy(jammer[1, 1]),
            ]
        )
        self.assertEqual(families.tolist(), ["tone", "comb"])
        np.testing.assert_allclose(occupancy, expected_view1)
        self.assertFalse(np.allclose(expected_view1, view2))
        self.assertFalse(
            np.allclose(occupancy, 0.5 * (expected_view1 + view2))
        )

    def test_quantized_per_emitter_stress_retains_component_identity(
        self,
    ) -> None:
        synthesizer = SignalSynthesizer(
            SynthesisConfig(sample_length=128)
        )
        time = np.arange(128, dtype=np.float64)
        sample = synthesizer.finalize_received_components(
            clean=np.exp(1j * 0.07 * time),
            raw_jammer=np.exp(1j * 0.19 * time),
            modulation="BPSK",
            source_seed=11,
            condition_seed=23,
            snr_db=-10.0,
            sir_db=-15.0,
            jammer_name="tone",
            jammer_labels=np.r_[1.0, np.zeros(8)],
            jammer_components=("tone",),
            channel_scenario="unit_test",
            channel_model="unit_test",
            speed_kmh=120.0,
            doppler_norm=0.0,
            overlap_profile="high",
            receiver_stress={
                "adc_bits": 10,
                "adc_full_scale": 3.0,
                "per_emitter_cfo_norm_max": 0.006,
                "per_emitter_timing_offset_max_samples": 3,
            },
        )
        residual = sample.mixture - (
            sample.clean
            + sample.jammer
            + sample.noise
            + sample.receiver_artifact
        )
        self.assertLess(float(np.max(np.abs(residual))), 2e-6)
        self.assertEqual(sample.metadata["adc_bits"], 10)
        self.assertEqual(
            sample.metadata["agc_mode"],
            "window_rms_before_adc",
        )
        self.assertTrue(
            sample.metadata[
                "receiver_stress_applied_pre_mixing_per_emitter"
            ]
        )

    def test_ood_calibration_uses_a0_opportunity_and_ci(self) -> None:
        labels = np.repeat(np.arange(2, dtype=np.int64), 30)
        sources_id = np.arange(100, 160, dtype=np.int64)
        sources_ood = np.arange(200, 260, dtype=np.int64)

        def bundle(
            errors_per_class: int,
            sources: np.ndarray,
        ) -> PredictionBundle:
            predictions = labels.copy()
            for class_index in range(2):
                selected = np.flatnonzero(labels == class_index)
                predictions[selected[:errors_per_class]] = 1 - class_index
            probabilities = np.full((len(labels), 2), 0.05)
            probabilities[np.arange(len(labels)), predictions] = 0.95
            return PredictionBundle(
                probabilities=probabilities,
                labels=labels,
                source_ids=sources,
                snr_db=np.zeros(len(labels)),
                sir_db=np.zeros(len(labels)),
            )

        seeds = range(10)
        result = ood_axis_calibrated_bootstrap(
            {seed: bundle(0, sources_id) for seed in seeds},
            {seed: bundle(12, sources_ood) for seed in seeds},
            {seed: bundle(2, sources_ood) for seed in seeds},
            draws=200,
            seed=991,
            recovery_fraction=0.25,
            opportunity_floor=0.005,
        )
        self.assertEqual(
            result["axis_classification"],
            "consequential_baseline_degradation",
        )
        self.assertTrue(result["axis_gate_passed"])
        self.assertEqual(result["algorithm_seed_count"], 10)

    def test_release_fit_grid_fails_closed_on_missing_fits(self) -> None:
        contract = load_contract()
        with tempfile.TemporaryDirectory(
            prefix="vimd_v2_fit_grid_"
        ) as temporary:
            reasons = _fit_grid_reasons(
                {
                    "results": [
                        {
                            "model": "a0_backbone",
                            "seed": 17,
                            "training": {
                                "checkpoint_selection": {
                                    "status": (
                                        "eligible_validation_checkpoint_selected"
                                    ),
                                    "selected_checkpoint_eligible": True,
                                    "fallback_used": False,
                                    "eligible_checkpoint_count": 1,
                                }
                            },
                            "checkpoint": "missing.pt",
                            "regimes": {},
                        }
                    ]
                },
                Path(temporary),
                contract,
            )
        self.assertTrue(
            any(reason.startswith("missing_fit_count:") for reason in reasons)
        )
        self.assertIn(
            "checkpoint_file_missing:a0_backbone/seed17",
            reasons,
        )

    def test_clean_retention_gate_is_derived_from_paired_predictions(
        self,
    ) -> None:
        contract = load_contract()
        contract["experiment"]["seeds"] = [17, 29]
        contract["experiment"]["statistics"]["bootstrap_draws"] = 100
        labels = np.tile(np.arange(2, dtype=np.int64), 20)
        profiles = np.tile(
            np.asarray([0, 1, 2, 4], dtype=np.int64),
            10,
        )
        sources = np.arange(len(labels), dtype=np.int64)

        def bundle(error_count: int) -> PredictionBundle:
            predictions = labels.copy()
            predictions[:error_count] = 1 - predictions[:error_count]
            probabilities = np.full((len(labels), 2), 0.05)
            probabilities[np.arange(len(labels)), predictions] = 0.95
            return PredictionBundle(
                probabilities=probabilities,
                labels=labels,
                source_ids=sources,
                snr_db=np.zeros(len(labels)),
                sir_db=np.full(len(labels), np.nan),
                target_profile_index=profiles,
            )

        baseline = bundle(6)
        candidate = bundle(2)

        def fake_load(
            run_root: Path,
            model: str,
            seed: int,
            regime: str,
            expected_cache_digest: str,
        ) -> PredictionBundle:
            del run_root, seed, regime, expected_cache_digest
            return (
                baseline
                if model == "cssl_amc_supervised_adaptation"
                else candidate
            )

        import tvt_submission.validate_v2_release as release

        original = release._load_bundle
        release._load_bundle = fake_load
        try:
            result = _clean_retention_tests(
                run_root=ROOT,
                cache_digest="a" * 64,
                contract=contract,
            )
        finally:
            release._load_bundle = original
        self.assertTrue(result["all_strata_complete"])
        self.assertTrue(result["all_strata_passed"])
        self.assertEqual(
            set(result["strata"]),
            {
                "clean_retention_seen_acd",
                "clean_retention_held_be",
            },
        )

    def test_prediction_grid_is_bound_to_cache_and_validation_split(
        self,
    ) -> None:
        contract = load_contract()
        contract["experiment"]["models"] = ["a0_backbone"]
        contract["experiment"]["seeds"] = [17]
        contract["cache"]["expected_split_source_counts"] = {
            "train": 4,
            "validation": 4,
        }
        digest = "b" * 64
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        sources = np.asarray([10, 11, 12, 13], dtype=np.int64)
        profiles = np.asarray([0, 1, 2, 3], dtype=np.int64)
        snr = np.asarray([0.0, -5.0, -10.0, -15.0])
        sir = np.asarray([0.0, -5.0, -10.0, -15.0])
        probabilities = np.asarray(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
            ]
        )
        with tempfile.TemporaryDirectory(
            prefix="vimd_v2_prediction_grid_"
        ) as temporary:
            root = Path(temporary)
            cache_root = root / "cache"
            cache_split = cache_root / "validation"
            cache_split.mkdir(parents=True)
            for name, values in {
                "source_id": sources,
                "label": labels,
                "snr_db": snr,
                "sir_db": sir,
                "target_profile_index": profiles,
            }.items():
                np.save(cache_split / f"{name}.npy", values)
            model_root = root / "models" / "a0_backbone_seed17"
            model_root.mkdir(parents=True)
            checkpoint = model_root / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            result = {
                "model": "a0_backbone",
                "seed": 17,
                "checkpoint": str(checkpoint.relative_to(root)),
                "regimes": {"validation": {}},
            }
            (model_root / "result.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )
            prediction = model_root / "predictions_validation.npz"
            np.savez_compressed(
                prediction,
                probabilities=probabilities,
                labels=labels,
                source_ids=sources,
                snr_db=snr,
                sir_db=sir,
                target_profile_index=profiles,
                cache_digest=np.asarray(digest),
                split=np.asarray("validation"),
            )
            reasons = _prediction_grid_reasons(
                run={"results": [result]},
                run_root=root,
                cache_root=cache_root,
                cache_digest=digest,
                contract=contract,
            )
            self.assertEqual(reasons, [])
            with np.load(prediction, allow_pickle=False) as archive:
                corrupted = {
                    name: np.asarray(archive[name])
                    for name in archive.files
                }
            corrupted["cache_digest"] = np.asarray("c" * 64)
            np.savez_compressed(prediction, **corrupted)
            reasons = _prediction_grid_reasons(
                run={"results": [result]},
                run_root=root,
                cache_root=cache_root,
                cache_digest=digest,
                contract=contract,
            )
            self.assertTrue(
                any(
                    reason.startswith("prediction_artifact_invalid:")
                    for reason in reasons
                )
            )


if __name__ == "__main__":
    unittest.main()
