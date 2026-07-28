from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.run_standard_experiment import available_model_factories  # noqa: E402
from vimd_amc.ablation import PAPER_ABLATION_PROTOCOLS  # noqa: E402
from vimd_amc.models.baselines import IQFormerInspiredClassifier  # noqa: E402
from vimd_amc.models.common import ModelConfig  # noqa: E402
from vimd_amc.models.iqformer_route_v4 import (  # noqa: E402
    DomainSpecificBatchNorm,
    SharedDSBNIQFormerEncoder,
    VIMDIQFormerRouteDSBNNet,
)


def _load_diagnostic_module():
    path = ROOT / "diagnostics" / "run_iqformer_route_v4_diagnostic.py"
    specification = importlib.util.spec_from_file_location(
        "vimd_v4_diagnostic_contract_module",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load VIMD-v4 diagnostic module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class IQFormerRouteV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.config = ModelConfig(
            spectral_channels=16,
            environment_dim=16,
            embedding_dim=48,
            n_fft=64,
            hop_length=16,
            dropout=0.0,
        )

    def test_only_batch_norm_affine_and_state_are_domain_specific(self) -> None:
        baseline = IQFormerInspiredClassifier(10)
        encoder = SharedDSBNIQFormerEncoder(10)
        self.assertEqual(
            len(
                [
                    module
                    for module in encoder.modules()
                    if isinstance(module, IQFormerInspiredClassifier)
                ]
            ),
            1,
        )
        shared_weight_types = (nn.Conv1d, nn.Conv2d, nn.Linear, nn.LSTM)
        baseline_modules = [
            type(module).__name__
            for module in baseline.modules()
            if isinstance(module, shared_weight_types)
        ]
        candidate_modules = [
            type(module).__name__
            for module in encoder.backbone.modules()
            if isinstance(module, shared_weight_types)
        ]
        self.assertEqual(candidate_modules, baseline_modules)

        accounting = encoder.normalization_parameter_accounting()
        self.assertEqual(accounting["batch_norm_layer_count"], 16)
        self.assertEqual(accounting["base_batch_norm_affine_parameters"], 1510)
        self.assertEqual(accounting["dsbn_extra_trainable_parameters"], 1510)
        self.assertEqual(
            accounting["single_domain_equivalent_parameters"],
            sum(parameter.numel() for parameter in baseline.parameters()),
        )
        for normalization in encoder.domain_norms():
            self.assertIsInstance(normalization, DomainSpecificBatchNorm)
            self.assertNotEqual(
                normalization.raw.weight.data_ptr(),
                normalization.route.weight.data_ptr(),
            )
            self.assertNotEqual(
                normalization.raw.running_mean.data_ptr(),
                normalization.route.running_mean.data_ptr(),
            )

    def test_raw_update_does_not_mutate_route_running_statistics(self) -> None:
        encoder = SharedDSBNIQFormerEncoder(10).train()
        first = encoder.domain_norms()[0]
        raw_before = first.raw.running_mean.detach().clone()
        route_before = first.route.running_mean.detach().clone()
        encoder.encode(torch.full((4, 2, 128), 3.0), domain="raw")
        self.assertFalse(torch.equal(first.raw.running_mean, raw_before))
        torch.testing.assert_close(first.route.running_mean, route_before)

    def test_forward_calls_one_shared_encoder_twice_and_route_is_nontrivial(
        self,
    ) -> None:
        model = VIMDIQFormerRouteDSBNNet(10, 9, self.config).eval()
        calls = []
        handle = model.shared_encoder.backbone.iq_stem[
            0
        ].register_forward_hook(lambda *_: calls.append(1))
        values = torch.randn(2, 2, 256)
        try:
            with torch.no_grad():
                output = model(values)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 2)
        self.assertEqual(tuple(output["logits"].shape), (2, 10))
        self.assertTrue(torch.all(output["route_gate"] >= 0.10))
        self.assertTrue(torch.all(output["route_gate"] <= 0.90))
        relative_difference = (
            (output["route_iq"] - values).flatten(1).norm(dim=1)
            / values.flatten(1).norm(dim=1).clamp_min(1e-8)
        )
        self.assertTrue(torch.all(relative_difference > 0.01))
        self.assertGreater(float(output["modulation_weight"].std()), 0.0)

    def test_gradients_reach_shared_weights_both_domains_and_route(self) -> None:
        model = VIMDIQFormerRouteDSBNNet(10, 9, self.config).train()
        output = model(torch.randn(3, 2, 256))
        output["logits"].square().mean().backward()
        shared_convolution = model.shared_encoder.backbone.iq_stem[0]
        self.assertIsNotNone(shared_convolution.weight.grad)
        self.assertGreater(float(shared_convolution.weight.grad.abs().sum()), 0.0)
        first_norm = model.shared_encoder.domain_norms()[0]
        self.assertGreater(float(first_norm.raw.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(first_norm.route.weight.grad.abs().sum()), 0.0)
        route_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.tri_mask.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(route_gradient, 0.0)

    def test_strict_reload_and_received_mixture_only_inference(self) -> None:
        candidate = VIMDIQFormerRouteDSBNNet(10, 9, self.config).eval()
        reloaded = VIMDIQFormerRouteDSBNNet(10, 9, self.config).eval()
        incompatible = reloaded.load_state_dict(
            candidate.state_dict(),
            strict=True,
        )
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        values = torch.randn(2, 2, 256)
        with torch.no_grad():
            first = candidate(values)["logits"]
            second = reloaded(values)["logits"]
        torch.testing.assert_close(first, second)
        self.assertEqual(
            tuple(inspect.signature(VIMDIQFormerRouteDSBNNet.forward).parameters),
            ("self", "values"),
        )
        self.assertFalse(any("teacher" in key for key in candidate.state_dict()))
        self.assertEqual(
            candidate.provenance["inference_inputs"],
            ["received_iq_mixture"],
        )

    def test_parameter_accounting_and_locked_registry(self) -> None:
        candidate = VIMDIQFormerRouteDSBNNet(10, 9, self.config)
        accounting = candidate.parameter_accounting()
        self.assertEqual(accounting["dsbn_extra_trainable_parameters"], 1510)
        self.assertEqual(accounting["shared_encoder_calls_per_forward"], 2)
        self.assertEqual(
            accounting["candidate_total_parameters"],
            sum(parameter.numel() for parameter in candidate.parameters()),
        )
        self.assertEqual(len(PAPER_ABLATION_PROTOCOLS), 8)
        self.assertNotIn("vimd_v4_dsbn", PAPER_ABLATION_PROTOCOLS)
        self.assertNotIn(
            "diagnostic_vimd_v4_iqformer_route_dsbn",
            available_model_factories(),
        )

    def test_diagnostic_is_locked_to_1024_cache_and_explicit_gate(self) -> None:
        diagnostic = _load_diagnostic_module()
        self.assertEqual(diagnostic.LOCKED_SAMPLE_LENGTH, 1024)
        self.assertEqual(
            diagnostic.LOCKED_CACHE_ROOT.name,
            "cache_factor_screening_1024_v1",
        )
        self.assertEqual(diagnostic.LOCKED_SEED, 17)
        self.assertEqual(diagnostic.LOCKED_EPOCHS, 12)
        self.assertEqual(
            diagnostic.LOCKED_PER_CLASS,
            {"train": 32, "validation": 10, "heldout_channel": 20},
        )
        self.assertTrue(
            diagnostic.PREREGISTERED_SUCCESS_GATE["all_required"]
        )
        parser_source = inspect.getsource(diagnostic.parse_arguments)
        self.assertIn("--execute-preregistered-diagnostic", parser_source)

    def test_v3_v4_initial_non_bn_weights_are_exactly_aligned(self) -> None:
        diagnostic = _load_diagnostic_module()
        torch.manual_seed(17)
        v3 = diagnostic.VIMDIQFormerRouteNet(10, 9, self.config)
        torch.manual_seed(17)
        v4 = VIMDIQFormerRouteDSBNNet(10, 9, self.config)
        audit = diagnostic._shared_non_bn_initialization_audit(v3, v4)
        self.assertTrue(
            audit["all_non_batch_norm_weights_single_and_initially_identical"]
        )
        self.assertEqual(audit["missing_in_v4"], [])
        self.assertEqual(audit["initial_values_mismatched"], [])


if __name__ == "__main__":
    unittest.main()
