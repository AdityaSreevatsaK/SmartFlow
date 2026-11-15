"""Offline report tests. No PyTorch, Transformers, GPU or model downloads."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "library_based" / "agent.py"


def load_agent():
    package = types.ModuleType("smartflow_test_package")
    package.__path__ = [str(SOURCE.parent)]
    utils = types.ModuleType("smartflow_test_package.utils")
    utils.sanitize_for_json = lambda value: value
    torch = types.ModuleType("torch")
    torch.bfloat16 = "mock-bfloat16"
    transformers = types.ModuleType("transformers")
    transformers.pipeline = Mock()
    spec = importlib.util.spec_from_file_location("smartflow_test_package.agent", SOURCE)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "smartflow_test_package": package,
        "smartflow_test_package.utils": utils,
        "torch": torch,
        "transformers": transformers,
    }):
        spec.loader.exec_module(module)
    return module


JOURNEYS = [{
    "truck_id": 7,
    "legs": [{
        "source": "Start Station",
        "target": "End Station",
        "move": 3,
        "dispatch_time": "09:15 AM",
    }],
}]


class DispatchReportTests(unittest.TestCase):
    def setUp(self):
        self.agent = load_agent()
        self.agent.pipeline.reset_mock()
        self.agent.pipeline.side_effect = None
        self.agent.pipeline.return_value = Mock(
            return_value=[{"generated_text": "<start_of_turn>model\n- Truck schedule confirmed"}]
        )

    def test_successful_briefing_uses_real_source_target_fields(self):
        report = self.agent.generate_dispatch_with_hf_agent(JOURNEYS)
        self.assertIn("Truck 7", report)
        self.assertIn("Start Station", report)
        self.assertIn("End Station", report)
        self.assertIn("09:15 AM", report)
        self.assertIn("Move 3 bikes", report)
        self.assertIn("Truck schedule confirmed", report)
        self.assertNotIn("N/A", report)

    def test_model_loader_does_not_execute_remote_custom_code(self):
        self.agent.generate_dispatch_with_hf_agent(JOURNEYS)
        self.agent.pipeline.assert_called_once_with(
            "text-generation",
            model="google/gemma-2b-it",
            device_map="auto",
            torch_dtype=self.agent.torch.bfloat16,
            trust_remote_code=False,
        )

    def test_failed_model_load_falls_back_to_correct_route(self):
        self.agent.pipeline.side_effect = RuntimeError("not installed or not available")
        report = self.agent.generate_dispatch_with_hf_agent(JOURNEYS)
        self.assertIn("Fallback", report)
        self.assertIn("Start Station", report)
        self.assertIn("End Station", report)
        self.assertIn("Move 3 bikes", report)
        self.assertNotIn("N/A", report)

    def test_empty_journeys_produce_report_without_dispatch_tickets(self):
        report = self.agent.generate_dispatch_with_hf_agent([])
        self.assertIn("Daily Rebalancing Report", report)
        self.assertNotIn("Dispatch Ticket", report)

    def test_legacy_names_still_accepted(self):
        older = [{"truck_id": 2, "legs": [{
            "src": "Legacy From", "tgt": "Legacy To", "move": 1
        }]}]
        report = self.agent.generate_dispatch_with_hf_agent(older)
        self.assertIn("Legacy From", report)
        self.assertIn("Legacy To", report)
        self.assertIn("Move 1 bike", report)

    def test_missing_optional_leg_fields_do_not_crash(self):
        source_only = [{"truck_id": 3, "legs": [{"source": "Only Source"}]}]
        self.agent.pipeline.side_effect = RuntimeError("offline")
        report = self.agent.generate_dispatch_with_hf_agent(source_only)
        self.assertIn("Only Source", report)
        self.assertIn("N/A", report)


if __name__ == "__main__":
    unittest.main()
