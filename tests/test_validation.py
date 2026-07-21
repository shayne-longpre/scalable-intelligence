from __future__ import annotations

import tempfile
import unittest

from ai_council.analysis import analyze_run
from ai_council.config import ExperimentConfig
from ai_council.core import TranscriptEntry
from ai_council.storage import RunStore
from ai_council.validation import revalidate_run


class ValidationTests(unittest.TestCase):
    def test_revalidate_flags_structured_value_errors(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "name": "revalidate_test",
                "providers": [{"name": "mock", "kind": "mock"}],
                "models": [{"name": "mock_model", "provider": "mock", "model": "mock:model"}],
                "participants": [{"id": "P1", "model": "mock_model"}],
                "protocol": {
                    "name": "revalidate_protocol",
                    "phases": [
                        {
                            "name": "final",
                            "kind": "private_judgment",
                            "prompt": "final_judgment",
                            "visibility": "private",
                            "require_json": True,
                            "required_keys": ["participant_id", "ranking", "confidence"],
                        }
                    ],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir, config)
            store.append_entry(
                TranscriptEntry(
                    turn_id=1,
                    phase="final",
                    speaker="P1",
                    visibility="private",
                    content='{"participant_id":"P2","ranking":["P2"],"confidence":"high"}',
                    parsed={"participant_id": "P2", "ranking": ["P2"], "confidence": "high"},
                )
            )
            summary = revalidate_run(store.run_dir)
            self.assertEqual(
                summary["codes"],
                {
                    "participant_id_mismatch": 1,
                    "unknown_ranking_ids": 1,
                    "missing_ranking_ids": 1,
                    "invalid_confidence": 1,
                },
            )
            analysis = analyze_run(store.run_dir)
            self.assertEqual(len(analysis["revalidation_findings"]), 4)


if __name__ == "__main__":
    unittest.main()
