from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_probe_study import collect_run_specs


class AnalyzeProbeStudyTests(unittest.TestCase):
    def test_base_summary_can_be_extended_without_relisting_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_run = root / "old"
            new_run = root / "new"
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "cohort": "accepted",
                                "run_dir": str(old_run),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            specs = collect_run_specs(
                str(summary), [f"ceiling_extension={new_run}"]
            )
        self.assertEqual(
            specs,
            [
                {"cohort": "accepted", "run_dir": str(old_run)},
                {"cohort": "ceiling_extension", "run_dir": str(new_run)},
            ],
        )

    def test_extension_run_excludes_replayed_opening_probes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "extension"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps(
                    {"metadata": {"archived_opening_probe_count": 5}}
                ),
                encoding="utf-8",
            )
            specs = collect_run_specs(
                None,
                [],
                extension_values=[f"ceiling={run_dir}"],
            )
        self.assertEqual(
            specs,
            [
                {
                    "cohort": "ceiling",
                    "run_dir": str(run_dir),
                    "probe_sequence_min": 6,
                }
            ],
        )

    def test_duplicate_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "same"
            with self.assertRaisesRegex(ValueError, "duplicate run"):
                collect_run_specs(
                    None,
                    [f"first={run_dir}", f"second={run_dir}"],
                )


if __name__ == "__main__":
    unittest.main()
