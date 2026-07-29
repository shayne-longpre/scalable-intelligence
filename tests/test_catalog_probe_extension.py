from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ai_council.catalog_probe_extension import (
    build_catalog_probe_extension_report,
)


class CatalogProbeExtensionTests(unittest.TestCase):
    def test_report_compares_checkpoints_and_sums_repair_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            first = root / "first"
            repaired = root / "repaired"
            for run in (source, first, repaired):
                run.mkdir()
                _write_config(run, repair_source=first if run == repaired else None)
                _write_transcript(run)
            _write_analysis(source, [(5, ["P1", "P2", "P3"], 0.7, 0.4)])
            _write_analysis(
                repaired,
                [
                    (10, ["P1", "P3", "P2"], 0.8, 0.6),
                    (11, ["P1", "P2", "P3"], 0.82, 0.65),
                    (12, ["P1", "P2", "P3"], 0.81, 0.64),
                ],
            )
            _write_run_summary(source, calls=10, cost=1)
            _write_run_summary(first, calls=20, cost=2)
            _write_run_summary(repaired, calls=5, cost=3)

            summary = build_catalog_probe_extension_report(
                pairs=[("Judge", source, repaired)],
                output_dir=root / "report",
            )

            condition = summary["conditions"][0]
            self.assertAlmostEqual(
                condition["opening_delta"]["pairwise_accuracy"],
                0.1,
            )
            self.assertEqual(condition["lineage"]["model_calls"], 25)
            self.assertEqual(condition["lineage"]["reported_cost_usd"], 5)
            self.assertTrue(condition["lineage"]["runtime_sensitive"])
            self.assertEqual(
                [row["probe_count"] for row in condition["extension_checkpoints"]],
                [10, 11, 12],
            )
            self.assertEqual(len(condition["adaptive_targets"]), 2)
            self.assertTrue((root / "report" / "report_card.html").exists())

    def test_interrupted_repair_accounting_uses_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            interrupted = root / "interrupted"
            repaired = root / "repaired"
            for run in (source, interrupted, repaired):
                run.mkdir()
            _write_config(source, repair_source=None)
            _write_config(interrupted, repair_source=None)
            _write_config(repaired, repair_source=interrupted)
            _write_transcript(source)
            _write_transcript(repaired)
            _write_transcript(
                interrupted,
                direct_usage_costs=[0.25, 0.75],
            )
            _write_analysis(source, [(5, ["P1", "P2", "P3"], 0.7, 0.4)])
            _write_analysis(repaired, [(10, ["P1", "P2", "P3"], 0.8, 0.6)])
            _write_run_summary(source, calls=10, cost=1)
            _write_run_summary(repaired, calls=3, cost=2)

            summary = build_catalog_probe_extension_report(
                pairs=[("Judge", source, repaired)],
                output_dir=root / "report",
            )

            lineage = summary["conditions"][0]["lineage"]
            self.assertEqual(lineage["model_calls"], 5)
            self.assertEqual(lineage["reported_cost_usd"], 3)


def _write_config(run: Path, repair_source: Path | None) -> None:
    metadata = {"ceiling_extension_source_run": "source-five"}
    if repair_source is not None:
        metadata.update(
            {
                "repair_source_run": str(repair_source),
                "repair_parameter_overrides": {
                    "model/p2": {"max_tokens": 100}
                },
            }
        )
    (run / "config.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "participants": [
                    {"id": "P1", "model": "p1"},
                    {"id": "P2", "model": "p2"},
                    {"id": "P3", "model": "p3"},
                ],
                "models": {
                    "p1": {"model": "model/p1"},
                    "p2": {"model": "model/p2"},
                    "p3": {"model": "model/p3"},
                },
            }
        ),
        encoding="utf-8",
    )


def _write_transcript(
    run: Path,
    *,
    direct_usage_costs: list[float] | None = None,
) -> None:
    rows = [
        {
            "round_index": round_index,
            "metadata": {
                "interaction_role": "question",
                "respondents": ["P1", "P2"],
                "probe_sequence_number": 9 + round_index,
            },
        }
        for round_index in (2, 3)
    ]
    rows.extend(
        {
            "round_index": 1,
            "metadata": {
                "interaction_role": "answer",
                "provider": "openrouter",
                "finish_reason": "stop",
                "usage": {"cost": cost},
            },
        }
        for cost in direct_usage_costs or []
    )
    (run / "transcript.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_analysis(
    run: Path,
    rows: list[tuple[int, list[str], float, float]],
) -> None:
    judgments = []
    for probe_count, ranking, accuracy, tau in rows:
        judgments.append(
            {
                "judgment_probe_count": probe_count,
                "reported_score_subset": {
                    "ranking": ranking,
                    "kendall_tau": tau,
                    "spearman_rho": tau,
                    "pairwise_accuracy": accuracy,
                    "rank_score_r_squared": tau * tau,
                },
            }
        )
    (run / "analysis_summary.json").write_text(
        json.dumps({"prior_agreement": {"judgments": judgments}}),
        encoding="utf-8",
    )


def _write_run_summary(run: Path, *, calls: int, cost: float) -> None:
    (run / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "model_calls": calls,
                "reported_cost_usd": cost,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
