from __future__ import annotations

import unittest

from scripts.plot_catalog_ladder import _run_data


class CatalogLadderPlotTests(unittest.TestCase):
    def test_primary_plot_uses_opening_checkpoint(self) -> None:
        run = {
            "name": "catalog_sol",
            "participants": [
                {"id": "P1", "provider_model_id": "model/a"},
                {"id": "P2", "provider_model_id": "model/b"},
            ],
            "judges": [{"provider_model_id": "openai/gpt-5.6-sol"}],
            "prior_participant_scores": {"P1": 2.0, "P2": 1.0},
            "prior_reported_score_participants": ["P1", "P2"],
            "probe_budget_results": [
                {
                    "probe_count": 10,
                    "ranking": ["P1", "P2"],
                    "pairwise_accuracy": 1.0,
                },
                {
                    "probe_count": 11,
                    "ranking": ["P2", "P1"],
                    "pairwise_accuracy": 0.0,
                },
            ],
        }

        plotted = _run_data(run, {"model/a": "A", "model/b": "B"})

        self.assertEqual(
            [row["predicted_rank"] for row in plotted["points"]],
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
