from __future__ import annotations

import unittest

from ai_council.output_health import inspect_output_health


class OutputHealthTests(unittest.TestCase):
    def test_extracts_openrouter_reasoning_token_details(self) -> None:
        health = inspect_output_health(
            "",
            {
                "completion_tokens": 750,
                "completion_tokens_details": {"reasoning_tokens": 749},
            },
        )
        self.assertFalse(health.has_visible_text)
        self.assertEqual(health.completion_tokens, 750)
        self.assertEqual(health.reasoning_tokens, 749)
        self.assertTrue(health.reasoning_dominated)

    def test_ignores_missing_reasoning_details(self) -> None:
        health = inspect_output_health("visible", {"completion_tokens": 12})
        self.assertTrue(health.has_visible_text)
        self.assertEqual(health.completion_tokens, 12)
        self.assertIsNone(health.reasoning_tokens)
        self.assertFalse(health.reasoning_dominated)


if __name__ == "__main__":
    unittest.main()
