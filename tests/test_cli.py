from __future__ import annotations

import unittest

from ai_council.cli import _smoke_max_tokens


class CliTests(unittest.TestCase):
    def test_smoke_budget_accounts_for_reasoning_effort(self) -> None:
        self.assertEqual(_smoke_max_tokens({}), 512)
        self.assertEqual(
            _smoke_max_tokens({"reasoning": {"effort": "medium"}}),
            1024,
        )
        self.assertEqual(
            _smoke_max_tokens({"reasoning": {"effort": "xhigh"}}),
            2048,
        )
        self.assertEqual(
            _smoke_max_tokens({"reasoning": {"max_tokens": 3000}}),
            3128,
        )


if __name__ == "__main__":
    unittest.main()
