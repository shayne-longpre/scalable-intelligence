from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ai_council.env import load_dotenv


class EnvTests(unittest.TestCase):
    def test_load_dotenv_does_not_override_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".env"
            path.write_text("AI_COUNCIL_TEST_ENV=from_file\n", encoding="utf-8")
            os.environ["AI_COUNCIL_TEST_ENV"] = "existing"
            try:
                loaded = load_dotenv(path)
                self.assertEqual(loaded, [])
                self.assertEqual(os.environ["AI_COUNCIL_TEST_ENV"], "existing")
            finally:
                os.environ.pop("AI_COUNCIL_TEST_ENV", None)

    def test_load_dotenv_strips_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".env"
            path.write_text("AI_COUNCIL_TEST_ENV='quoted value'\n", encoding="utf-8")
            os.environ.pop("AI_COUNCIL_TEST_ENV", None)
            try:
                loaded = load_dotenv(path)
                self.assertEqual(loaded, ["AI_COUNCIL_TEST_ENV"])
                self.assertEqual(os.environ["AI_COUNCIL_TEST_ENV"], "quoted value")
            finally:
                os.environ.pop("AI_COUNCIL_TEST_ENV", None)


if __name__ == "__main__":
    unittest.main()
