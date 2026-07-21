from __future__ import annotations

import unittest

from ai_council.json_tools import extract_json_object


class JsonToolsTests(unittest.TestCase):
    def test_extracts_fenced_json(self) -> None:
        parsed = extract_json_object('```json\n{"a": 1, "b": ["x"]}\n```')
        self.assertEqual(parsed, {"a": 1, "b": ["x"]})

    def test_extracts_json_embedded_in_prose(self) -> None:
        parsed = extract_json_object('Here is the judgment:\n{"ranking": ["P1"], "confidence": 0.5}\nDone.')
        self.assertEqual(parsed["ranking"], ["P1"])

    def test_handles_braces_inside_strings(self) -> None:
        parsed = extract_json_object('{"claim": "Use {braces} literally", "ok": true}')
        self.assertEqual(parsed["claim"], "Use {braces} literally")


if __name__ == "__main__":
    unittest.main()
