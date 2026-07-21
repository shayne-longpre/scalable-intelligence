from __future__ import annotations

import unittest

from ai_council.config import ContextSpec
from ai_council.context import render_context_sections, render_private_memory
from ai_council.core import TranscriptEntry
from ai_council.transcript import Transcript


class ContextTests(unittest.TestCase):
    def test_private_memory_renders_compact_parsed_json(self) -> None:
        entry = TranscriptEntry(
            turn_id=3,
            phase="memory_update_1",
            speaker="P1",
            visibility="private",
            content="long raw text that should not be needed when JSON parsed",
            parsed={
                "participant_id": "P1",
                "phase": "memory_update_1",
                "question_summary": "Asked for an analogy test.",
                "answer_summary": "P2 gave a plausible but shallow analogy.",
                "assessment": "Good fluency, weak transfer.",
                "irrelevant_verbose_key": "omit this from compact memory",
            },
        )

        rendered = render_private_memory([entry])

        self.assertIn("Asked for an analogy test.", rendered)
        self.assertIn("Good fluency, weak transfer.", rendered)
        self.assertNotIn("long raw text", rendered)
        self.assertNotIn("irrelevant_verbose_key", rendered)

    def test_private_memory_context_can_limit_public_transcript(self) -> None:
        transcript = Transcript()
        transcript.append(
            TranscriptEntry(1, "round", "P1", "public", "old public turn")
        )
        transcript.append(
            TranscriptEntry(2, "round", "P2", "public", "recent public turn")
        )
        transcript.append(
            TranscriptEntry(
                3,
                "memory_update",
                "P1",
                "private",
                "",
                parsed={"participant_id": "P1", "assessment": "private compact assessment"},
            )
        )

        public_context, private_context = render_context_sections(
            transcript,
            "P1",
            ContextSpec(mode="private_memory", max_public_turns=1, max_private_turns=4),
            default_public_turns=80,
        )

        self.assertNotIn("old public turn", public_context)
        self.assertIn("recent public turn", public_context)
        self.assertIn("private compact assessment", private_context)

    def test_stream_context_hides_assessments_from_respondent(self) -> None:
        transcript = Transcript()
        stream_id = "interview:P1->P2"
        transcript.append(
            TranscriptEntry(
                1,
                "interview",
                "P1",
                "private",
                "question",
                metadata={"stream_id": stream_id, "interaction_role": "question"},
            )
        )
        transcript.append(
            TranscriptEntry(
                2,
                "interview",
                "P2",
                "private",
                "answer",
                metadata={"stream_id": stream_id, "interaction_role": "answer"},
            )
        )
        transcript.append(
            TranscriptEntry(
                3,
                "interview",
                "P1",
                "private",
                "private assessment",
                metadata={"stream_id": stream_id, "interaction_role": "assessment"},
            )
        )

        _, respondent_private_context = render_context_sections(
            transcript,
            "P2",
            ContextSpec(mode="private_memory", max_public_turns=0, max_private_turns=0),
            default_public_turns=80,
            stream_id=stream_id,
        )
        _, interviewer_private_context = render_context_sections(
            transcript,
            "P1",
            ContextSpec(mode="private_memory", max_public_turns=0, max_private_turns=0),
            default_public_turns=80,
            stream_id=stream_id,
        )

        self.assertIn("question", respondent_private_context)
        self.assertIn("answer", respondent_private_context)
        self.assertNotIn("private assessment", respondent_private_context)
        self.assertIn("private assessment", interviewer_private_context)

    def test_stream_only_scope_omits_unrelated_private_memory(self) -> None:
        transcript = Transcript()
        stream_id = "probe:P3->P1"
        transcript.append(
            TranscriptEntry(
                1,
                "probe_rounds",
                "P1",
                "private",
                "unrelated private note from P1's own probe",
                parsed={"next_round_plan": "answer P1's own probe"},
            )
        )
        transcript.append(
            TranscriptEntry(
                2,
                "probe_rounds",
                "P3",
                "private",
                "P3 routed probe",
                metadata={"stream_id": stream_id, "interaction_role": "question"},
            )
        )

        _, private_context = render_context_sections(
            transcript,
            "P1",
            ContextSpec(mode="private_memory", max_public_turns=0, max_private_turns=8),
            default_public_turns=80,
            stream_id=stream_id,
            private_scope="stream_only",
        )

        self.assertIn("P3 routed probe", private_context)
        self.assertNotIn("P1's own probe", private_context)


if __name__ == "__main__":
    unittest.main()
