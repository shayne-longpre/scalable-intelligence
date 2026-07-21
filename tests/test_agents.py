from __future__ import annotations

import unittest

from ai_council.agents import ParticipantAgent
from ai_council.clients.base import ModelClient
from ai_council.config import ModelSpec, ParticipantSpec, PhaseSpec
from ai_council.core import ModelRequest, ModelResponse
from ai_council.prompts import PromptLibrary


class CapturingClient(ModelClient):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(content="{}", model=request.model, provider="test")


class ParticipantAgentTests(unittest.TestCase):
    def test_repair_prompt_preserves_fixed_stage_identity(self) -> None:
        client = CapturingClient()
        agent = ParticipantAgent(
            spec=ParticipantSpec(
                id="J2",
                model="judge",
                system_prompt="independent_intelligence_judge",
            ),
            model=ModelSpec(name="judge", provider="test", model="test/judge"),
            client=client,
            prompts=PromptLibrary(),
            all_participants=["P1", "P2"],
        )
        phase = PhaseSpec(
            name="judge_ranking",
            kind="private_judgment",
            prompt="independent_judge_wave_judgment",
            require_json=True,
        )

        agent.repair_structured_json(
            phase,
            4,
            original_content='{"participant_id":"P2"',
            error="participant_id_mismatch",
            prompt_values={"participant_id": "P2", "judge_id": "J2"},
        )

        request = client.requests[0]
        repair_prompt = request.messages[1]["content"]
        self.assertIn('participant_id must be "J2"', repair_prompt)
        self.assertIn('"participant_id": "J2"', repair_prompt)
        self.assertIn('"judge_id": "J2"', repair_prompt)
        self.assertIn('"phase": "judge_ranking"', repair_prompt)
        self.assertIn('"round_index": 4', repair_prompt)
        self.assertEqual(request.metadata["participant_id"], "J2")
        self.assertEqual(request.metadata["phase"], "judge_ranking")
        self.assertEqual(request.metadata["round_index"], 4)


if __name__ == "__main__":
    unittest.main()
