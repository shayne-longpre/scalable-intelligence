from __future__ import annotations

import json
from dataclasses import dataclass

from ai_council.clients.base import ModelClient
from ai_council.config import ContextSpec, ModelSpec, ParticipantSpec, PhaseSpec
from ai_council.context import render_context_sections
from ai_council.core import ModelRequest, ModelResponse
from ai_council.prompts import PromptLibrary
from ai_council.transcript import Transcript


@dataclass
class ParticipantAgent:
    spec: ParticipantSpec
    model: ModelSpec
    client: ModelClient
    prompts: PromptLibrary
    all_participants: list[str]
    max_context_turns: int = 80
    context: ContextSpec = ContextSpec()

    def request_params_for_phase(self, phase: PhaseSpec) -> dict[str, object]:
        return {**self.model.params, **phase.model_params}

    def recovery_params_for_phase(self, phase: PhaseSpec) -> dict[str, object]:
        return {**phase.recovery_model_params, **self.model.recovery_params}

    def generate_turn(
        self,
        transcript: Transcript,
        phase: PhaseSpec,
        round_index: int | None = None,
        prompt_values: dict[str, object] | None = None,
        extra_context: str | None = None,
        model_params_override: dict[str, object] | None = None,
    ) -> ModelResponse:
        request = self.build_turn_request(
            transcript,
            phase,
            round_index,
            prompt_values=prompt_values,
            extra_context=extra_context,
            model_params_override=model_params_override,
        )
        return self.client.generate(request)

    def build_turn_request(
        self,
        transcript: Transcript,
        phase: PhaseSpec,
        round_index: int | None = None,
        prompt_values: dict[str, object] | None = None,
        extra_context: str | None = None,
        model_params_override: dict[str, object] | None = None,
    ) -> ModelRequest:
        values = {
            **(prompt_values or {}),
            "participant_id": self.spec.id,
            "phase": phase.name,
            "round_index": round_index,
            "participants": ", ".join(self.all_participants),
            "participants_json": json.dumps(self.all_participants),
        }
        system_prompt = self.prompts.render(
            self.spec.system_prompt,
            **values,
        )
        phase_prompt = self.prompts.render(
            phase.prompt,
            **values,
        )
        stream_id = str(values["stream_id"]) if values.get("stream_id") else None
        private_scope = str(values.get("context_scope") or "default")
        user_prompt = self._build_user_prompt(
            transcript,
            phase_prompt,
            phase,
            round_index,
            extra_context,
            stream_id=stream_id,
            private_scope=private_scope,
        )
        return ModelRequest(
            model=self.model.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            params={
                **self.request_params_for_phase(phase),
                **(model_params_override or {}),
            },
            metadata={
                **(prompt_values or {}),
                "participant_id": self.spec.id,
                "phase": phase.name,
                "round_index": round_index,
                "participants": self.all_participants,
                "require_json": phase.require_json,
            },
        )

    def repair_structured_json(
        self,
        phase: PhaseSpec,
        round_index: int | None,
        *,
        original_content: str,
        error: str,
        prompt_values: dict[str, object] | None = None,
    ) -> ModelResponse:
        fixed_stage_values = {
            **(prompt_values or {}),
            "participant_id": self.spec.id,
            "phase": phase.name,
            "round_index": round_index,
        }
        values = {
            **(prompt_values or {}),
            "participant_id": self.spec.id,
            "phase": phase.name,
            "round_index": round_index,
            "participants": ", ".join(self.all_participants),
            "participants_json": json.dumps(self.all_participants),
            "required_keys_json": json.dumps(phase.required_keys),
            "structured_json_error": error,
            "original_structured_response": original_content,
            "stage_values_json": json.dumps(fixed_stage_values, default=str),
        }
        system_prompt = self.prompts.render(self.spec.system_prompt, **values)
        user_prompt = self.prompts.render("structured_json_repair", **values)
        return self.client.generate(
            ModelRequest(
                model=self.model.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                params=_repair_params(
                    {
                        **self.request_params_for_phase(phase),
                        **self.recovery_params_for_phase(phase),
                    }
                ),
                metadata={
                    **(prompt_values or {}),
                    "participant_id": self.spec.id,
                    "phase": phase.name,
                    "round_index": round_index,
                    "participants": self.all_participants,
                    "require_json": True,
                    "repair_json": True,
                    "original_parse_error": error,
                },
            )
        )

    def _build_user_prompt(
        self,
        transcript: Transcript,
        phase_prompt: str,
        phase: PhaseSpec,
        round_index: int | None,
        extra_context: str | None,
        stream_id: str | None,
        private_scope: str,
    ) -> str:
        round_text = f"Round: {round_index}\n" if round_index is not None else ""
        extra_context_text = f"\nRelevant routed context:\n{extra_context}\n" if extra_context else ""
        public_context, private_context = render_context_sections(
            transcript,
            self.spec.id,
            self.context,
            default_public_turns=self.max_context_turns,
            stream_id=stream_id,
            private_scope=private_scope,
        )
        return f"""Current phase: {phase.name}
{round_text}Participants: {", ".join(self.all_participants)}

Phase instruction:
{phase_prompt}

Public transcript:
{public_context}

Your private notes, memories, and judgments:
{private_context}
{extra_context_text}

Respond as {self.spec.id}. Keep the response concise enough for a multi-agent transcript."""


def _repair_params(params: dict[str, object]) -> dict[str, object]:
    repaired = dict(params)
    repaired["temperature"] = 0
    max_tokens = _as_int(repaired.get("max_tokens"))
    repaired["max_tokens"] = max(max_tokens or 0, 1000)
    return repaired


def _as_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
