from __future__ import annotations

import json

from ai_council.clients.base import ModelClient
from ai_council.config import ProviderSpec
from ai_council.core import ModelRequest, ModelResponse


class MockModelClient(ModelClient):
    def __init__(self, provider: ProviderSpec):
        self.provider = provider
        self._malformed_once_seen: set[tuple[str, str, str]] = set()
        self._missing_key_once_seen: set[tuple[str, str, str]] = set()
        self._empty_once_seen: set[tuple[str, str, str]] = set()

    def generate(self, request: ModelRequest) -> ModelResponse:
        participant_id = str(request.metadata.get("participant_id", "PX"))
        phase = str(request.metadata.get("phase", "phase"))
        all_participants = list(request.metadata.get("participants", []))
        require_json = bool(request.metadata.get("require_json", False))
        interaction_role = request.metadata.get("interaction_role")

        if _should_emit_empty_once(request, participant_id, phase, str(interaction_role), self._empty_once_seen):
            content = ""
        elif _should_emit_malformed_once(request, participant_id, phase, str(interaction_role), self._malformed_once_seen):
            content = '{"participant_id": "' + participant_id + '", "phase": "' + phase + '",'
        elif _should_emit_missing_key_once(request, participant_id, phase, str(interaction_role), self._missing_key_once_seen):
            content = json.dumps(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "ranking": _rotated_ranking(all_participants, participant_id),
                },
                indent=2,
            )
        elif require_json and interaction_role == "assessment":
            ranking = _rotated_ranking(all_participants, participant_id)
            respondent_id = str(request.metadata.get("respondent_id", ""))
            content = json.dumps(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "round_index": request.metadata.get("round_index"),
                    "interviewer_id": str(request.metadata.get("interviewer_id", participant_id)),
                    "respondent_id": respondent_id,
                    "target_participant_id": respondent_id,
                    "question_summary": "Mock diagnostic question.",
                    "answer_summary": "Mock respondent answer.",
                    "assessment": "Mock assessment for stream routing validation.",
                    "current_ranking": ranking,
                    "confidence": 0.4,
                    "criteria": ["question_quality", "reasoning_depth"],
                    "evidence": ["Mock evidence placeholder."],
                    "uncertainties": ["Mock client does not represent real capability."],
                    "updates": ["No substantive update in mock mode."],
                    "next_probe": "Ask a sharper follow-up.",
                },
                indent=2,
            )
        elif require_json and interaction_role == "round_ranking":
            ranking = _rotated_ranking(all_participants, participant_id)
            content = json.dumps(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "round_index": request.metadata.get("round_index"),
                    "interviewer_id": str(request.metadata.get("interviewer_id", participant_id)),
                    "ranking": ranking,
                    "confidence": 0.42,
                    "criteria": ["answer_quality", "calibration"],
                    "evidence": ["Mock round comparison evidence."],
                    "uncertainties": ["Mock client does not represent real capability."],
                    "updates": ["Mock round ranking update."],
                    "next_probe_strategy": ["Ask a more discriminating follow-up next round."],
                },
                indent=2,
            )
        elif require_json and interaction_role == "memory_update":
            content = json.dumps(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "round_index": request.metadata.get("round_index"),
                    "interviewer_id": str(request.metadata.get("interviewer_id", participant_id)),
                    "qa_assessment_summaries": [
                        {
                            "respondent_id": "mock",
                            "question_summary": "Mock probe.",
                            "answer_summary": "Mock answer.",
                            "assessment_summary": "Mock assessment.",
                            "evidence_to_remember": ["Mock evidence."],
                        }
                    ],
                    "ranking_summary": "Mock compact ranking summary.",
                    "uncertainties": ["Mock uncertainty."],
                    "next_round_plan": "Ask a sharper follow-up.",
                },
                indent=2,
            )
        elif require_json and interaction_role == "probe_comparison":
            respondents = [str(value) for value in request.metadata.get("respondents", [])]
            probe_id = str(request.metadata.get("probe_id", ""))
            content = json.dumps(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "round_index": request.metadata.get("round_index"),
                    "judge_id": participant_id,
                    "probe_id": probe_id,
                    "candidate_summaries": {
                        candidate_id: f"Mock comparative summary for {candidate_id}."
                        for candidate_id in respondents
                    },
                    "ordering": respondents,
                    "ties": [],
                    "confidence": 0.4,
                    "comparative_evidence": ["Mock answers were compared directly."],
                    "probe_validity": "limited",
                    "uncertainties": ["Mock output is not capability evidence."],
                },
                indent=2,
            )
        elif require_json and interaction_role == "wave_judgment":
            content = json.dumps(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "round_index": request.metadata.get("round_index"),
                    "judge_id": participant_id,
                    "ranking": all_participants,
                    "confidence": 0.4,
                    "criteria": ["reasoning", "calibration"],
                    "candidate_dossiers": {
                        candidate_id: f"Cumulative mock evidence for {candidate_id}."
                        for candidate_id in all_participants
                    },
                    "comparative_evidence": ["Mock probe comparisons were merged."],
                    "uncertainties": ["Mock output is not capability evidence."],
                    "uncertain_pairs": [all_participants[:2]] if len(all_participants) >= 2 else [],
                    "follow_up_candidates": all_participants[:2],
                    "follow_up_rationale": ["Exercise adaptive routing in mock runs."],
                    "next_probe_strategy": ["Use a different diagnostic angle."],
                },
                indent=2,
            )
        elif require_json and interaction_role == "evidence_card":
            candidate_id = str(request.metadata.get("candidate_id", ""))
            content = json.dumps(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "round_index": request.metadata.get("round_index"),
                    "judge_id": participant_id,
                    "candidate_id": candidate_id,
                    "ability_score": 50,
                    "confidence": 0.4,
                    "strengths": ["Mock reasoning evidence."],
                    "weaknesses": ["Mock answers are not capability evidence."],
                    "evidence": ["Completed every routed probe."],
                    "uncertainties": ["Mock client does not represent real capability."],
                    "summary": "A neutral mock evidence card for routing validation.",
                },
                indent=2,
            )
        elif require_json and interaction_role == "judge_ranking":
            content = json.dumps(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "round_index": request.metadata.get("round_index"),
                    "judge_id": participant_id,
                    "ranking": all_participants,
                    "scores": {
                        candidate_id: 60 - index
                        for index, candidate_id in enumerate(all_participants)
                    },
                    "confidence": 0.4,
                    "criteria": ["reasoning", "calibration"],
                    "evidence": ["Mock evidence cards were compared."],
                    "uncertainties": ["Mock client does not represent real capability."],
                    "uncertain_pairs": [all_participants[:2]] if len(all_participants) >= 2 else [],
                    "follow_up_candidates": all_participants[:2],
                    "follow_up_rationale": ["Exercise adaptive routing in mock runs."],
                },
                indent=2,
            )
        elif require_json:
            ranking = _rotated_ranking(all_participants, participant_id)
            content = json.dumps(
                {
                    "participant_id": participant_id,
                    "phase": phase,
                    "ranking": ranking,
                    "confidence": 0.41,
                    "criteria": ["question_quality", "reasoning_depth", "calibration"],
                    "evidence": [
                        f"{ranking[0] if ranking else participant_id} showed structured evaluation behavior."
                    ],
                    "uncertainties": ["Mock client does not represent real capability."],
                    "updates": ["No substantive update in mock mode."],
                    "next_evidence_needed": ["Run with real model clients."],
                    **(
                        {"final_reflection": "Mock final reflection for infrastructure validation."}
                        if "final" in phase
                        else {}
                    ),
                },
                indent=2,
            )
        else:
            content = (
                f"{participant_id}: In this mock {phase} turn, I would seek diagnostic evidence "
                "about question design, abstraction, calibration, and ability to update beliefs."
            )

        return ModelResponse(
            content=content,
            raw={"mock": True},
            usage={},
            model=request.model,
            provider=self.provider.name,
        )


def _rotated_ranking(participants: list[str], participant_id: str) -> list[str]:
    if not participants:
        return [participant_id]
    try:
        index = participants.index(participant_id)
    except ValueError:
        index = 0
    return participants[index:] + participants[:index]


def _should_emit_malformed_once(
    request: ModelRequest,
    participant_id: str,
    phase: str,
    interaction_role: str,
    seen: set[tuple[str, str, str]],
) -> bool:
    if request.metadata.get("repair_json"):
        return False
    if not request.params.get("mock_malformed_json_once"):
        return False
    key = (participant_id, phase, interaction_role)
    if key in seen:
        return False
    seen.add(key)
    return True


def _should_emit_missing_key_once(
    request: ModelRequest,
    participant_id: str,
    phase: str,
    interaction_role: str,
    seen: set[tuple[str, str, str]],
) -> bool:
    if request.metadata.get("repair_json"):
        return False
    if not request.params.get("mock_missing_json_key_once"):
        return False
    key = (participant_id, phase, interaction_role)
    if key in seen:
        return False
    seen.add(key)
    return True


def _should_emit_empty_once(
    request: ModelRequest,
    participant_id: str,
    phase: str,
    interaction_role: str,
    seen: set[tuple[str, str, str]],
) -> bool:
    if not request.params.get("mock_empty_visible_once"):
        return False
    key = (participant_id, phase, interaction_role)
    if key in seen:
        return False
    seen.add(key)
    return True
