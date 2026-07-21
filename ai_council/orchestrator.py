from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from time import perf_counter

from ai_council.agents import ParticipantAgent
from ai_council.clients.base import ModelClient, ModelClientError
from ai_council.config import ExperimentConfig, PhaseSpec
from ai_council.core import ModelRequest, ModelResponse, TranscriptEntry, Visibility
from ai_council.json_tools import JsonExtractionError, extract_json_object
from ai_council.monitors import RuleBasedMonitor
from ai_council.prompts import PromptLibrary
from ai_council.storage import RunStore
from ai_council.transcript import Transcript


ROUND_ROBIN_ASSESSMENT_KEYS = [
    "participant_id",
    "phase",
    "round_index",
    "interviewer_id",
    "respondent_id",
    "question_summary",
    "answer_summary",
    "assessment",
    "confidence",
    "criteria",
    "evidence",
    "uncertainties",
    "updates",
    "next_probe",
]

ROUND_ROBIN_RANKING_KEYS = [
    "participant_id",
    "phase",
    "round_index",
    "interviewer_id",
    "ranking",
    "confidence",
    "criteria",
    "evidence",
    "uncertainties",
    "updates",
    "next_probe_strategy",
]

ROUND_ROBIN_MEMORY_KEYS = [
    "participant_id",
    "phase",
    "round_index",
    "interviewer_id",
    "qa_assessment_summaries",
    "ranking_summary",
    "uncertainties",
    "next_round_plan",
]

INDEPENDENT_JUDGE_EVIDENCE_KEYS = [
    "participant_id",
    "phase",
    "round_index",
    "judge_id",
    "candidate_id",
    "ability_score",
    "confidence",
    "strengths",
    "weaknesses",
    "evidence",
    "uncertainties",
    "summary",
]

INDEPENDENT_JUDGE_RANKING_KEYS = [
    "participant_id",
    "phase",
    "round_index",
    "judge_id",
    "ranking",
    "scores",
    "confidence",
    "criteria",
    "evidence",
    "uncertainties",
    "uncertain_pairs",
    "follow_up_candidates",
    "follow_up_rationale",
]

INDEPENDENT_JUDGE_COMPARISON_KEYS = [
    "participant_id",
    "phase",
    "round_index",
    "judge_id",
    "probe_id",
    "candidate_summaries",
    "ordering",
    "ties",
    "confidence",
    "comparative_evidence",
    "probe_validity",
    "uncertainties",
]

INDEPENDENT_JUDGE_WAVE_KEYS = [
    "participant_id",
    "phase",
    "round_index",
    "judge_id",
    "ranking",
    "confidence",
    "criteria",
    "candidate_dossiers",
    "comparative_evidence",
    "uncertainties",
    "uncertain_pairs",
    "follow_up_candidates",
    "follow_up_rationale",
    "next_probe_strategy",
]


class ExperimentViolationError(RuntimeError):
    """Raised when strict monitoring detects a protocol violation."""


class BudgetExceededError(RuntimeError):
    """Raised when a configured call or reported-cost budget stops the run."""


@dataclass(frozen=True)
class _TurnRequest:
    agent: ParticipantAgent
    phase: PhaseSpec
    round_index: int | None
    visibility: Visibility
    prompt_values: dict[str, object] | None = None
    extra_context: str | None = None
    metadata: dict[str, object] | None = None
    fallback_parsed: dict[str, object] | None = None


@dataclass(frozen=True)
class _PreparedCall:
    client: ModelClient
    request: ModelRequest
    model_ref: str


class _PartialBatchError(RuntimeError):
    def __init__(
        self,
        responses: list[ModelResponse | None],
        failures: list[tuple[int, Exception]],
    ) -> None:
        super().__init__(str(failures[0][1]))
        self.responses = responses
        self.failures = failures


@dataclass
class _ModelSpend:
    provider: str
    provider_model_id: str
    model_calls: int = 0
    reported_cost_usd: float = 0.0


class CouncilRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        clients: dict[str, ModelClient],
        store: RunStore,
    ):
        self.config = config
        self.clients = clients
        self.store = store
        self.transcript = Transcript()
        self.prompts = PromptLibrary(config.prompt_overrides)
        self.monitor = RuleBasedMonitor(
            identity_terms=config.monitor.identity_terms,
            strict=config.monitor.strict,
        )
        self.next_turn_id = 1
        self.model_calls = 0
        self.reported_cost_usd = 0.0
        self.model_spend: dict[str, _ModelSpend] = {}
        self.public_rounds_completed = 0
        self.participant_ids = [participant.id for participant in config.participants]
        self.agents = [
            ParticipantAgent(
                spec=participant,
                model=config.models[participant.model],
                client=clients[config.models[participant.model].provider],
                prompts=self.prompts,
                all_participants=self.participant_ids,
                max_context_turns=config.run.max_context_turns,
                context=config.context,
            )
            for participant in config.participants
        ]
        self.judges = [
            ParticipantAgent(
                spec=judge,
                model=config.models[judge.model],
                client=clients[config.models[judge.model].provider],
                prompts=self.prompts,
                all_participants=self.participant_ids,
                max_context_turns=config.run.max_context_turns,
                context=config.context,
            )
            for judge in config.judges
        ]

    def run(self) -> RunStore:
        started_at = datetime.now(timezone.utc)
        started_clock = perf_counter()
        try:
            for phase in self.config.protocol.phases:
                self._run_phase(phase)
        except Exception as exc:
            self._write_run_summary(
                started_at,
                started_clock,
                status="failed",
                error=exc,
            )
            raise
        self._write_run_summary(started_at, started_clock, status="completed")
        return self.store

    def _write_run_summary(
        self,
        started_at: datetime,
        started_clock: float,
        *,
        status: str,
        error: Exception | None = None,
    ) -> None:
        self.store.write_json(
            "run_summary.json",
            {
                "name": self.config.name,
                "status": status,
                "started_at": started_at.isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": perf_counter() - started_clock,
                "turns": len(self.transcript.entries),
                "participants": [agent.spec.id for agent in self.agents],
                "judges": [agent.spec.id for agent in self.judges],
                "max_parallel_calls": self.config.run.max_parallel_calls,
                "model_calls": self.model_calls,
                "reported_cost_usd": self.reported_cost_usd,
                "model_spend": {
                    model_ref: self.model_spend[model_ref]
                    for model_ref in sorted(self.model_spend)
                },
                **(
                    {"error": {"type": type(error).__name__, "message": str(error)}}
                    if error is not None
                    else {}
                ),
            },
        )

    def _run_phase(self, phase: PhaseSpec) -> None:
        if phase.kind == "independent_judge_ranking":
            self._run_independent_judge_ranking_phase(phase)
            return

        if phase.kind in {"private", "private_reflection", "private_judgment"}:
            for agent in self.agents:
                self._run_agent_turn(agent, phase, round_index=None, visibility="private")
            return

        if phase.kind in {"public", "public_round_robin", "round_robin"}:
            for round_index in range(1, phase.rounds + 1):
                for agent in self._public_turn_order():
                    self._run_agent_turn(agent, phase, round_index=round_index, visibility="public")
                self.public_rounds_completed += 1
            return

        if phase.kind == "interactive_discussion":
            self._run_interactive_discussion_phase(phase)
            return

        if phase.kind == "round_robin_probes":
            self._run_round_robin_probes_phase(phase)
            return

        if phase.kind == "public_test_matrix":
            self._run_test_matrix_phase(phase)
            return

        if phase.kind == "public_test_evaluation":
            self._run_test_evaluation_phase(phase)
            return

        if phase.kind == "separate_interviews":
            self._run_separate_interviews_phase(phase)
            return

        raise ValueError(f"unknown phase kind: {phase.kind}")

    def _run_independent_judge_ranking_phase(self, phase: PhaseSpec) -> None:
        if phase.probe_schedule:
            self._run_adaptive_judge_waves_phase(phase)
            return

        probe_phase = replace(
            phase,
            prompt=phase.question_prompt or "independent_judge_probe",
            require_json=False,
            required_keys=[],
        )
        answer_phase = replace(
            phase,
            prompt=phase.answer_prompt or "independent_judge_answer",
            require_json=False,
            required_keys=[],
        )
        evidence_phase = replace(
            phase,
            prompt=phase.assessment_prompt or "independent_judge_evidence_card",
            require_json=True,
            required_keys=INDEPENDENT_JUDGE_EVIDENCE_KEYS,
        )
        ranking_phase = replace(
            phase,
            prompt=phase.ranking_prompt or "independent_judge_ranking",
            require_json=True,
            required_keys=INDEPENDENT_JUDGE_RANKING_KEYS,
        )
        preauthored_probes = _load_preauthored_probes(phase.preauthored_probe_file)
        preauthored_answers = _load_preauthored_answers(
            phase.preauthored_answer_file,
            set(phase.preauthored_answer_participants),
            include_unavailable=phase.reuse_unavailable_answers,
            retry_unavailable_rounds=set(phase.retry_unavailable_rounds),
        )
        preauthored_evidence = _load_preauthored_evidence(
            phase.preauthored_evidence_file
        )
        preauthored_rankings = _load_preauthored_rankings(
            phase.preauthored_ranking_file
        )

        for judge in self.judges:
            evidence_cards: dict[str, TranscriptEntry] = {}
            active_candidates = list(self.agents)
            previous_ranking: TranscriptEntry | None = None
            for round_index in range(1, phase.rounds + 1):
                if round_index > 1 and len(active_candidates) < 2:
                    break
                probe_count = (
                    phase.probes_per_round
                    if round_index == 1
                    else phase.adaptive_probes_per_round
                )
                probe_entries: list[TranscriptEntry] = []
                candidate_ids = [candidate.spec.id for candidate in active_candidates]
                available_evidence = [
                    evidence_cards[candidate_id]
                    for candidate_id in candidate_ids
                    if candidate_id in evidence_cards
                ]
                evidence_turn_ids = [card.turn_id for card in available_evidence]
                if previous_ranking is not None:
                    evidence_turn_ids.append(previous_ranking.turn_id)
                generation_stage = (
                    "baseline_battery" if round_index == 1 else "adaptive_followup"
                )
                evaluation_stage = (
                    "common baseline battery"
                    if round_index == 1
                    else "selective adaptive follow-up"
                )
                evidence_rule = (
                    "No candidate answers have been observed yet; choose this item from "
                    "the battery's remaining diagnostic gaps."
                    if round_index == 1
                    else "Use the prior ranking and evidence cards below to target a real "
                    "unresolved comparison among these candidates."
                )
                for probe_number in range(1, probe_count + 1):
                    probe_id = _independent_judge_probe_id(
                        phase.name,
                        judge.spec.id,
                        round_index,
                        probe_number,
                    )
                    probe_metadata = {
                        "stream_id": probe_id,
                        "interaction_mode": "independent_judge_ranking",
                        "interaction_role": "question",
                        "interviewer": judge.spec.id,
                        "respondents": candidate_ids,
                        "probe_id": probe_id,
                        "probe_number": probe_number,
                        "probe_count": probe_count,
                        "generation_stage": generation_stage,
                        "evidence_turn_ids_available": evidence_turn_ids,
                        **(
                            {"prior_ranking_turn_id": previous_ranking.turn_id}
                            if previous_ranking is not None
                            else {}
                        ),
                    }
                    replay = preauthored_probes.get(
                        (judge.spec.id, round_index, probe_number)
                    )
                    if replay is not None:
                        probe_entry = self._append_preauthored_probe(
                            judge,
                            probe_phase,
                            round_index=round_index,
                            content=str(replay["content"]),
                            metadata=probe_metadata,
                            provenance=replay,
                        )
                    else:
                        probe_entry = self._run_agent_turn(
                            judge,
                            probe_phase,
                            round_index=round_index,
                            visibility="private",
                            prompt_values={
                                "stream_id": probe_id,
                                "context_scope": "stream_only",
                                "interaction_mode": "independent_judge_ranking",
                                "interaction_role": "question",
                                "interviewer_id": judge.spec.id,
                                "respondents": ", ".join(candidate_ids),
                                "respondents_json": _json_list(candidate_ids),
                                "probe_id": probe_id,
                                "probe_number": probe_number,
                                "probe_count": probe_count,
                                "evaluation_stage": evaluation_stage,
                                "probe_evidence_rule": evidence_rule,
                            },
                            extra_context=_format_independent_judge_probe_instruction(
                                judge.spec.id,
                                candidate_ids,
                                round_index,
                                probe_entries,
                                previous_ranking,
                                available_evidence,
                            ),
                            metadata=probe_metadata,
                        )
                    probe_entries.append(probe_entry)
                    if not probe_entries[-1].content.strip():
                        raise ExperimentViolationError(
                            f"judge {judge.spec.id} produced no visible text for {probe_id}"
                        )
                    if probe_entries[-1].metadata.get("finish_reason") == "length":
                        raise ExperimentViolationError(
                            f"judge {judge.spec.id} produced a truncated probe for {probe_id}"
                        )

                answers_by_candidate: dict[str, list[TranscriptEntry]] = {
                    candidate_id: [] for candidate_id in candidate_ids
                }
                answer_requests: list[_TurnRequest] = []
                answer_destinations: list[str] = []
                replayed_answers: list[TranscriptEntry] = []
                replayed_destinations: list[str] = []
                replay_prefix_open = True
                for candidate in active_candidates:
                    for probe_entry in probe_entries:
                        probe_id = str(probe_entry.metadata["probe_id"])
                        stream_id = f"{probe_id}:{candidate.spec.id}"
                        request = _TurnRequest(
                                agent=candidate,
                                phase=answer_phase,
                                round_index=round_index,
                                visibility="private",
                                prompt_values={
                                    "stream_id": stream_id,
                                    "context_scope": "stream_only",
                                    "interaction_mode": "independent_judge_ranking",
                                    "interaction_role": "answer",
                                    "interviewer_id": judge.spec.id,
                                    "respondent_id": candidate.spec.id,
                                    "probe_id": probe_id,
                                    "question_turn_id": probe_entry.turn_id,
                                },
                                extra_context=_format_round_robin_probe_for_answer(probe_entry),
                                metadata={
                                    "stream_id": stream_id,
                                    "interaction_mode": "independent_judge_ranking",
                                    "interaction_role": "answer",
                                    "interviewer": judge.spec.id,
                                    "respondent": candidate.spec.id,
                                    "probe_id": probe_id,
                                    "question_turn_id": probe_entry.turn_id,
                                },
                            )
                        replay = preauthored_answers.get(stream_id)
                        if replay_prefix_open and replay is not None:
                            _validate_preauthored_answer_source(request, replay)
                            replayed_answers.append(
                                self._append_preauthored_answer(request, replay)
                            )
                            replayed_destinations.append(candidate.spec.id)
                            continue
                        replay_prefix_open = False
                        answer_requests.append(request)
                        answer_destinations.append(candidate.spec.id)

                answer_entries = [
                    *replayed_answers,
                    *self._run_agent_turn_batch(
                        answer_requests,
                        validator=_validate_independent_answer,
                    ),
                ]
                for candidate_id, answer_entry in zip(
                    [*replayed_destinations, *answer_destinations],
                    answer_entries,
                    strict=True,
                ):
                    answers_by_candidate[candidate_id].append(answer_entry)

                judgment_counts = _independent_judgment_probe_counts(
                    phase,
                    round_index,
                    len(probe_entries),
                )
                primary_count = len(probe_entries)
                branch_cards: dict[int, dict[str, TranscriptEntry]] = {
                    count: {} for count in judgment_counts
                }
                evidence_requests: list[_TurnRequest] = []
                evidence_destinations: list[tuple[int, str]] = []
                replayed_evidence: list[TranscriptEntry] = []
                replayed_evidence_destinations: list[tuple[int, str]] = []
                evidence_replay_prefix_open = True
                for count in judgment_counts:
                    is_primary = count == primary_count
                    for candidate in active_candidates:
                        candidate_id = candidate.spec.id
                        prior_card = evidence_cards.get(candidate_id) if round_index > 1 else None
                        probes = probe_entries[:count]
                        answers = answers_by_candidate[candidate_id][:count]
                        evidence_stream_id = _independent_judge_evidence_stream_id(
                            phase.name,
                            judge.spec.id,
                            candidate_id,
                            count,
                            round_index,
                        )
                        request = _TurnRequest(
                                agent=judge,
                                phase=evidence_phase,
                                round_index=round_index,
                                visibility="private",
                                prompt_values={
                                    "stream_id": evidence_stream_id,
                                    "context_scope": "stream_only",
                                    "interaction_mode": "independent_judge_ranking",
                                    "interaction_role": "evidence_card",
                                    "judge_id": judge.spec.id,
                                    "candidate_id": candidate_id,
                                    "probe_count_used": count,
                                },
                                extra_context=_format_independent_judge_evidence_context(
                                    probes,
                                    answers,
                                    prior_card,
                                ),
                                metadata={
                                    "stream_id": evidence_stream_id,
                                    "interaction_mode": "independent_judge_ranking",
                                    "interaction_role": "evidence_card",
                                    "judge": judge.spec.id,
                                    "candidate": candidate_id,
                                    "judgment_probe_count": count,
                                    "judgment_probe_total": primary_count,
                                    "is_primary_judgment": is_primary,
                                    "answer_turn_ids": [answer.turn_id for answer in answers],
                                    "question_turn_ids": [probe.turn_id for probe in probes],
                                    "answer_stream_ids": [
                                        answer.metadata.get("stream_id") for answer in answers
                                    ],
                                    "question_stream_ids": [
                                        probe.metadata.get("stream_id") for probe in probes
                                    ],
                                },
                            )
                        replay = preauthored_evidence.get(evidence_stream_id)
                        if evidence_replay_prefix_open and replay is not None:
                            _validate_preauthored_evidence_source(request, replay)
                            replayed_evidence.append(
                                self._append_preauthored_evidence(request, replay)
                            )
                            replayed_evidence_destinations.append((count, candidate_id))
                            continue
                        evidence_replay_prefix_open = False
                        evidence_requests.append(request)
                        evidence_destinations.append((count, candidate_id))
                evidence_entries = [
                    *replayed_evidence,
                    *self._run_agent_turn_batch(evidence_requests),
                ]
                for (count, candidate_id), evidence_entry in zip(
                    [*replayed_evidence_destinations, *evidence_destinations],
                    evidence_entries,
                    strict=True,
                ):
                    branch_cards[count][candidate_id] = evidence_entry

                ranking_requests: list[_TurnRequest] = []
                replayed_rankings: list[TranscriptEntry] = []
                replayed_ranking_counts: list[int] = []
                fresh_ranking_counts: list[int] = []
                ranking_replay_prefix_open = True
                for count in judgment_counts:
                    cards_for_count = {
                        **evidence_cards,
                        **branch_cards[count],
                    }
                    ordered_cards = [
                        cards_for_count[candidate.spec.id] for candidate in self.agents
                    ]
                    ranking_stream_id = _independent_judge_ranking_stream_id(
                        phase.name,
                        judge.spec.id,
                        count,
                        round_index,
                    )
                    request = _TurnRequest(
                        agent=judge,
                        phase=ranking_phase,
                        round_index=round_index,
                        visibility="private",
                        prompt_values={
                            "stream_id": ranking_stream_id,
                            "context_scope": "stream_only",
                            "interaction_mode": "independent_judge_ranking",
                            "interaction_role": "judge_ranking",
                            "judge_id": judge.spec.id,
                            "max_adaptive_candidates": phase.max_adaptive_candidates,
                            "probe_count_used": count,
                        },
                        extra_context=_format_independent_judge_ranking_context(ordered_cards),
                        metadata={
                            "stream_id": ranking_stream_id,
                            "interaction_mode": "independent_judge_ranking",
                            "interaction_role": "judge_ranking",
                            "judge": judge.spec.id,
                            "participants": self.participant_ids,
                            "judgment_probe_count": count,
                            "judgment_probe_total": primary_count,
                            "is_primary_judgment": count == primary_count,
                            "evidence_card_turn_ids": [
                                card.turn_id for card in ordered_cards
                            ],
                            "evidence_card_stream_ids": [
                                card.metadata.get("stream_id") for card in ordered_cards
                            ],
                        },
                    )
                    replay = preauthored_rankings.get(ranking_stream_id)
                    if ranking_replay_prefix_open and replay is not None:
                        _validate_preauthored_ranking_source(request, replay)
                        replayed_rankings.append(
                            self._append_preauthored_ranking(request, replay)
                        )
                        replayed_ranking_counts.append(count)
                        continue
                    ranking_replay_prefix_open = False
                    ranking_requests.append(request)
                    fresh_ranking_counts.append(count)
                ranking_entries = [
                    *replayed_rankings,
                    *self._run_agent_turn_batch(ranking_requests),
                ]
                ranking_by_count = dict(
                    zip(
                        [*replayed_ranking_counts, *fresh_ranking_counts],
                        ranking_entries,
                        strict=True,
                    )
                )

                evidence_cards.update(branch_cards[primary_count])
                previous_ranking = ranking_by_count[primary_count]
                active_candidates = _adaptive_candidates(
                    previous_ranking,
                    self.agents,
                    phase.max_adaptive_candidates,
                )

    def _run_adaptive_judge_waves_phase(self, phase: PhaseSpec) -> None:
        probe_phase = replace(
            phase,
            prompt=phase.question_prompt or "adaptive_judge_probe",
            require_json=False,
            required_keys=[],
        )
        answer_phase = replace(
            phase,
            prompt=phase.answer_prompt or "independent_judge_answer",
            require_json=False,
            required_keys=[],
        )
        comparison_phase = replace(
            phase,
            prompt=phase.assessment_prompt or "independent_judge_probe_comparison",
            require_json=True,
            required_keys=INDEPENDENT_JUDGE_COMPARISON_KEYS,
        )
        judgment_phase = replace(
            phase,
            prompt=phase.ranking_prompt or "independent_judge_wave_judgment",
            require_json=True,
            required_keys=INDEPENDENT_JUDGE_WAVE_KEYS,
        )
        preauthored_probes = _load_preauthored_probes(phase.preauthored_probe_file)
        preauthored_answers = _load_preauthored_answers(
            phase.preauthored_answer_file,
            set(phase.preauthored_answer_participants),
            include_unavailable=phase.reuse_unavailable_answers,
            retry_unavailable_rounds=set(phase.retry_unavailable_rounds),
        )
        preauthored_comparisons = _load_preauthored_evidence(
            phase.preauthored_evidence_file
        )
        preauthored_judgments = _load_preauthored_rankings(
            phase.preauthored_ranking_file
        )
        total_probe_count = sum(phase.probe_schedule)

        for judge in self.judges:
            previous_judgment: TranscriptEntry | None = None
            active_candidates = list(self.agents)
            cumulative_probe_count = 0
            replay_prefix_open = True
            replayed_probe_ids: set[str] = set()
            for round_index, probe_count in enumerate(phase.probe_schedule, start=1):
                if round_index == 1 or phase.adaptive_targeting == "all":
                    active_candidates = list(self.agents)
                candidate_ids = [candidate.spec.id for candidate in active_candidates]
                generation_stage = (
                    "baseline_battery" if round_index == 1 else "adaptive_followup"
                )
                evidence_turn_ids = (
                    [previous_judgment.turn_id] if previous_judgment is not None else []
                )
                probe_entries: list[TranscriptEntry] = []
                for probe_number in range(1, probe_count + 1):
                    probe_sequence_number = cumulative_probe_count + probe_number
                    probe_id = _independent_judge_probe_id(
                        phase.name,
                        judge.spec.id,
                        round_index,
                        probe_number,
                    )
                    metadata = {
                        "stream_id": probe_id,
                        "interaction_mode": "independent_judge_ranking",
                        "interaction_role": "question",
                        "interviewer": judge.spec.id,
                        "respondents": candidate_ids,
                        "probe_id": probe_id,
                        "probe_number": probe_number,
                        "probe_count": probe_count,
                        "probe_sequence_number": probe_sequence_number,
                        "probe_schedule": phase.probe_schedule,
                        "generation_stage": generation_stage,
                        "evidence_turn_ids_available": evidence_turn_ids,
                        **(
                            {"prior_ranking_turn_id": previous_judgment.turn_id}
                            if previous_judgment is not None
                            else {}
                        ),
                    }
                    replay = preauthored_probes.get(
                        (judge.spec.id, round_index, probe_number)
                    )
                    if replay_prefix_open and replay is not None:
                        probe_entry = self._append_preauthored_probe(
                            judge,
                            probe_phase,
                            round_index=round_index,
                            content=str(replay["content"]),
                            metadata=metadata,
                            provenance=replay,
                        )
                        replayed_probe_ids.add(probe_id)
                    else:
                        replay_prefix_open = False
                        probe_entry = self._run_agent_turn(
                            judge,
                            probe_phase,
                            round_index=round_index,
                            visibility="private",
                            prompt_values={
                                "stream_id": probe_id,
                                "context_scope": "stream_only",
                                "interaction_mode": "independent_judge_ranking",
                                "interaction_role": "question",
                                "interviewer_id": judge.spec.id,
                                "respondents": ", ".join(candidate_ids),
                                "respondents_json": _json_list(candidate_ids),
                                "probe_id": probe_id,
                                "probe_number": probe_number,
                                "probe_count": probe_count,
                                "probe_evidence_rule": _adaptive_probe_evidence_rule(
                                    round_index
                                ),
                            },
                            extra_context=_format_adaptive_judge_probe_context(
                                judge.spec.id,
                                candidate_ids,
                                round_index,
                                probe_entries,
                                previous_judgment,
                            ),
                            metadata=metadata,
                        )
                    if not probe_entry.content.strip():
                        raise ExperimentViolationError(
                            f"judge {judge.spec.id} produced no visible text for {probe_id}"
                        )
                    if probe_entry.metadata.get("finish_reason") == "length":
                        raise ExperimentViolationError(
                            f"judge {judge.spec.id} produced a truncated probe for {probe_id}"
                        )
                    probe_entries.append(probe_entry)

                answers_by_probe: dict[str, dict[str, TranscriptEntry]] = {
                    str(probe.metadata["probe_id"]): {} for probe in probe_entries
                }
                answer_requests: list[_TurnRequest] = []
                answer_destinations: list[tuple[str, str]] = []
                replayed_answers: list[TranscriptEntry] = []
                replayed_destinations: list[tuple[str, str]] = []
                for probe_entry in probe_entries:
                    for candidate in active_candidates:
                        probe_id = str(probe_entry.metadata["probe_id"])
                        stream_id = f"{probe_id}:{candidate.spec.id}"
                        request = _TurnRequest(
                            agent=candidate,
                            phase=answer_phase,
                            round_index=round_index,
                            visibility="private",
                            prompt_values={
                                "stream_id": stream_id,
                                "context_scope": "stream_only",
                                "interaction_mode": "independent_judge_ranking",
                                "interaction_role": "answer",
                                "interviewer_id": judge.spec.id,
                                "respondent_id": candidate.spec.id,
                                "probe_id": probe_id,
                                "question_turn_id": probe_entry.turn_id,
                            },
                            extra_context=_format_round_robin_probe_for_answer(probe_entry),
                            metadata={
                                "stream_id": stream_id,
                                "interaction_mode": "independent_judge_ranking",
                                "interaction_role": "answer",
                                "interviewer": judge.spec.id,
                                "respondent": candidate.spec.id,
                                "probe_id": probe_id,
                                "question_turn_id": probe_entry.turn_id,
                                "probe_sequence_number": probe_entry.metadata.get(
                                    "probe_sequence_number"
                                ),
                            },
                        )
                        replay = preauthored_answers.get(stream_id)
                        destination = (probe_id, candidate.spec.id)
                        if probe_id in replayed_probe_ids and replay is not None:
                            _validate_preauthored_answer_source(request, replay)
                            replayed_answers.append(
                                self._append_preauthored_answer(request, replay)
                            )
                            replayed_destinations.append(destination)
                            continue
                        replay_prefix_open = False
                        answer_requests.append(request)
                        answer_destinations.append(destination)

                answer_entries = [
                    *replayed_answers,
                    *self._run_agent_turn_batch(
                        answer_requests,
                        validator=_validate_independent_answer,
                    ),
                ]
                for (probe_id, candidate_id), answer_entry in zip(
                    [*replayed_destinations, *answer_destinations],
                    answer_entries,
                    strict=True,
                ):
                    answers_by_probe[probe_id][candidate_id] = answer_entry

                comparison_requests = []
                replayed_comparisons: list[TranscriptEntry] = []
                fresh_comparison_positions: list[int] = []
                replayed_comparison_positions: list[int] = []
                for probe_entry in probe_entries:
                    probe_id = str(probe_entry.metadata["probe_id"])
                    presentation_order = _comparison_presentation_order(
                        candidate_ids,
                        phase.comparison_order,
                        phase.comparison_seed,
                        int(probe_entry.metadata.get("probe_sequence_number", 0)),
                    )
                    ordered_answers = [
                        answers_by_probe[probe_id][candidate_id]
                        for candidate_id in presentation_order
                    ]
                    stream_id = _adaptive_probe_comparison_stream_id(probe_id)
                    request = _TurnRequest(
                        agent=judge,
                        phase=comparison_phase,
                        round_index=round_index,
                        visibility="private",
                        prompt_values={
                            "stream_id": stream_id,
                            "context_scope": "stream_only",
                            "interaction_mode": "independent_judge_ranking",
                            "interaction_role": "probe_comparison",
                            "judge_id": judge.spec.id,
                            "probe_id": probe_id,
                            "respondents": candidate_ids,
                            "respondents_json": _json_list(candidate_ids),
                        },
                        extra_context=_format_probe_comparison_context(
                            probe_entry,
                            ordered_answers,
                        ),
                        metadata={
                            "stream_id": stream_id,
                            "interaction_mode": "independent_judge_ranking",
                            "interaction_role": "probe_comparison",
                            "judge": judge.spec.id,
                            "probe_id": probe_id,
                            "respondents": candidate_ids,
                            "answer_presentation_order": presentation_order,
                            "comparison_order": phase.comparison_order,
                            "comparison_seed": phase.comparison_seed,
                            "question_turn_id": probe_entry.turn_id,
                            "answer_turn_ids": [
                                answer.turn_id for answer in ordered_answers
                            ],
                            "question_stream_ids": [
                                probe_entry.metadata.get("stream_id")
                            ],
                            "answer_stream_ids": [
                                answer.metadata.get("stream_id")
                                for answer in ordered_answers
                            ],
                            "probe_sequence_number": probe_entry.metadata.get(
                                "probe_sequence_number"
                            ),
                        },
                    )
                    replay = preauthored_comparisons.get(stream_id)
                    position = int(probe_entry.metadata.get("probe_number", 0))
                    if replay is not None:
                        _validate_preauthored_evidence_source(request, replay)
                        replayed_comparisons.append(
                            self._append_preauthored_evidence(request, replay)
                        )
                        replayed_comparison_positions.append(position)
                        continue
                    comparison_requests.append(request)
                    fresh_comparison_positions.append(position)
                generated_comparisons = self._run_agent_turn_batch(comparison_requests)
                comparison_by_position = dict(
                    zip(
                        [*replayed_comparison_positions, *fresh_comparison_positions],
                        [*replayed_comparisons, *generated_comparisons],
                        strict=True,
                    )
                )
                comparison_entries = [
                    comparison_by_position[position]
                    for position in range(1, probe_count + 1)
                ]

                cumulative_probe_count += probe_count
                judgment_stream_id = _adaptive_wave_judgment_stream_id(
                    phase.name,
                    judge.spec.id,
                    round_index,
                )
                previous_turn_id = (
                    previous_judgment.turn_id if previous_judgment is not None else None
                )
                judgment_prompt_values = {
                    "stream_id": judgment_stream_id,
                    "context_scope": "stream_only",
                    "interaction_mode": "independent_judge_ranking",
                    "interaction_role": "wave_judgment",
                    "judge_id": judge.spec.id,
                    "max_adaptive_candidates": phase.max_adaptive_candidates,
                }
                judgment_metadata = {
                    "stream_id": judgment_stream_id,
                    "interaction_mode": "independent_judge_ranking",
                    "interaction_role": "wave_judgment",
                    "judge": judge.spec.id,
                    "participants": self.participant_ids,
                    "probe_schedule": phase.probe_schedule,
                    "judgment_probe_count": cumulative_probe_count,
                    "judgment_probe_total": total_probe_count,
                    "is_primary_judgment": True,
                    "probe_comparison_turn_ids": [
                        entry.turn_id for entry in comparison_entries
                    ],
                    "probe_comparison_stream_ids": [
                        entry.metadata.get("stream_id") for entry in comparison_entries
                    ],
                    **(
                        {
                            "prior_judgment_turn_id": previous_turn_id,
                            "prior_judgment_stream_id": previous_judgment.metadata.get(
                                "stream_id"
                            ),
                        }
                        if previous_turn_id is not None
                        else {}
                    ),
                }
                judgment_request = _TurnRequest(
                    agent=judge,
                    phase=judgment_phase,
                    round_index=round_index,
                    visibility="private",
                    prompt_values=judgment_prompt_values,
                    extra_context=_format_wave_judgment_context(
                        previous_judgment,
                        comparison_entries,
                    ),
                    metadata=judgment_metadata,
                )
                judgment_replay = preauthored_judgments.get(judgment_stream_id)
                if replay_prefix_open and judgment_replay is not None:
                    _validate_preauthored_ranking_source(
                        judgment_request,
                        judgment_replay,
                    )
                    previous_judgment = self._append_preauthored_ranking(
                        judgment_request,
                        judgment_replay,
                    )
                else:
                    replay_prefix_open = False
                    previous_judgment = self._run_turn_request(judgment_request)
                active_candidates = _adaptive_wave_candidates(
                    previous_judgment,
                    self.agents,
                    phase.max_adaptive_candidates,
                )

    def _append_preauthored_probe(
        self,
        judge: ParticipantAgent,
        phase: PhaseSpec,
        *,
        round_index: int,
        content: str,
        metadata: dict[str, object],
        provenance: dict[str, object],
    ) -> TranscriptEntry:
        entry = TranscriptEntry(
            turn_id=self.next_turn_id,
            phase=phase.name,
            round_index=round_index,
            speaker=judge.spec.id,
            visibility="private",
            content=content,
            metadata={
                "model_ref": judge.model.name,
                "provider": "preauthored",
                "model": judge.model.model,
                "request_params": {},
                "usage": {},
                "finish_reason": "replayed",
                "response_message_keys": [],
                "parse_error": None,
                "structured_error": None,
                **metadata,
                "preauthored_probe": True,
                "source_run": provenance.get("source_run"),
                "source_turn_id": provenance.get("source_turn_id"),
            },
        )
        self.next_turn_id += 1
        self.transcript.append(entry)
        self.store.append_entry(entry)
        self._run_monitor_checks(entry, phase, None)
        return entry

    def _append_preauthored_answer(
        self,
        request: _TurnRequest,
        provenance: dict[str, object],
    ) -> TranscriptEntry:
        metadata = {
            "model_ref": request.agent.model.name,
            "provider": "preauthored",
            "model": request.agent.model.model,
            "request_params": {},
            "usage": {},
            "finish_reason": "replayed",
            "response_message_keys": [],
            "parse_error": None,
            "structured_error": None,
            **(request.metadata or {}),
            "preauthored_answer": True,
            "source_run": provenance.get("source_run"),
            "source_turn_id": provenance.get("source_turn_id"),
        }
        entry = TranscriptEntry(
            turn_id=self.next_turn_id,
            phase=request.phase.name,
            round_index=request.round_index,
            speaker=request.agent.spec.id,
            visibility=request.visibility,
            content=str(provenance["content"]),
            metadata=metadata,
        )
        _validate_independent_answer(request, entry)
        self.next_turn_id += 1
        self.transcript.append(entry)
        self.store.append_entry(entry)
        self._run_monitor_checks(entry, request.phase, None)
        return entry

    def _append_preauthored_evidence(
        self,
        request: _TurnRequest,
        provenance: dict[str, object],
    ) -> TranscriptEntry:
        parsed = provenance.get("parsed")
        entry = TranscriptEntry(
            turn_id=self.next_turn_id,
            phase=request.phase.name,
            round_index=request.round_index,
            speaker=request.agent.spec.id,
            visibility=request.visibility,
            content=str(provenance["content"]),
            parsed=parsed,
            metadata={
                "model_ref": request.agent.model.name,
                "provider": "preauthored",
                "model": request.agent.model.model,
                "request_params": {},
                "usage": {},
                "finish_reason": "replayed",
                "response_message_keys": [],
                "parse_error": None,
                "structured_error": None,
                "round_index": request.round_index,
                **(request.metadata or {}),
                "preauthored_evidence": True,
                "source_run": provenance.get("source_run"),
                "source_turn_id": provenance.get("source_turn_id"),
            },
        )
        structured_error = self._structured_error_summary(
            request.phase,
            parsed if isinstance(parsed, dict) else None,
            None,
            entry.metadata,
            entry.speaker,
        )
        if structured_error is not None:
            raise ExperimentViolationError(
                f"preauthored evidence {entry.metadata.get('stream_id')} is invalid: "
                f"{structured_error}"
            )
        self.next_turn_id += 1
        self.transcript.append(entry)
        self.store.append_entry(entry)
        self._run_monitor_checks(entry, request.phase, parsed)
        return entry

    def _append_preauthored_ranking(
        self,
        request: _TurnRequest,
        provenance: dict[str, object],
    ) -> TranscriptEntry:
        parsed = provenance.get("parsed")
        entry = TranscriptEntry(
            turn_id=self.next_turn_id,
            phase=request.phase.name,
            round_index=request.round_index,
            speaker=request.agent.spec.id,
            visibility=request.visibility,
            content=str(provenance["content"]),
            parsed=parsed,
            metadata={
                "model_ref": request.agent.model.name,
                "provider": "preauthored",
                "model": request.agent.model.model,
                "request_params": {},
                "usage": {},
                "finish_reason": "replayed",
                "response_message_keys": [],
                "parse_error": None,
                "structured_error": None,
                "round_index": request.round_index,
                **(request.metadata or {}),
                "preauthored_ranking": True,
                "source_run": provenance.get("source_run"),
                "source_turn_id": provenance.get("source_turn_id"),
            },
        )
        structured_error = self._structured_error_summary(
            request.phase,
            parsed if isinstance(parsed, dict) else None,
            None,
            entry.metadata,
            entry.speaker,
        )
        if structured_error is not None:
            raise ExperimentViolationError(
                f"preauthored ranking {entry.metadata.get('stream_id')} is invalid: "
                f"{structured_error}"
            )
        self.next_turn_id += 1
        self.transcript.append(entry)
        self.store.append_entry(entry)
        self._run_monitor_checks(entry, request.phase, parsed)
        return entry

    def _run_interactive_discussion_phase(self, phase: PhaseSpec) -> None:
        for round_index in range(1, phase.rounds + 1):
            for agent in self._public_turn_order():
                self._run_agent_turn(
                    agent,
                    phase,
                    round_index=round_index,
                    visibility="public",
                    prompt_values={
                        "stream_id": phase.name,
                        "interaction_mode": "interactive_discussion",
                        "interaction_role": "discussion",
                    },
                    metadata={
                        "stream_id": phase.name,
                        "interaction_mode": "interactive_discussion",
                        "interaction_role": "discussion",
                    },
                )
            self.public_rounds_completed += 1

    def _run_test_matrix_phase(self, phase: PhaseSpec) -> None:
        if not phase.source_phase:
            raise ValueError(f"phase {phase.name} requires source_phase")
        source_entries = self.transcript.entries_for_phase(phase.source_phase)
        source_by_speaker = {entry.speaker: entry for entry in source_entries if entry.visibility == "public"}
        for originator in self._public_turn_order():
            source_entry = source_by_speaker.get(originator.spec.id)
            if source_entry is None:
                raise ValueError(
                    f"phase {phase.name} could not find test proposal from {originator.spec.id} "
                    f"in source phase {phase.source_phase}"
                )
            for respondent in self._public_turn_order():
                if not phase.include_self and respondent.spec.id == originator.spec.id:
                    continue
                self._run_agent_turn(
                    respondent,
                    phase,
                    round_index=None,
                    visibility=_visibility_from_config(phase.response_visibility),
                    prompt_values={
                        "originator_id": originator.spec.id,
                        "respondent_id": respondent.spec.id,
                        "source_turn_id": source_entry.turn_id,
                    },
                    extra_context=_format_test_context(source_entry),
                    metadata={
                        "test_originator": originator.spec.id,
                        "respondent": respondent.spec.id,
                        "source_phase": phase.source_phase,
                        "source_turn_id": source_entry.turn_id,
                    },
                )
            self.public_rounds_completed += 1

    def _run_test_evaluation_phase(self, phase: PhaseSpec) -> None:
        if not phase.source_phase or not phase.answer_phase:
            raise ValueError(f"phase {phase.name} requires source_phase and answer_phase")
        source_entries = {
            entry.speaker: entry
            for entry in self.transcript.entries_for_phase(phase.source_phase)
            if entry.visibility == "public"
        }
        answer_entries = self.transcript.entries_for_phase(phase.answer_phase)
        for originator in self._public_turn_order():
            source_entry = source_entries.get(originator.spec.id)
            if source_entry is None:
                raise ValueError(
                    f"phase {phase.name} could not find test proposal from {originator.spec.id} "
                    f"in source phase {phase.source_phase}"
                )
            answers = [
                entry
                for entry in answer_entries
                if entry.metadata.get("test_originator") == originator.spec.id
            ]
            if not answers:
                raise ValueError(f"phase {phase.name} found no answers to {originator.spec.id}'s test")
            self._run_agent_turn(
                originator,
                phase,
                round_index=None,
                visibility="public",
                prompt_values={
                    "originator_id": originator.spec.id,
                    "source_turn_id": source_entry.turn_id,
                },
                extra_context=_format_evaluation_context(source_entry, answers),
                metadata={
                    "test_originator": originator.spec.id,
                    "source_phase": phase.source_phase,
                    "answer_phase": phase.answer_phase,
                    "source_turn_id": source_entry.turn_id,
                    "answer_turn_ids": [entry.turn_id for entry in answers],
                },
            )
            self.public_rounds_completed += 1

    def _run_separate_interviews_phase(self, phase: PhaseSpec) -> None:
        question_phase = replace(
            phase,
            prompt=phase.question_prompt or "separate_interview_question",
            require_json=False,
            required_keys=[],
        )
        answer_phase = replace(
            phase,
            prompt=phase.answer_prompt or "separate_interview_answer",
            require_json=False,
            required_keys=[],
        )
        assessment_phase = replace(
            phase,
            prompt=phase.assessment_prompt or "separate_interview_assessment",
        )
        for round_index in range(1, phase.rounds + 1):
            for interviewer in self._public_turn_order():
                for respondent in self._public_turn_order():
                    if not phase.include_self and interviewer.spec.id == respondent.spec.id:
                        continue
                    stream_id = _interview_stream_id(phase.name, interviewer.spec.id, respondent.spec.id)
                    base_values = {
                        "stream_id": stream_id,
                        "interaction_mode": "separate_interviews",
                        "interviewer_id": interviewer.spec.id,
                        "respondent_id": respondent.spec.id,
                        "target_participant_id": respondent.spec.id,
                    }
                    base_metadata = {
                        "stream_id": stream_id,
                        "interaction_mode": "separate_interviews",
                        "interviewer": interviewer.spec.id,
                        "respondent": respondent.spec.id,
                    }
                    question_entry = self._run_agent_turn(
                        interviewer,
                        question_phase,
                        round_index=round_index,
                        visibility="private",
                        prompt_values={**base_values, "interaction_role": "question"},
                        extra_context=_format_interview_instruction(
                            interviewer.spec.id,
                            respondent.spec.id,
                            stream_id,
                        ),
                        metadata={**base_metadata, "interaction_role": "question"},
                    )
                    answer_entry = self._run_agent_turn(
                        respondent,
                        answer_phase,
                        round_index=round_index,
                        visibility="private",
                        prompt_values={
                            **base_values,
                            "interaction_role": "answer",
                            "question_turn_id": question_entry.turn_id,
                            "context_scope": "stream_only",
                        },
                        extra_context=_format_interview_question(question_entry),
                        metadata={
                            **base_metadata,
                            "interaction_role": "answer",
                            "question_turn_id": question_entry.turn_id,
                        },
                    )
                    self._run_agent_turn(
                        interviewer,
                        assessment_phase,
                        round_index=round_index,
                        visibility="private",
                        prompt_values={
                            **base_values,
                            "interaction_role": "assessment",
                            "question_turn_id": question_entry.turn_id,
                            "answer_turn_id": answer_entry.turn_id,
                        },
                        extra_context=_format_interview_assessment_context(question_entry, answer_entry),
                        metadata={
                            **base_metadata,
                            "interaction_role": "assessment",
                            "question_turn_id": question_entry.turn_id,
                            "answer_turn_id": answer_entry.turn_id,
                        },
                    )
            self.public_rounds_completed += 1

    def _run_round_robin_probes_phase(self, phase: PhaseSpec) -> None:
        question_phase = replace(
            phase,
            prompt=phase.question_prompt or "round_robin_probe_question",
            require_json=False,
            required_keys=[],
        )
        answer_phase = replace(
            phase,
            prompt=phase.answer_prompt or "round_robin_probe_answer",
            require_json=False,
            required_keys=[],
        )
        assessment_phase = replace(
            phase,
            prompt=phase.assessment_prompt or "round_robin_probe_assessment",
            require_json=True,
            required_keys=ROUND_ROBIN_ASSESSMENT_KEYS,
        )
        ranking_phase = replace(
            phase,
            prompt=phase.ranking_prompt or "round_robin_round_ranking",
            require_json=True,
            required_keys=ROUND_ROBIN_RANKING_KEYS,
        )
        memory_phase = (
            replace(
                phase,
                prompt=phase.memory_prompt,
                require_json=True,
                required_keys=ROUND_ROBIN_MEMORY_KEYS,
            )
            if phase.memory_prompt
            else None
        )

        for round_index in range(1, phase.rounds + 1):
            turn_order = self._public_turn_order()
            question_entries: dict[str, TranscriptEntry] = {}
            respondents_by_interviewer: dict[str, list[ParticipantAgent]] = {}

            for interviewer in turn_order:
                respondents = [
                    agent
                    for agent in turn_order
                    if phase.include_self or agent.spec.id != interviewer.spec.id
                ]
                respondents_by_interviewer[interviewer.spec.id] = respondents
                respondent_ids = [agent.spec.id for agent in respondents]
                probe_id = _round_robin_probe_id(phase.name, round_index, interviewer.spec.id)
                private_limit = self.config.context.max_private_turns
                available_private = (
                    self.transcript.private_entries_for(interviewer.spec.id)[-private_limit:]
                    if private_limit
                    else []
                )
                evidence_turn_ids = [entry.turn_id for entry in available_private]
                question_entries[interviewer.spec.id] = self._run_agent_turn(
                    interviewer,
                    question_phase,
                    round_index=round_index,
                    visibility="private",
                    prompt_values={
                        "stream_id": probe_id,
                        "interaction_mode": "round_robin_probes",
                        "interaction_role": "question",
                        "interviewer_id": interviewer.spec.id,
                        "respondents": ", ".join(respondent_ids),
                        "respondents_json": _json_list(respondent_ids),
                        "probe_id": probe_id,
                    },
                    extra_context=_format_round_robin_question_instruction(
                        interviewer.spec.id,
                        respondent_ids,
                        probe_id,
                    ),
                    metadata={
                        "stream_id": probe_id,
                        "interaction_mode": "round_robin_probes",
                        "interaction_role": "question",
                        "interviewer": interviewer.spec.id,
                        "respondents": respondent_ids,
                        "probe_id": probe_id,
                        "probe_number": 1,
                        "probe_count": 1,
                        "generation_stage": (
                            "baseline_probe"
                            if round_index == 1
                            else "iterative_round_robin"
                        ),
                        "evidence_turn_ids_available": evidence_turn_ids,
                    },
                )

            for interviewer in turn_order:
                question_entry = question_entries[interviewer.spec.id]
                respondents = respondents_by_interviewer[interviewer.spec.id]
                probe_id = _round_robin_probe_id(phase.name, round_index, interviewer.spec.id)
                round_records: list[dict[str, TranscriptEntry]] = []
                for respondent in respondents:
                    stream_id = _interview_stream_id(phase.name, interviewer.spec.id, respondent.spec.id)
                    base_values = {
                        "stream_id": stream_id,
                        "interaction_mode": "round_robin_probes",
                        "interviewer_id": interviewer.spec.id,
                        "respondent_id": respondent.spec.id,
                        "probe_id": probe_id,
                    }
                    base_metadata = {
                        "stream_id": stream_id,
                        "interaction_mode": "round_robin_probes",
                        "interviewer": interviewer.spec.id,
                        "respondent": respondent.spec.id,
                        "probe_id": probe_id,
                        "question_turn_id": question_entry.turn_id,
                    }
                    answer_entry = self._run_agent_turn(
                        respondent,
                        answer_phase,
                        round_index=round_index,
                        visibility="private",
                        prompt_values={
                            **base_values,
                            "interaction_role": "answer",
                            "question_turn_id": question_entry.turn_id,
                        },
                        extra_context=_format_round_robin_probe_for_answer(question_entry),
                        metadata={**base_metadata, "interaction_role": "answer"},
                    )
                    assessment_entry = self._run_agent_turn(
                        interviewer,
                        assessment_phase,
                        round_index=round_index,
                        visibility="private",
                        prompt_values={
                            **base_values,
                            "interaction_role": "assessment",
                            "question_turn_id": question_entry.turn_id,
                            "answer_turn_id": answer_entry.turn_id,
                        },
                        extra_context=_format_interview_assessment_context(question_entry, answer_entry),
                        metadata={
                            **base_metadata,
                            "interaction_role": "assessment",
                            "answer_turn_id": answer_entry.turn_id,
                        },
                    )
                    round_records.append(
                        {
                            "answer": answer_entry,
                            "assessment": assessment_entry,
                        }
                    )

                ranking_entry = self._run_agent_turn(
                    interviewer,
                    ranking_phase,
                    round_index=round_index,
                    visibility="private",
                    prompt_values={
                        "stream_id": probe_id,
                        "interaction_mode": "round_robin_probes",
                        "interaction_role": "round_ranking",
                        "interviewer_id": interviewer.spec.id,
                        "probe_id": probe_id,
                    },
                    extra_context=_format_round_robin_round_context(question_entry, round_records),
                    metadata={
                        "stream_id": probe_id,
                        "interaction_mode": "round_robin_probes",
                        "interaction_role": "round_ranking",
                        "interviewer": interviewer.spec.id,
                        "probe_id": probe_id,
                        "question_turn_id": question_entry.turn_id,
                        "answer_turn_ids": [record["answer"].turn_id for record in round_records],
                        "assessment_turn_ids": [
                            record["assessment"].turn_id for record in round_records
                        ],
                    },
                )

                if memory_phase is not None:
                    self._run_agent_turn(
                        interviewer,
                        memory_phase,
                        round_index=round_index,
                        visibility="private",
                        prompt_values={
                            "stream_id": probe_id,
                            "interaction_mode": "round_robin_probes",
                            "interaction_role": "memory_update",
                            "interviewer_id": interviewer.spec.id,
                            "probe_id": probe_id,
                            "ranking_turn_id": ranking_entry.turn_id,
                        },
                        extra_context=_format_round_robin_memory_context(
                            question_entry,
                            round_records,
                            ranking_entry,
                        ),
                        metadata={
                            "stream_id": probe_id,
                            "interaction_mode": "round_robin_probes",
                            "interaction_role": "memory_update",
                            "interviewer": interviewer.spec.id,
                            "probe_id": probe_id,
                            "question_turn_id": question_entry.turn_id,
                            "ranking_turn_id": ranking_entry.turn_id,
                        },
                        fallback_parsed=_round_robin_memory_fallback(
                            interviewer.spec.id,
                            memory_phase.name,
                            round_index,
                            question_entry,
                            round_records,
                            ranking_entry,
                        ),
                    )
            self.public_rounds_completed += 1

    def _run_agent_turn_batch(
        self,
        requests: list[_TurnRequest],
        *,
        validator: Callable[[_TurnRequest, TranscriptEntry], None] | None = None,
    ) -> list[TranscriptEntry]:
        if not requests:
            return []
        self._check_batch_call_budget(len(requests))
        one_client = requests[0].agent.client
        only_one_serial_client = (
            not one_client.supports_parallel_requests
            and all(request.agent.client is one_client for request in requests)
        )
        if (
            self.config.run.max_parallel_calls == 1
            or len(requests) == 1
            or only_one_serial_client
        ):
            entries = []
            for index, request in enumerate(requests):
                try:
                    entry = self._run_turn_request(
                        request,
                        reserved_model_calls=len(requests) - index - 1,
                    )
                except ModelClientError as error:
                    self._record_batch_failure(index, request, error)
                    if not self._can_record_unavailable(request):
                        raise
                    staged = self._unavailable_answer_entry(
                        request,
                        request.agent.request_params_for_phase(request.phase),
                        error,
                    )
                    entry = replace(staged, turn_id=self.next_turn_id)
                    self.next_turn_id += 1
                    self.transcript.append(entry)
                    self.store.append_entry(entry)
                    self._run_monitor_checks(entry, request.phase, entry.parsed)
                if validator is not None:
                    validator(request, entry)
                entries.append(entry)
            return entries

        workers = min(self.config.run.max_parallel_calls, len(requests))
        prepared_calls = [self._prepare_turn_request(request) for request in requests]
        staged_entries: list[TranscriptEntry | None] = [None] * len(requests)
        self.store.reset_pending_batch()

        def stage_response(index: int, response: ModelResponse) -> None:
            entry = self._run_turn_request(
                requests[index],
                initial_response=response,
                initial_response_recorded=True,
                persist=False,
            )
            staged_entries[index] = entry
            try:
                if validator is not None:
                    validator(requests[index], entry)
            finally:
                self.store.append_pending_entry(entry, index)

        try:
            self._run_prepared_calls(prepared_calls, workers, on_response=stage_response)
        except _PartialBatchError as exc:
            for index, error in exc.failures:
                self._record_batch_failure(index, requests[index], error)
            unrecovered_failures = []
            for index, error in exc.failures:
                request = requests[index]
                can_record_unavailable = (
                    self._can_record_unavailable(request)
                    and isinstance(error, ModelClientError)
                    and staged_entries[index] is None
                )
                if not can_record_unavailable:
                    unrecovered_failures.append((index, error))
                    continue
                entry = self._unavailable_answer_entry(
                    request,
                    prepared_calls[index].request.params,
                    error,
                )
                staged_entries[index] = entry
                self.store.append_pending_entry(entry, index)
            entries = self._commit_staged_entries(requests, staged_entries)
            if unrecovered_failures:
                raise unrecovered_failures[0][1]
            if len(entries) != len(requests):
                raise RuntimeError("parallel call batch ended without every answer recorded")
            return entries
        entries = self._commit_staged_entries(requests, staged_entries)
        if len(entries) != len(requests):
            raise RuntimeError("parallel call batch completed without every staged entry")
        return entries

    def _commit_staged_entries(
        self,
        requests: list[_TurnRequest],
        staged_entries: list[TranscriptEntry | None],
    ) -> list[TranscriptEntry]:
        committed = []
        for request, staged in zip(requests, staged_entries, strict=True):
            if staged is None:
                continue
            entry = replace(staged, turn_id=self.next_turn_id)
            self.next_turn_id += 1
            self.transcript.append(entry)
            self.store.append_entry(entry)
            self._run_monitor_checks(entry, request.phase, entry.parsed)
            committed.append(entry)
        self.store.reset_pending_batch()
        return committed

    def _can_record_unavailable(self, request: _TurnRequest) -> bool:
        return (
            self.config.run.continue_batch_on_call_error
            and request.phase.incomplete_answer_policy == "record_unavailable"
            and (request.metadata or {}).get("interaction_role") == "answer"
        )

    def _record_batch_failure(
        self,
        index: int,
        request: _TurnRequest,
        error: Exception,
    ) -> None:
        self.store.append_batch_failure(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "batch_position": index,
                "model_ref": request.agent.model.name,
                "provider": request.agent.model.provider,
                "model": request.agent.model.model,
                "speaker": request.agent.spec.id,
                "stream_id": (request.metadata or {}).get("stream_id"),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    @staticmethod
    def _unavailable_answer_entry(
        request: _TurnRequest,
        request_params: dict[str, object],
        error: Exception,
    ) -> TranscriptEntry:
        return TranscriptEntry(
            turn_id=0,
            phase=request.phase.name,
            round_index=request.round_index,
            speaker=request.agent.spec.id,
            visibility=request.visibility,
            content="",
            created_at=datetime.now(timezone.utc).isoformat(),
            parsed=None,
            metadata={
                "model_ref": request.agent.model.name,
                "provider": request.agent.model.provider,
                "model": request.agent.model.model,
                "request_params": request_params,
                "usage": {},
                "finish_reason": "provider_error",
                "response_message_keys": [],
                "parse_error": None,
                "structured_error": None,
                **(request.metadata or {}),
                "answer_unavailable": True,
                "provider_error_type": type(error).__name__,
                "provider_error": str(error),
            },
        )

    def _prepare_turn_request(self, request: _TurnRequest) -> _PreparedCall:
        model_request = request.agent.build_turn_request(
            self.transcript,
            request.phase,
            request.round_index,
            prompt_values=request.prompt_values,
            extra_context=request.extra_context,
        )
        return _PreparedCall(
            client=request.agent.client,
            request=model_request,
            model_ref=request.agent.model.name,
        )

    def _run_prepared_calls(
        self,
        calls: list[_PreparedCall],
        workers: int,
        on_response: Callable[[int, ModelResponse], None] | None = None,
    ) -> list[ModelResponse]:
        if workers == 1:
            responses = []
            for index, call in enumerate(calls):
                response = call.client.generate(call.request)
                self._record_model_call(response, call.model_ref)
                if on_response is not None:
                    on_response(index, response)
                responses.append(response)
            return responses

        responses: list[ModelResponse | None] = [None] * len(calls)
        failures: list[tuple[int, Exception]] = []
        pending = list(range(len(calls)))
        active_serial_clients: set[int] = set()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            in_flight: dict[Future[ModelResponse], int] = {}

            def submit_available() -> None:
                while pending and len(in_flight) < workers:
                    position = next(
                        (
                            position
                            for position, index in enumerate(pending)
                            if calls[index].client.supports_parallel_requests
                            or id(calls[index].client) not in active_serial_clients
                        ),
                        None,
                    )
                    if position is None:
                        return
                    index = pending.pop(position)
                    call = calls[index]
                    if not call.client.supports_parallel_requests:
                        active_serial_clients.add(id(call.client))
                    future = executor.submit(call.client.generate, call.request)
                    in_flight[future] = index

            submit_available()

            while in_flight:
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                completed_indices = sorted(
                    ((in_flight.pop(future), future) for future in completed),
                    key=lambda item: item[0],
                )
                for index, future in completed_indices:
                    client = calls[index].client
                    if not client.supports_parallel_requests:
                        active_serial_clients.discard(id(client))
                    try:
                        response = future.result()
                    except Exception as exc:
                        failures.append((index, exc))
                        continue
                    responses[index] = response
                    try:
                        self._record_model_call(response, calls[index].model_ref)
                    except BudgetExceededError as exc:
                        failures.append((index, exc))
                    if on_response is not None:
                        try:
                            on_response(index, response)
                        except Exception as exc:
                            failures.append((index, exc))

                if all(
                    isinstance(error, ExperimentViolationError)
                    or (
                        self.config.run.continue_batch_on_call_error
                        and isinstance(error, ModelClientError)
                    )
                    for _, error in failures
                ):
                    submit_available()

        if failures:
            failures.sort(key=lambda item: item[0])
            raise _PartialBatchError(responses, failures)
        if any(response is None for response in responses):
            raise RuntimeError("parallel call batch completed without every response")
        return [response for response in responses if response is not None]

    def _run_turn_request(
        self,
        request: _TurnRequest,
        *,
        initial_response: ModelResponse | None = None,
        initial_response_recorded: bool = False,
        reserved_model_calls: int = 0,
        persist: bool = True,
    ) -> TranscriptEntry:
        return self._run_agent_turn(
            request.agent,
            request.phase,
            request.round_index,
            request.visibility,
            prompt_values=request.prompt_values,
            extra_context=request.extra_context,
            metadata=request.metadata,
            fallback_parsed=request.fallback_parsed,
            initial_response=initial_response,
            initial_response_recorded=initial_response_recorded,
            reserved_model_calls=reserved_model_calls,
            persist=persist,
        )

    def _run_agent_turn(
        self,
        agent: ParticipantAgent,
        phase: PhaseSpec,
        round_index: int | None,
        visibility: Visibility,
        prompt_values: dict[str, object] | None = None,
        extra_context: str | None = None,
        metadata: dict[str, object] | None = None,
        fallback_parsed: dict[str, object] | None = None,
        initial_response: ModelResponse | None = None,
        initial_response_recorded: bool = False,
        reserved_model_calls: int = 0,
        persist: bool = True,
    ) -> TranscriptEntry:
        if initial_response is None:
            self._check_call_budget()
            response = agent.generate_turn(
                self.transcript,
                phase,
                round_index,
                prompt_values=prompt_values,
                extra_context=extra_context,
            )
            self._record_model_call(response, agent.model.name)
        else:
            response = initial_response
            if not initial_response_recorded:
                self._record_model_call(response, agent.model.name)
        request_params = agent.request_params_for_phase(phase)
        response, visible_retry_metadata, request_params = self._retry_empty_visible_output(
            agent,
            phase,
            round_index,
            response,
            prompt_values=prompt_values,
            extra_context=extra_context,
            reserved_model_calls=reserved_model_calls,
        )
        parsed, parse_error = _parse_structured_json(response.content, phase.require_json)
        repair_metadata: dict[str, object] = {}
        base_metadata = {
            "model_ref": agent.model.name,
            "provider": response.provider,
            "model": response.model,
            "request_params": request_params,
            "usage": response.usage,
            "finish_reason": _extract_finish_reason(response.raw),
            "response_message_keys": _extract_response_message_keys(response.raw),
            "parse_error": parse_error,
            "round_index": round_index,
            **(metadata or {}),
        }
        structured_error = self._structured_error_summary(
            phase,
            parsed,
            parse_error,
            base_metadata,
            agent.spec.id,
        )
        if self._should_repair_structured_json(
            phase,
            visibility,
            structured_error,
            reserved_model_calls=reserved_model_calls,
        ):
            response, parsed, parse_error, repair_metadata, structured_error = self._repair_structured_json(
                agent,
                phase,
                round_index,
                response,
                structured_error or "unknown structured JSON error",
                base_metadata=base_metadata,
                prompt_values=prompt_values,
                reserved_model_calls=reserved_model_calls,
            )
        fallback_metadata: dict[str, object] = {}
        if structured_error is not None and fallback_parsed is not None:
            fallback_metadata = _structured_fallback_metadata(response, structured_error)
            parsed = fallback_parsed
            parse_error = None
            structured_error = None
            response = ModelResponse(
                content=json.dumps(fallback_parsed, ensure_ascii=False),
                raw={"deterministic_fallback": True},
                usage={},
                model=response.model,
                provider=response.provider,
            )

        entry = TranscriptEntry(
            turn_id=self.next_turn_id if persist else 0,
            phase=phase.name,
            round_index=round_index,
            speaker=agent.spec.id,
            visibility=visibility,
            content=response.content,
            parsed=parsed,
            metadata={
                "model_ref": agent.model.name,
                "provider": response.provider,
                "model": response.model,
                "request_params": request_params,
                "usage": response.usage,
                "finish_reason": _extract_finish_reason(response.raw),
                "response_message_keys": _extract_response_message_keys(response.raw),
                "parse_error": parse_error,
                "structured_error": structured_error,
                **visible_retry_metadata,
                **repair_metadata,
                **fallback_metadata,
                **(metadata or {}),
            },
        )
        if persist:
            self.next_turn_id += 1
            self.transcript.append(entry)
            self.store.append_entry(entry)
            self._run_monitor_checks(entry, phase, parsed)
        return entry

    def _retry_empty_visible_output(
        self,
        agent: ParticipantAgent,
        phase: PhaseSpec,
        round_index: int | None,
        original_response: ModelResponse,
        *,
        prompt_values: dict[str, object] | None,
        extra_context: str | None,
        reserved_model_calls: int,
    ) -> tuple[ModelResponse, dict[str, object], dict[str, object]]:
        request_params = agent.request_params_for_phase(phase)
        original_finish_reason = _extract_finish_reason(original_response.raw)
        recovery_reason = (
            "empty" if not original_response.content.strip()
            else "truncated" if original_finish_reason == "length"
            else None
        )
        if recovery_reason is None or self.config.run.visible_text_retries == 0:
            return original_response, {}, request_params

        attempts = []
        latest = original_response
        recovery_override = agent.recovery_params_for_phase(phase)
        recovery_params = {**request_params, **recovery_override}
        for attempt_index in range(1, self.config.run.visible_text_retries + 1):
            if not self._has_call_budget(reserved_model_calls):
                break
            retry_context = _append_visible_text_recovery_instruction(
                extra_context,
                recovery_reason,
            )
            rejected_override = None
            try:
                latest = agent.generate_turn(
                    self.transcript,
                    phase,
                    round_index,
                    prompt_values=prompt_values,
                    extra_context=retry_context,
                    model_params_override=recovery_override,
                )
            except ModelClientError as error:
                if not recovery_override or not _is_recovery_parameter_error(error):
                    raise
                rejected_override = str(error)
                recovery_override = {}
                recovery_params = request_params
                latest = agent.generate_turn(
                    self.transcript,
                    phase,
                    round_index,
                    prompt_values=prompt_values,
                    extra_context=retry_context,
                    model_params_override={},
                )
            self._record_model_call(latest, agent.model.name)
            attempt = {
                "attempt": attempt_index,
                "content": latest.content,
                "usage": latest.usage,
                "finish_reason": _extract_finish_reason(latest.raw),
                "response_message_keys": _extract_response_message_keys(latest.raw),
                "request_params": recovery_params,
            }
            if rejected_override is not None:
                attempt["rejected_recovery_override"] = rejected_override
            attempts.append(attempt)
            if latest.content.strip() and _extract_finish_reason(latest.raw) != "length":
                break
        recovered = bool(latest.content.strip()) and _extract_finish_reason(latest.raw) != "length"
        return (
            latest,
            {
                "visible_text_retry": {
                    "attempted": True,
                    "reason": recovery_reason,
                    "recovered": recovered,
                    "original_content": original_response.content,
                    "original_usage": original_response.usage,
                    "original_finish_reason": _extract_finish_reason(original_response.raw),
                    "attempts": attempts,
                }
            },
            recovery_params,
        )
    def _run_monitor_checks(
        self,
        entry: TranscriptEntry,
        phase: PhaseSpec,
        parsed: dict | None,
    ) -> None:
        if not self.config.monitor.enabled:
            return
        findings = []
        findings.extend(self.monitor.check_entry(entry))
        findings.extend(
            self.monitor.check_required_keys(
                entry,
                parsed,
                phase.required_keys,
                require_json=phase.require_json,
            )
        )
        if phase.require_json:
            findings.extend(self.monitor.check_structured_values(entry, parsed, self.participant_ids))
        for finding in findings:
            self.store.append_finding(finding)
        if self.config.monitor.strict and any(finding.severity == "error" for finding in findings):
            raise ExperimentViolationError(
                f"strict monitor stopped run after {findings[0].code} from {entry.speaker}"
            )

    def _record_model_call(self, response: ModelResponse, model_ref: str) -> None:
        reported_cost = _extract_reported_cost(response.usage)
        self.model_calls += 1
        self.reported_cost_usd += reported_cost
        model = self.config.models[model_ref]
        spend = self.model_spend.setdefault(
            model_ref,
            _ModelSpend(provider=model.provider, provider_model_id=model.model),
        )
        spend.model_calls += 1
        spend.reported_cost_usd += reported_cost
        self._check_cost_budget()

    def _should_repair_structured_json(
        self,
        phase: PhaseSpec,
        visibility: Visibility,
        structured_error: str | None,
        *,
        reserved_model_calls: int,
    ) -> bool:
        return (
            phase.require_json
            and visibility == "private"
            and structured_error is not None
            and self.config.run.structured_json_retries > 0
            and self._has_call_budget(reserved_model_calls)
        )

    def _repair_structured_json(
        self,
        agent: ParticipantAgent,
        phase: PhaseSpec,
        round_index: int | None,
        original_response: ModelResponse,
        original_error: str,
        *,
        base_metadata: dict[str, object],
        prompt_values: dict[str, object] | None,
        reserved_model_calls: int,
    ) -> tuple[ModelResponse, dict | None, str | None, dict[str, object], str | None]:
        attempts = []
        latest_response = original_response
        latest_parsed, latest_parse_error = _parse_structured_json(original_response.content, True)
        latest_error = original_error
        for attempt_index in range(1, self.config.run.structured_json_retries + 1):
            if not self._has_call_budget(reserved_model_calls):
                break
            repair_response = agent.repair_structured_json(
                phase,
                round_index,
                original_content=latest_response.content,
                error=latest_error,
                prompt_values=prompt_values,
            )
            self._record_model_call(repair_response, agent.model.name)
            parsed, parse_error = _parse_structured_json(repair_response.content, True)
            attempt_metadata = {
                **base_metadata,
                "provider": repair_response.provider,
                "model": repair_response.model,
                "usage": repair_response.usage,
                "finish_reason": _extract_finish_reason(repair_response.raw),
                "response_message_keys": _extract_response_message_keys(repair_response.raw),
                "parse_error": parse_error,
            }
            structured_error = self._structured_error_summary(
                phase,
                parsed,
                parse_error,
                attempt_metadata,
                agent.spec.id,
            )
            attempts.append(
                {
                    "attempt": attempt_index,
                    "parse_error": parse_error,
                    "structured_error": structured_error,
                    "content": repair_response.content,
                    "usage": repair_response.usage,
                    "finish_reason": _extract_finish_reason(repair_response.raw),
                    "response_message_keys": _extract_response_message_keys(repair_response.raw),
                }
            )
            latest_response = repair_response
            latest_parsed = parsed
            latest_parse_error = parse_error
            if structured_error is None:
                return repair_response, parsed, None, _structured_repair_metadata(
                    original_response,
                    original_error,
                    attempts,
                    repaired=True,
                ), None
            latest_error = structured_error
        return latest_response, latest_parsed, latest_parse_error, _structured_repair_metadata(
            original_response,
            original_error,
            attempts,
            repaired=False,
        ), latest_error

    def _structured_error_summary(
        self,
        phase: PhaseSpec,
        parsed: dict | None,
        parse_error: str | None,
        metadata: dict[str, object],
        speaker: str,
    ) -> str | None:
        if not phase.require_json:
            return None
        entry = TranscriptEntry(
            turn_id=self.next_turn_id,
            phase=phase.name,
            round_index=metadata.get("round_index") if isinstance(metadata.get("round_index"), int) else None,
            speaker=speaker,
            visibility="private",
            content="",
            parsed=parsed,
            metadata=metadata,
        )
        findings = self.monitor.check_required_keys(
            entry,
            parsed,
            phase.required_keys,
            require_json=True,
        )
        findings.extend(self.monitor.check_structured_values(entry, parsed, self.participant_ids))
        error_findings = [finding for finding in findings if finding.severity == "error"]
        if not error_findings:
            return None
        parts = []
        if parse_error:
            parts.append(f"parse_error: {parse_error}")
        for finding in error_findings:
            evidence = f" Evidence: {finding.evidence}." if finding.evidence else ""
            parts.append(f"{finding.code}: {finding.message}{evidence}")
        expected = _expected_structured_values(metadata)
        if expected:
            parts.append(f"Expected fixed values: {json.dumps(expected, ensure_ascii=False)}")
        return " ".join(parts)

    def _check_call_budget(self) -> None:
        max_calls = self.config.run.max_model_calls
        if max_calls is not None and self.model_calls >= max_calls:
            raise BudgetExceededError(f"model call budget exceeded: {self.model_calls} >= {max_calls}")

    def _check_batch_call_budget(self, call_count: int) -> None:
        max_calls = self.config.run.max_model_calls
        if max_calls is not None and self.model_calls + call_count > max_calls:
            raise BudgetExceededError(
                "model call budget cannot cover batch: "
                f"{self.model_calls} + {call_count} > {max_calls}"
            )

    def _has_call_budget(self, reserved_calls: int = 0) -> bool:
        max_calls = self.config.run.max_model_calls
        return max_calls is None or self.model_calls + reserved_calls < max_calls

    def _check_cost_budget(self) -> None:
        max_cost = self.config.run.max_reported_cost_usd
        if max_cost is not None and self.reported_cost_usd > max_cost:
            raise BudgetExceededError(
                f"reported cost budget exceeded: ${self.reported_cost_usd:.6f} > ${max_cost:.6f}"
            )

    def _public_turn_order(self) -> list[ParticipantAgent]:
        if self.config.protocol.turn_order == "fixed" or not self.agents:
            return list(self.agents)
        if self.config.protocol.turn_order == "rotate":
            offset = self.public_rounds_completed % len(self.agents)
            return self.agents[offset:] + self.agents[:offset]
        raise ValueError(f"unknown turn order: {self.config.protocol.turn_order}")


def _validate_independent_answer(
    request: _TurnRequest,
    entry: TranscriptEntry,
) -> None:
    if entry.content.strip() and entry.metadata.get("finish_reason") != "length":
        return
    if request.phase.incomplete_answer_policy == "record_unavailable":
        entry.metadata["answer_unavailable"] = True
        return
    metadata = request.metadata or {}
    raise ExperimentViolationError(
        f"candidate {metadata.get('respondent')} produced an incomplete answer "
        f"for {metadata.get('probe_id')}"
    )


def _extract_reported_cost(usage: dict) -> float:
    value = usage.get("cost")
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return 0.0
    return cost if isfinite(cost) and cost >= 0 else 0.0


def _is_recovery_parameter_error(error: ModelClientError) -> bool:
    message = str(error)
    return "HTTP 400" in message or "HTTP 422" in message


def _extract_finish_reason(raw_response: dict) -> str | None:
    try:
        value = raw_response["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None
    return value if isinstance(value, str) else None


def _extract_response_message_keys(raw_response: dict) -> list[str]:
    try:
        message = raw_response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return []
    if not isinstance(message, dict):
        return []
    return sorted(str(key) for key in message.keys())


def _parse_structured_json(content: str, required: bool) -> tuple[dict | None, str | None]:
    if not required:
        return None, None
    try:
        return extract_json_object(content), None
    except JsonExtractionError as exc:
        return None, str(exc)


def _structured_repair_metadata(
    original_response: ModelResponse,
    original_error: str,
    attempts: list[dict[str, object]],
    *,
    repaired: bool,
) -> dict[str, object]:
    return {
        "structured_json_repair": {
            "attempted": True,
            "repaired": repaired,
            "original_parse_error": original_error,
            "original_structured_error": original_error,
            "original_content": original_response.content,
            "original_usage": original_response.usage,
            "original_finish_reason": _extract_finish_reason(original_response.raw),
            "attempts": attempts,
        }
    }


def _structured_fallback_metadata(
    failed_response: ModelResponse,
    failed_error: str,
) -> dict[str, object]:
    return {
        "structured_json_fallback": {
            "applied": True,
            "source": "deterministic_round_robin_memory",
            "failed_error": failed_error,
            "failed_content": failed_response.content,
            "failed_usage": failed_response.usage,
            "failed_finish_reason": _extract_finish_reason(failed_response.raw),
        }
    }


def _round_robin_memory_fallback(
    interviewer_id: str,
    phase_name: str,
    round_index: int,
    question_entry: TranscriptEntry,
    round_records: list[dict[str, TranscriptEntry]],
    ranking_entry: TranscriptEntry,
) -> dict[str, object]:
    summaries = []
    for record in round_records:
        answer = record["answer"]
        assessment = record["assessment"]
        assessment_parsed = assessment.parsed if isinstance(assessment.parsed, dict) else {}
        summaries.append(
            {
                "respondent_id": answer.speaker,
                "question_summary": _short_text(
                    assessment_parsed.get("question_summary") or question_entry.content
                ),
                "answer_summary": _short_text(
                    assessment_parsed.get("answer_summary") or answer.content
                ),
                "assessment_summary": _short_text(
                    assessment_parsed.get("assessment") or assessment.content
                ),
                "evidence_to_remember": _short_list(
                    assessment_parsed.get("evidence"),
                    fallback=assessment.content,
                    limit=2,
                ),
            }
        )
    return {
        "participant_id": interviewer_id,
        "phase": phase_name,
        "round_index": round_index,
        "interviewer_id": interviewer_id,
        "qa_assessment_summaries": summaries,
        "ranking_summary": _ranking_summary(ranking_entry),
        "uncertainties": _short_list(
            ranking_entry.parsed.get("uncertainties") if isinstance(ranking_entry.parsed, dict) else None,
            fallback="memory update was generated from routed records after invalid JSON",
            limit=3,
        ),
        "next_round_plan": _short_text(
            (
                ranking_entry.parsed.get("next_probe_strategy")
                if isinstance(ranking_entry.parsed, dict)
                else None
            )
            or "continue probing unresolved ranking uncertainty"
        ),
    }


def _ranking_summary(entry: TranscriptEntry) -> str:
    parsed = entry.parsed if isinstance(entry.parsed, dict) else {}
    ranking = parsed.get("ranking")
    if isinstance(ranking, list):
        return " > ".join(str(item) for item in ranking)
    if isinstance(ranking, dict):
        ranked = sorted(ranking.items(), key=lambda item: item[1])
        return " > ".join(str(item[0]) for item in ranked)
    return _short_text(entry.content)


def _short_list(value: object, *, fallback: object, limit: int) -> list[str]:
    if isinstance(value, list):
        items = [_short_text(item) for item in value if str(item).strip()]
    else:
        items = []
    if not items:
        items = [_short_text(fallback)]
    return items[:limit]


def _short_text(value: object, limit: int = 160) -> str:
    if isinstance(value, list):
        text = "; ".join(str(item) for item in value)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _append_visible_text_recovery_instruction(
    extra_context: str | None,
    reason: str,
) -> str:
    failure = "returned no visible text" if reason == "empty" else "was truncated before completion"
    recovery = (
        f"Recovery instruction: the previous attempt {failure}. "
        "Redo the same task from the beginning, place the complete answer in visible response text, "
        "and stay comfortably within the requested length. Do not continue from the partial cutoff."
    )
    return f"{extra_context}\n\n{recovery}" if extra_context else recovery


def _expected_structured_values(metadata: dict[str, object]) -> dict[str, object]:
    keys = [
        "participant_id",
        "phase",
        "round_index",
        "interviewer_id",
        "respondent_id",
        "target_participant_id",
        "participants",
    ]
    expected = {key: metadata[key] for key in keys if key in metadata}
    if "interviewer" in metadata and "interviewer_id" not in expected:
        expected["interviewer_id"] = metadata["interviewer"]
    if "respondent" in metadata and "respondent_id" not in expected:
        expected["respondent_id"] = metadata["respondent"]
    return expected


def _visibility_from_config(value: str) -> Visibility:
    if value == "public" or value == "private":
        return value
    raise ValueError(f"unknown transcript visibility: {value}")


def _format_test_context(source_entry: TranscriptEntry) -> str:
    return (
        f"Test originator: {source_entry.speaker}\n"
        f"Test proposal turn: {source_entry.turn_id}\n\n"
        f"{source_entry.content}"
    )


def _format_evaluation_context(source_entry: TranscriptEntry, answer_entries: list[TranscriptEntry]) -> str:
    answer_blocks = []
    for entry in answer_entries:
        answer_blocks.append(
            f"Answer by {entry.speaker} (turn {entry.turn_id}):\n{entry.content}"
        )
    return (
        f"Your test proposal (turn {source_entry.turn_id}):\n{source_entry.content}\n\n"
        "Answers to evaluate:\n\n"
        + "\n\n".join(answer_blocks)
    )


def _interview_stream_id(phase_name: str, interviewer_id: str, respondent_id: str) -> str:
    return f"{phase_name}:{interviewer_id}->{respondent_id}"


def _round_robin_probe_id(phase_name: str, round_index: int, interviewer_id: str) -> str:
    return f"{phase_name}:round_{round_index}:{interviewer_id}:probe"


def _load_preauthored_probes(
    path: str | None,
) -> dict[tuple[str, int, int], dict[str, object]]:
    if path is None:
        return {}
    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8")
        if source.suffix == ".jsonl":
            transcript_entries = [
                json.loads(line) for line in raw.splitlines() if line.strip()
            ]
            records = [
                {
                    "judge_id": entry.get("speaker"),
                    "round_index": entry.get("round_index", 1),
                    "probe_number": entry.get("metadata", {}).get("probe_number"),
                    "content": entry.get("content"),
                    "source_run": str(source.parent),
                    "source_turn_id": entry.get("turn_id"),
                }
                for entry in transcript_entries
                if entry.get("metadata", {}).get("interaction_role") == "question"
                and entry.get("metadata", {}).get("finish_reason") != "length"
                and isinstance(entry.get("content"), str)
                and entry.get("content", "").strip()
            ]
        else:
            data = json.loads(raw)
            records = data.get("probes") if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentViolationError(
            f"could not load preauthored probe file {source}: {exc}"
        ) from exc
    if not isinstance(records, list):
        raise ExperimentViolationError(
            f"preauthored probe file {source} must contain a probes list"
        )
    probes: dict[tuple[str, int, int], dict[str, object]] = {}
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ExperimentViolationError(
                f"preauthored probe {index} in {source} must be an object"
            )
        try:
            judge_id = record["judge_id"]
            content = record["content"]
            if not isinstance(judge_id, str) or not judge_id.strip():
                raise ValueError("judge_id must be a non-empty string")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("content must be a non-empty string")
            key = (
                judge_id,
                int(record.get("round_index", 1)),
                int(record["probe_number"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentViolationError(
                f"preauthored probe {index} in {source} has invalid identity fields"
            ) from exc
        if key in probes:
            raise ExperimentViolationError(
                f"preauthored probe file {source} contains duplicate key {key}"
            )
        probes[key] = {**record, "content": content}
    return probes


def _load_preauthored_answers(
    path: str | None,
    allowed_participants: set[str],
    *,
    include_unavailable: bool = False,
    retry_unavailable_rounds: set[int] | None = None,
) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    source = Path(path)
    try:
        source_files = (
            [
                candidate
                for candidate in (
                    source / "transcript.jsonl",
                    source / "pending_batch_entries.jsonl",
                )
                if candidate.exists()
            ]
            if source.is_dir()
            else [source]
        )
        entries = [
            (json.loads(line), source_file)
            for source_file in source_files
            for line in source_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentViolationError(
            f"could not load preauthored answer file {source}: {exc}"
        ) from exc
    answers: dict[str, dict[str, object]] = {}
    for entry, source_file in entries:
        metadata = entry.get("metadata", {}) if isinstance(entry, dict) else {}
        if metadata.get("interaction_role") != "answer":
            continue
        if allowed_participants and entry.get("speaker") not in allowed_participants:
            continue
        content = entry.get("content")
        stream_id = metadata.get("stream_id")
        answer_unavailable = metadata.get("answer_unavailable") is True
        if answer_unavailable:
            round_index = entry.get("round_index", metadata.get("round_index", 1))
            if not include_unavailable or round_index in (retry_unavailable_rounds or set()):
                continue
        if (
            not isinstance(content, str)
            or (not content.strip() and not answer_unavailable)
            or not isinstance(stream_id, str)
            or (metadata.get("finish_reason") == "length" and not answer_unavailable)
        ):
            continue
        if stream_id in answers:
            if _same_preauthored_answer(answers[stream_id], entry):
                continue
            raise ExperimentViolationError(
                f"preauthored answer file {source} contains conflicting entries "
                f"for stream {stream_id}"
            )
        answers[stream_id] = {
            **entry,
            "source_run": str(source if source.is_dir() else source.parent),
            "source_file": str(source_file),
            "source_turn_id": entry.get("turn_id"),
        }
    return answers


def _same_preauthored_answer(
    left: dict[str, object],
    right: dict[str, object],
) -> bool:
    left_metadata = left.get("metadata", {})
    right_metadata = right.get("metadata", {})
    if not isinstance(left_metadata, dict) or not isinstance(right_metadata, dict):
        return False
    return (
        left.get("speaker") == right.get("speaker")
        and left.get("content") == right.get("content")
        and left_metadata.get("model_ref") == right_metadata.get("model_ref")
        and left_metadata.get("answer_unavailable")
        == right_metadata.get("answer_unavailable")
        and left_metadata.get("finish_reason") == right_metadata.get("finish_reason")
    )


def _validate_preauthored_answer_source(
    request: _TurnRequest,
    source: dict[str, object],
) -> None:
    metadata = source.get("metadata")
    source_metadata = metadata if isinstance(metadata, dict) else {}
    expected_stream = (request.metadata or {}).get("stream_id")
    if source.get("speaker") != request.agent.spec.id:
        raise ExperimentViolationError(
            f"preauthored answer {expected_stream} has the wrong speaker"
        )
    if source_metadata.get("model_ref") != request.agent.model.name:
        raise ExperimentViolationError(
            f"preauthored answer {expected_stream} has the wrong model_ref"
        )
    if source_metadata.get("stream_id") != expected_stream:
        raise ExperimentViolationError(
            f"preauthored answer {expected_stream} has the wrong stream_id"
        )


def _load_preauthored_evidence(path: str | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    source = Path(path)
    try:
        entries = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentViolationError(
            f"could not load preauthored evidence file {source}: {exc}"
        ) from exc
    entries_by_turn = {
        entry.get("turn_id"): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("turn_id"), int)
    }
    evidence: dict[str, dict[str, object]] = {}
    for entry in entries:
        metadata = entry.get("metadata", {}) if isinstance(entry, dict) else {}
        if metadata.get("interaction_role") not in {
            "evidence_card",
            "probe_comparison",
        }:
            continue
        content = entry.get("content")
        parsed = entry.get("parsed")
        stream_id = metadata.get("stream_id")
        if (
            not isinstance(content, str)
            or not content.strip()
            or not isinstance(parsed, dict)
            or not isinstance(stream_id, str)
            or metadata.get("finish_reason") == "length"
            or metadata.get("structured_error") is not None
        ):
            continue
        if stream_id in evidence:
            raise ExperimentViolationError(
                f"preauthored evidence file {source} contains duplicate stream {stream_id}"
            )
        evidence[stream_id] = {
            **entry,
            "source_run": str(source.parent),
            "source_turn_id": entry.get("turn_id"),
            "source_question_stream_ids": _semantic_stream_ids(
                metadata,
                "question_stream_ids",
                ("question_turn_ids", "question_turn_id"),
                entries_by_turn,
            ),
            "source_answer_stream_ids": _semantic_stream_ids(
                metadata,
                "answer_stream_ids",
                ("answer_turn_ids",),
                entries_by_turn,
            ),
        }
    return evidence


def _validate_preauthored_evidence_source(
    request: _TurnRequest,
    source: dict[str, object],
) -> None:
    metadata = source.get("metadata")
    source_metadata = metadata if isinstance(metadata, dict) else {}
    expected = request.metadata or {}
    expected_role = expected.get("interaction_role")
    if source_metadata.get("interaction_role") != expected_role:
        raise ExperimentViolationError(
            f"preauthored evidence {expected.get('stream_id')} has the wrong interaction role"
        )
    if source.get("speaker") != request.agent.spec.id:
        raise ExperimentViolationError(
            f"preauthored evidence {expected.get('stream_id')} has the wrong speaker"
        )
    if source_metadata.get("model_ref") != request.agent.model.name:
        raise ExperimentViolationError(
            f"preauthored evidence {expected.get('stream_id')} has the wrong model_ref"
        )
    fields = (
        (
            "stream_id",
            "probe_id",
            "respondents",
        )
        if expected_role == "probe_comparison"
        else (
            "stream_id",
            "candidate",
            "judgment_probe_count",
        )
    )
    for field in fields:
        if source_metadata.get(field) != expected.get(field):
            raise ExperimentViolationError(
                f"preauthored evidence {expected.get('stream_id')} has mismatched {field}"
            )
    for source_field, expected_field in (
        ("source_question_stream_ids", "question_stream_ids"),
        ("source_answer_stream_ids", "answer_stream_ids"),
    ):
        if source.get(source_field) != expected.get(expected_field):
            raise ExperimentViolationError(
                f"preauthored evidence {expected.get('stream_id')} has mismatched {expected_field}"
            )


def _load_preauthored_rankings(path: str | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    source = Path(path)
    try:
        entries = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentViolationError(
            f"could not load preauthored ranking file {source}: {exc}"
        ) from exc
    entries_by_turn = {
        entry.get("turn_id"): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("turn_id"), int)
    }
    rankings: dict[str, dict[str, object]] = {}
    for entry in entries:
        metadata = entry.get("metadata", {}) if isinstance(entry, dict) else {}
        if metadata.get("interaction_role") not in {
            "judge_ranking",
            "wave_judgment",
        }:
            continue
        content = entry.get("content")
        parsed = entry.get("parsed")
        stream_id = metadata.get("stream_id")
        if (
            not isinstance(content, str)
            or not content.strip()
            or not isinstance(parsed, dict)
            or not isinstance(stream_id, str)
            or metadata.get("finish_reason") == "length"
            or metadata.get("structured_error") is not None
        ):
            continue
        if stream_id in rankings:
            raise ExperimentViolationError(
                f"preauthored ranking file {source} contains duplicate stream {stream_id}"
            )
        rankings[stream_id] = {
            **entry,
            "source_run": str(source.parent),
            "source_turn_id": entry.get("turn_id"),
            "source_probe_comparison_stream_ids": _semantic_stream_ids(
                metadata,
                "probe_comparison_stream_ids",
                ("probe_comparison_turn_ids",),
                entries_by_turn,
            ),
            "source_evidence_card_stream_ids": _semantic_stream_ids(
                metadata,
                "evidence_card_stream_ids",
                ("evidence_card_turn_ids",),
                entries_by_turn,
            ),
            "source_prior_judgment_stream_id": (
                metadata.get("prior_judgment_stream_id")
                or _stream_id_for_turn(
                    metadata.get("prior_judgment_turn_id"),
                    entries_by_turn,
                )
            ),
        }
    return rankings


def _validate_preauthored_ranking_source(
    request: _TurnRequest,
    source: dict[str, object],
) -> None:
    metadata = source.get("metadata")
    source_metadata = metadata if isinstance(metadata, dict) else {}
    expected = request.metadata or {}
    expected_role = expected.get("interaction_role")
    if source_metadata.get("interaction_role") != expected_role:
        raise ExperimentViolationError(
            f"preauthored ranking {expected.get('stream_id')} has the wrong interaction role"
        )
    if source.get("speaker") != request.agent.spec.id:
        raise ExperimentViolationError(
            f"preauthored ranking {expected.get('stream_id')} has the wrong speaker"
        )
    if source_metadata.get("model_ref") != request.agent.model.name:
        raise ExperimentViolationError(
            f"preauthored ranking {expected.get('stream_id')} has the wrong model_ref"
        )
    fields = (
        (
            "stream_id",
            "participants",
            "judgment_probe_count",
            "judgment_probe_total",
        )
        if expected_role == "wave_judgment"
        else (
            "stream_id",
            "participants",
            "judgment_probe_count",
            "judgment_probe_total",
        )
    )
    for field in fields:
        if source_metadata.get(field) != expected.get(field):
            raise ExperimentViolationError(
                f"preauthored ranking {expected.get('stream_id')} has mismatched {field}"
            )
    semantic_fields = (
        (
            ("source_probe_comparison_stream_ids", "probe_comparison_stream_ids"),
            ("source_prior_judgment_stream_id", "prior_judgment_stream_id"),
        )
        if expected_role == "wave_judgment"
        else (("source_evidence_card_stream_ids", "evidence_card_stream_ids"),)
    )
    for source_field, expected_field in semantic_fields:
        if source.get(source_field) != expected.get(expected_field):
            raise ExperimentViolationError(
                f"preauthored ranking {expected.get('stream_id')} has mismatched {expected_field}"
            )


def _semantic_stream_ids(
    metadata: dict[str, object],
    stream_field: str,
    turn_fields: tuple[str, ...],
    entries_by_turn: dict[int, dict[str, object]],
) -> list[object]:
    explicit = metadata.get(stream_field)
    if isinstance(explicit, list):
        return explicit
    turn_ids = []
    for field in turn_fields:
        value = metadata.get(field)
        if isinstance(value, list):
            turn_ids.extend(item for item in value if isinstance(item, int))
        elif isinstance(value, int):
            turn_ids.append(value)
    return [_stream_id_for_turn(turn_id, entries_by_turn) for turn_id in turn_ids]


def _stream_id_for_turn(
    turn_id: object,
    entries_by_turn: dict[int, dict[str, object]],
) -> object:
    entry = entries_by_turn.get(turn_id) if isinstance(turn_id, int) else None
    metadata = entry.get("metadata") if isinstance(entry, dict) else None
    return metadata.get("stream_id") if isinstance(metadata, dict) else None


def _independent_judge_probe_id(
    phase_name: str,
    judge_id: str,
    round_index: int,
    probe_number: int,
) -> str:
    return f"{phase_name}:{judge_id}:round_{round_index}:probe_{probe_number}"


def _adaptive_probe_comparison_stream_id(probe_id: str) -> str:
    return f"{probe_id}:comparison"


def _adaptive_wave_judgment_stream_id(
    phase_name: str,
    judge_id: str,
    round_index: int,
) -> str:
    return f"{phase_name}:{judge_id}:round_{round_index}:judgment"


def _independent_judge_evidence_stream_id(
    phase_name: str,
    judge_id: str,
    candidate_id: str,
    probe_count: int,
    round_index: int = 1,
) -> str:
    round_part = "" if round_index == 1 else f":round_{round_index}"
    return f"{phase_name}:{judge_id}:evidence:{candidate_id}{round_part}:probes_{probe_count}"


def _independent_judge_ranking_stream_id(
    phase_name: str,
    judge_id: str,
    probe_count: int,
    round_index: int = 1,
) -> str:
    round_part = "" if round_index == 1 else f":round_{round_index}"
    return f"{phase_name}:{judge_id}:ranking{round_part}:probes_{probe_count}"


def _independent_judgment_probe_counts(
    phase: PhaseSpec,
    round_index: int,
    available_probe_count: int,
) -> list[int]:
    if round_index > 1 or not phase.judgment_probe_counts:
        return [available_probe_count]
    return sorted({*phase.judgment_probe_counts, available_probe_count})


def _json_list(items: list[str]) -> str:
    return json.dumps(items)


def _format_round_robin_question_instruction(
    interviewer_id: str,
    respondent_ids: list[str],
    probe_id: str,
) -> str:
    return (
        f"Probe id: {probe_id}\n"
        f"Questioner: {interviewer_id}\n"
        f"Respondents who will receive the same probe: {', '.join(respondent_ids)}\n\n"
        "Write one probe for all listed respondents. Do not tailor this probe "
        "to a single respondent in this turn."
    )


def _format_independent_judge_probe_instruction(
    judge_id: str,
    candidate_ids: list[str],
    round_index: int,
    prior_probes: list[TranscriptEntry],
    previous_ranking: TranscriptEntry | None,
    evidence_cards: list[TranscriptEntry],
) -> str:
    stage = "common baseline" if round_index == 1 else "selective adaptive follow-up"
    blocks = [
        f"Judge: {judge_id}",
        f"Evaluation stage: {stage}",
        f"Candidates receiving this probe: {', '.join(candidate_ids)}",
    ]
    if prior_probes:
        prior_text = "\n\n".join(
            f"Earlier probe {index} this round:\n{entry.content}"
            for index, entry in enumerate(prior_probes, start=1)
        )
        blocks.append(f"Already selected for this round:\n{prior_text}")
    if previous_ranking is not None:
        blocks.append(
            f"Prior full-evidence ranking (turn {previous_ranking.turn_id}):\n"
            f"{previous_ranking.content}"
        )
    if evidence_cards:
        card_text = "\n\n".join(
            f"Evidence card for {card.metadata.get('candidate')} "
            f"(turn {card.turn_id}):\n{card.content}"
            for card in evidence_cards
        )
        blocks.append(f"Relevant prior evidence cards:\n{card_text}")
    return "\n\n".join(blocks)


def _adaptive_probe_evidence_rule(round_index: int) -> str:
    if round_index == 1:
        return (
            "No candidate answers have been observed. Cover a diagnostic area that "
            "complements the other Round 1 probes instead of duplicating them."
        )
    return (
        "Use the prior cumulative judgment below. The same probe must be useful for "
        "comparing every listed candidate; do not tailor hidden variants by candidate."
    )


def _format_adaptive_judge_probe_context(
    judge_id: str,
    candidate_ids: list[str],
    round_index: int,
    prior_probes: list[TranscriptEntry],
    previous_judgment: TranscriptEntry | None,
) -> str:
    blocks = [
        f"Judge: {judge_id}",
        f"Round: {round_index}",
        f"Candidates receiving this probe: {', '.join(candidate_ids)}",
    ]
    if prior_probes:
        blocks.append(
            "Probes already chosen for this round:\n"
            + "\n\n".join(
                f"Probe {index} (turn {entry.turn_id}):\n{entry.content}"
                for index, entry in enumerate(prior_probes, start=1)
            )
        )
    if previous_judgment is not None:
        blocks.append(
            f"Prior cumulative judgment (turn {previous_judgment.turn_id}):\n"
            f"{previous_judgment.content}"
        )
    return "\n\n".join(blocks)


def _format_probe_comparison_context(
    probe: TranscriptEntry,
    answers: list[TranscriptEntry],
) -> str:
    blocks = [f"Probe {probe.metadata.get('probe_id')} (turn {probe.turn_id}):\n{probe.content}"]
    for answer in answers:
        content = answer.content
        if answer.metadata.get("answer_unavailable"):
            content = (
                "[Response unavailable because the provider returned no complete visible "
                "answer. Treat this as missing evidence, not evidence of low capability.]"
            )
        blocks.append(
            f"Candidate {answer.metadata.get('respondent')} answer "
            f"(turn {answer.turn_id}):\n{content}"
        )
    return "\n\n---\n\n".join(blocks)


def _comparison_presentation_order(
    candidate_ids: list[str],
    order: str,
    seed: int,
    probe_sequence_number: int,
) -> list[str]:
    presented = list(candidate_ids)
    if order == "fixed":
        return presented
    seed_material = f"{seed}:{probe_sequence_number}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    random.Random(stable_seed).shuffle(presented)
    return presented


def _format_wave_judgment_context(
    previous_judgment: TranscriptEntry | None,
    comparisons: list[TranscriptEntry],
) -> str:
    blocks = []
    if previous_judgment is not None:
        blocks.append(
            f"Prior cumulative judgment (turn {previous_judgment.turn_id}):\n"
            f"{previous_judgment.content}"
        )
    blocks.extend(
        f"Current probe comparison for {entry.metadata.get('probe_id')} "
        f"(turn {entry.turn_id}):\n{entry.content}"
        for entry in comparisons
    )
    return "\n\n---\n\n".join(blocks)


def _format_independent_judge_evidence_context(
    probe_entries: list[TranscriptEntry],
    answer_entries: list[TranscriptEntry],
    prior_card: TranscriptEntry | None,
) -> str:
    blocks = []
    if prior_card is not None:
        blocks.append(f"Prior evidence card:\n{prior_card.content}")
    for probe, answer in zip(probe_entries, answer_entries):
        blocks.append(
            f"Probe {probe.metadata.get('probe_number', '?')} (turn {probe.turn_id}):\n"
            f"{probe.content}\n\n"
            f"Candidate answer (turn {answer.turn_id}):\n{answer.content}"
        )
    return "\n\n---\n\n".join(blocks)


def _format_independent_judge_ranking_context(
    evidence_cards: list[TranscriptEntry],
) -> str:
    return "\n\n---\n\n".join(
        f"Evidence card for {card.metadata.get('candidate')} (turn {card.turn_id}):\n"
        f"{card.content}"
        for card in evidence_cards
    )


def _adaptive_candidates(
    ranking_entry: TranscriptEntry,
    candidates: list[ParticipantAgent],
    limit: int,
) -> list[ParticipantAgent]:
    parsed = ranking_entry.parsed if isinstance(ranking_entry.parsed, dict) else {}
    requested = parsed.get("follow_up_candidates")
    candidate_by_id = {candidate.spec.id: candidate for candidate in candidates}
    selected_ids = _valid_unique_ids(requested, candidate_by_id)
    if len(selected_ids) < 2:
        uncertain_pairs = parsed.get("uncertain_pairs")
        flattened = []
        if isinstance(uncertain_pairs, list):
            for pair in uncertain_pairs:
                if isinstance(pair, list):
                    flattened.extend(pair)
        selected_ids = _valid_unique_ids(flattened, candidate_by_id)
    if len(selected_ids) < 2:
        return []
    return [candidate_by_id[candidate_id] for candidate_id in selected_ids[:limit]]


def _adaptive_wave_candidates(
    judgment_entry: TranscriptEntry,
    candidates: list[ParticipantAgent],
    limit: int,
) -> list[ParticipantAgent]:
    selected = _adaptive_candidates(judgment_entry, candidates, limit)
    return selected if len(selected) >= 2 else list(candidates)


def _valid_unique_ids(value: object, valid: dict[str, ParticipantAgent]) -> list[str]:
    if not isinstance(value, list):
        return []
    selected = []
    for item in value:
        candidate_id = str(item)
        if candidate_id in valid and candidate_id not in selected:
            selected.append(candidate_id)
    return selected


def _format_interview_instruction(interviewer_id: str, respondent_id: str, stream_id: str) -> str:
    return (
        f"Interview stream: {stream_id}\n"
        f"Interviewer: {interviewer_id}\n"
        f"Respondent: {respondent_id}\n\n"
        "Ask a question for this respondent only. The stream-local history above "
        "contains prior turns in this interview stream, if any."
    )


def _format_interview_question(question_entry: TranscriptEntry) -> str:
    return (
        f"Question turn: {question_entry.turn_id}\n"
        f"Interviewer: {question_entry.speaker}\n\n"
        f"{question_entry.content}"
    )


def _format_round_robin_probe_for_answer(question_entry: TranscriptEntry) -> str:
    probe_id = question_entry.metadata.get("probe_id") or question_entry.metadata.get("stream_id") or ""
    return (
        "Exact routed probe to answer now. If any prior private note conflicts "
        "with this probe, answer this probe.\n"
        f"Probe id: {probe_id}\n"
        f"Probe turn: {question_entry.turn_id}\n"
        f"Questioner: {question_entry.speaker}\n\n"
        "<PROBE>\n"
        f"{question_entry.content}"
        "\n</PROBE>"
    )


def _format_interview_assessment_context(
    question_entry: TranscriptEntry,
    answer_entry: TranscriptEntry,
) -> str:
    return (
        f"Question turn {question_entry.turn_id} by {question_entry.speaker}:\n"
        f"{question_entry.content}\n\n"
        f"Answer turn {answer_entry.turn_id} by {answer_entry.speaker}:\n"
        f"{answer_entry.content}"
    )


def _format_round_robin_round_context(
    question_entry: TranscriptEntry,
    round_records: list[dict[str, TranscriptEntry]],
) -> str:
    blocks = [
        f"Probe turn {question_entry.turn_id} by {question_entry.speaker}:\n"
        f"{question_entry.content}"
    ]
    for record in round_records:
        answer = record["answer"]
        assessment = record["assessment"]
        blocks.append(
            f"Respondent {answer.speaker}\n\n"
            f"Answer turn {answer.turn_id}:\n{answer.content}\n\n"
            f"Your assessment turn {assessment.turn_id}:\n{assessment.content}"
        )
    return "\n\n---\n\n".join(blocks)


def _format_round_robin_memory_context(
    question_entry: TranscriptEntry,
    round_records: list[dict[str, TranscriptEntry]],
    ranking_entry: TranscriptEntry,
) -> str:
    return (
        _format_round_robin_round_context(question_entry, round_records)
        + "\n\n---\n\n"
        + f"Your round ranking turn {ranking_entry.turn_id}:\n{ranking_entry.content}"
    )
