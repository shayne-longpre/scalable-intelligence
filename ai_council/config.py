from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment config is invalid or internally inconsistent."""


PRIVATE_PHASE_KINDS = {"private", "private_reflection", "private_judgment"}
PUBLIC_PHASE_KINDS = {
    "public",
    "public_round_robin",
    "round_robin",
    "interactive_discussion",
    "round_robin_probes",
    "public_test_matrix",
    "public_test_evaluation",
    "separate_interviews",
}
COMPOSITE_PHASE_KINDS = {"independent_judge_ranking"}
VALID_PHASE_KINDS = PRIVATE_PHASE_KINDS | PUBLIC_PHASE_KINDS | COMPOSITE_PHASE_KINDS
VALID_VISIBILITIES = {"public", "private"}
VALID_TURN_ORDERS = {"fixed", "rotate"}
VALID_CONTEXT_MODES = {"transcript", "private_memory"}
VALID_ADAPTIVE_TARGETING = {"judge_selected", "all"}
VALID_COMPARISON_ORDERS = {"fixed", "seeded_shuffle"}
VALID_INCOMPLETE_ANSWER_POLICIES = {"fail", "record_unavailable"}


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    kind: str
    client_factory: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 90.0
    request_retries: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderSpec":
        client_factory = data.get("client_factory")
        if client_factory is not None and (
            not isinstance(client_factory, str) or not client_factory.strip()
        ):
            raise ConfigError("provider client_factory must be a non-empty string")
        return cls(
            name=_required(data, "name"),
            kind=_required(data, "kind"),
            client_factory=client_factory,
            api_key_env=data.get("api_key_env"),
            base_url=data.get("base_url"),
            headers=dict(data.get("headers", {})),
            options=dict(data.get("options", {})),
            timeout_seconds=float(data.get("timeout_seconds", 90.0)),
            request_retries=int(data.get("request_retries", 2)),
        )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    recovery_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        return cls(
            name=_required(data, "name"),
            provider=_required(data, "provider"),
            model=_required(data, "model"),
            params=dict(data.get("params", {})),
            recovery_params=dict(data.get("recovery_params", {})),
        )


@dataclass(frozen=True)
class ParticipantSpec:
    id: str
    model: str
    system_prompt: str = "blind_council_participant"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParticipantSpec":
        return cls(
            id=_required(data, "id"),
            model=_required(data, "model"),
            system_prompt=data.get("system_prompt", "blind_council_participant"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class MonitorSpec:
    enabled: bool = True
    model: str | None = None
    identity_terms: list[str] = field(default_factory=list)
    strict: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MonitorSpec":
        if data is None:
            return cls()
        return cls(
            enabled=_as_bool(data.get("enabled", True), "monitor.enabled"),
            model=data.get("model"),
            identity_terms=list(data.get("identity_terms", [])),
            strict=_as_bool(data.get("strict", False), "monitor.strict"),
        )


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    kind: str
    prompt: str
    rounds: int = 1
    visibility: str = "public"
    require_json: bool = False
    required_keys: list[str] = field(default_factory=list)
    source_phase: str | None = None
    answer_phase: str | None = None
    include_self: bool = True
    response_visibility: str = "public"
    question_prompt: str | None = None
    answer_prompt: str | None = None
    assessment_prompt: str | None = None
    ranking_prompt: str | None = None
    memory_prompt: str | None = None
    probes_per_round: int = 1
    probe_schedule: list[int] = field(default_factory=list)
    judgment_probe_counts: list[int] = field(default_factory=list)
    adaptive_probes_per_round: int = 1
    max_adaptive_candidates: int = 4
    adaptive_targeting: str = "judge_selected"
    comparison_order: str = "fixed"
    comparison_seed: int = 0
    incomplete_answer_policy: str = "fail"
    reuse_unavailable_answers: bool = False
    retry_unavailable_rounds: list[int] = field(default_factory=list)
    replay_source_targets: bool = False
    preauthored_probe_file: str | None = None
    preauthored_answer_file: str | None = None
    preauthored_answer_participants: list[str] = field(default_factory=list)
    preauthored_evidence_file: str | None = None
    preauthored_ranking_file: str | None = None
    model_params: dict[str, Any] = field(default_factory=dict)
    recovery_model_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhaseSpec":
        probe_schedule = [int(value) for value in data.get("probe_schedule", [])]
        rounds = int(data.get("rounds", len(probe_schedule) or 1))
        return cls(
            name=_required(data, "name"),
            kind=_required(data, "kind"),
            prompt=_required(data, "prompt"),
            rounds=rounds,
            visibility=data.get("visibility", "public"),
            require_json=_as_bool(data.get("require_json", False), f"phase {data.get('name', '<unknown>')}.require_json"),
            required_keys=list(data.get("required_keys", [])),
            source_phase=data.get("source_phase"),
            answer_phase=data.get("answer_phase"),
            include_self=_as_bool(data.get("include_self", True), f"phase {data.get('name', '<unknown>')}.include_self"),
            response_visibility=data.get("response_visibility", "public"),
            question_prompt=data.get("question_prompt"),
            answer_prompt=data.get("answer_prompt"),
            assessment_prompt=data.get("assessment_prompt"),
            ranking_prompt=data.get("ranking_prompt"),
            memory_prompt=data.get("memory_prompt"),
            probes_per_round=int(data.get("probes_per_round", 1)),
            probe_schedule=probe_schedule,
            judgment_probe_counts=[
                int(value) for value in data.get("judgment_probe_counts", [])
            ],
            adaptive_probes_per_round=int(data.get("adaptive_probes_per_round", 1)),
            max_adaptive_candidates=int(data.get("max_adaptive_candidates", 4)),
            adaptive_targeting=str(data.get("adaptive_targeting", "judge_selected")),
            comparison_order=str(data.get("comparison_order", "fixed")),
            comparison_seed=int(data.get("comparison_seed", 0)),
            incomplete_answer_policy=str(data.get("incomplete_answer_policy", "fail")),
            reuse_unavailable_answers=_as_bool(
                data.get("reuse_unavailable_answers", False),
                "phase.reuse_unavailable_answers",
            ),
            retry_unavailable_rounds=[
                int(value) for value in data.get("retry_unavailable_rounds", [])
            ],
            replay_source_targets=_as_bool(
                data.get("replay_source_targets", False),
                "phase.replay_source_targets",
            ),
            preauthored_probe_file=data.get("preauthored_probe_file"),
            preauthored_answer_file=data.get("preauthored_answer_file"),
            preauthored_answer_participants=list(
                data.get("preauthored_answer_participants", [])
            ),
            preauthored_evidence_file=data.get("preauthored_evidence_file"),
            preauthored_ranking_file=data.get("preauthored_ranking_file"),
            model_params=dict(data.get("model_params", {})),
            recovery_model_params=dict(data.get("recovery_model_params", {})),
        )


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    phases: list[PhaseSpec]
    turn_order: str = "fixed"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProtocolSpec":
        phases = [PhaseSpec.from_dict(item) for item in data.get("phases", [])]
        if not phases:
            raise ConfigError("protocol.phases must contain at least one phase")
        return cls(
            name=data.get("name", "unnamed_protocol"),
            phases=phases,
            turn_order=data.get("turn_order", "fixed"),
        )


@dataclass(frozen=True)
class RunSpec:
    output_dir: str = "runs"
    max_context_turns: int = 80
    max_parallel_calls: int = 1
    max_model_calls: int | None = None
    max_reported_cost_usd: float | None = None
    structured_json_retries: int = 1
    visible_text_retries: int = 1
    continue_batch_on_call_error: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RunSpec":
        if data is None:
            return cls()
        return cls(
            output_dir=data.get("output_dir", "runs"),
            max_context_turns=int(data.get("max_context_turns", 80)),
            max_parallel_calls=int(data.get("max_parallel_calls", 1)),
            max_model_calls=(
                int(data["max_model_calls"]) if data.get("max_model_calls") is not None else None
            ),
            max_reported_cost_usd=(
                float(data["max_reported_cost_usd"])
                if data.get("max_reported_cost_usd") is not None
                else None
            ),
            structured_json_retries=int(data.get("structured_json_retries", 1)),
            visible_text_retries=int(data.get("visible_text_retries", 1)),
            continue_batch_on_call_error=_as_bool(
                data.get("continue_batch_on_call_error", False),
                "run.continue_batch_on_call_error",
            ),
        )


@dataclass(frozen=True)
class ContextSpec:
    mode: str = "transcript"
    max_public_turns: int | None = None
    max_private_turns: int = 20
    max_stream_turns: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ContextSpec":
        if data is None:
            return cls()
        return cls(
            mode=data.get("mode", "transcript"),
            max_public_turns=(
                int(data["max_public_turns"])
                if data.get("max_public_turns") is not None
                else None
            ),
            max_private_turns=int(data.get("max_private_turns", 20)),
            max_stream_turns=int(data.get("max_stream_turns", 20)),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    providers: dict[str, ProviderSpec]
    models: dict[str, ModelSpec]
    participants: list[ParticipantSpec]
    protocol: ProtocolSpec
    judges: list[ParticipantSpec] = field(default_factory=list)
    monitor: MonitorSpec = field(default_factory=MonitorSpec)
    run: RunSpec = field(default_factory=RunSpec)
    context: ContextSpec = field(default_factory=ContextSpec)
    prompt_overrides: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        providers = _load_named_items(_named_items(data.get("providers", [])), ProviderSpec.from_dict)
        models = _load_named_items(_named_items(data.get("models", [])), ModelSpec.from_dict)
        participants = [ParticipantSpec.from_dict(item) for item in data.get("participants", [])]
        judges = [ParticipantSpec.from_dict(item) for item in data.get("judges", [])]
        if not providers:
            raise ConfigError("providers must contain at least one provider")
        if not models:
            raise ConfigError("models must contain at least one model")
        if not participants:
            raise ConfigError("participants must contain at least one participant")

        config = cls(
            name=data.get("name", "ai_council_experiment"),
            providers=providers,
            models=models,
            participants=participants,
            protocol=ProtocolSpec.from_dict(_required(data, "protocol")),
            judges=judges,
            monitor=MonitorSpec.from_dict(data.get("monitor")),
            run=RunSpec.from_dict(data.get("run")),
            context=ContextSpec.from_dict(data.get("context")),
            prompt_overrides=dict(data.get("prompt_overrides", {})),
            metadata=dict(data.get("metadata", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        participant_ids = [participant.id for participant in self.participants]
        if len(participant_ids) != len(set(participant_ids)):
            raise ConfigError("participant ids must be unique")
        judge_ids = [judge.id for judge in self.judges]
        if len(judge_ids) != len(set(judge_ids)):
            raise ConfigError("judge ids must be unique")
        if set(participant_ids) & set(judge_ids):
            raise ConfigError("participant and judge ids must be disjoint")
        if self.protocol.turn_order not in VALID_TURN_ORDERS:
            raise ConfigError(f"protocol turn_order {self.protocol.turn_order!r} is not implemented")
        if self.context.mode not in VALID_CONTEXT_MODES:
            raise ConfigError(f"context mode {self.context.mode!r} is not implemented")
        if self.context.max_public_turns is not None and self.context.max_public_turns < 0:
            raise ConfigError("context.max_public_turns must be non-negative")
        if self.context.max_private_turns < 0:
            raise ConfigError("context.max_private_turns must be non-negative")
        if self.context.max_stream_turns < 0:
            raise ConfigError("context.max_stream_turns must be non-negative")
        if self.run.structured_json_retries < 0:
            raise ConfigError("run.structured_json_retries must be non-negative")
        if self.run.visible_text_retries < 0:
            raise ConfigError("run.visible_text_retries must be non-negative")
        if self.run.max_parallel_calls < 1:
            raise ConfigError("run.max_parallel_calls must be at least 1")
        phase_names_seen: set[str] = set()
        phases_by_name = {phase.name: phase for phase in self.protocol.phases}
        if len(phases_by_name) != len(self.protocol.phases):
            raise ConfigError("phase names must be unique")
        for phase in self.protocol.phases:
            self._validate_prompt_id(phase.prompt, f"phase {phase.name!r}")
            for field_name, prompt_id in (
                ("question_prompt", phase.question_prompt),
                ("answer_prompt", phase.answer_prompt),
                ("assessment_prompt", phase.assessment_prompt),
                ("ranking_prompt", phase.ranking_prompt),
                ("memory_prompt", phase.memory_prompt),
            ):
                if prompt_id is not None:
                    self._validate_prompt_id(prompt_id, f"phase {phase.name!r} {field_name}")
            if phase.kind not in VALID_PHASE_KINDS:
                raise ConfigError(f"phase {phase.name!r} has unknown kind {phase.kind!r}")
            if phase.visibility not in VALID_VISIBILITIES:
                raise ConfigError(f"phase {phase.name!r} has unknown visibility {phase.visibility!r}")
            if phase.response_visibility not in VALID_VISIBILITIES:
                raise ConfigError(f"phase {phase.name!r} has unknown response_visibility {phase.response_visibility!r}")
            if phase.kind != "public_test_matrix" and phase.response_visibility != "public":
                raise ConfigError(
                    f"phase {phase.name!r} sets response_visibility, "
                    "but response_visibility is only used by public_test_matrix phases"
                )
            if phase.rounds < 1:
                raise ConfigError(f"phase {phase.name!r} rounds must be at least 1")
            if phase.probes_per_round < 1:
                raise ConfigError(f"phase {phase.name!r} probes_per_round must be at least 1")
            if any(value < 1 for value in phase.probe_schedule):
                raise ConfigError(
                    f"phase {phase.name!r} probe_schedule must contain positive integers"
                )
            if phase.probe_schedule and phase.kind != "independent_judge_ranking":
                raise ConfigError(
                    f"phase {phase.name!r} sets probe_schedule, but only "
                    "independent_judge_ranking phases support adaptive probe waves"
                )
            if phase.probe_schedule and phase.rounds != len(phase.probe_schedule):
                raise ConfigError(
                    f"phase {phase.name!r} rounds must equal the number of probe_schedule entries"
                )
            if phase.probe_schedule and phase.judgment_probe_counts:
                raise ConfigError(
                    f"phase {phase.name!r} cannot combine probe_schedule with "
                    "judgment_probe_counts"
                )
            if phase.probe_schedule and (
                phase.probes_per_round != 1
                or phase.adaptive_probes_per_round != 1
            ):
                raise ConfigError(
                    f"phase {phase.name!r} cannot combine probe_schedule with legacy "
                    "probes_per_round or adaptive_probes_per_round settings"
                )
            if any(value < 1 for value in phase.judgment_probe_counts):
                raise ConfigError(
                    f"phase {phase.name!r} judgment_probe_counts must contain positive integers"
                )
            if len(phase.judgment_probe_counts) != len(set(phase.judgment_probe_counts)):
                raise ConfigError(
                    f"phase {phase.name!r} judgment_probe_counts must not contain duplicates"
                )
            if any(value > phase.probes_per_round for value in phase.judgment_probe_counts):
                raise ConfigError(
                    f"phase {phase.name!r} judgment_probe_counts cannot exceed probes_per_round"
                )
            if phase.judgment_probe_counts and phase.kind != "independent_judge_ranking":
                raise ConfigError(
                    f"phase {phase.name!r} sets judgment_probe_counts, but only "
                    "independent_judge_ranking phases support probe-prefix judgments"
                )
            if phase.adaptive_probes_per_round < 1:
                raise ConfigError(
                    f"phase {phase.name!r} adaptive_probes_per_round must be at least 1"
                )
            if phase.max_adaptive_candidates < 2:
                raise ConfigError(
                    f"phase {phase.name!r} max_adaptive_candidates must be at least 2"
                )
            if phase.adaptive_targeting not in VALID_ADAPTIVE_TARGETING:
                raise ConfigError(
                    f"phase {phase.name!r} adaptive_targeting must be one of "
                    f"{sorted(VALID_ADAPTIVE_TARGETING)}"
                )
            if phase.comparison_order not in VALID_COMPARISON_ORDERS:
                raise ConfigError(
                    f"phase {phase.name!r} comparison_order must be one of "
                    f"{sorted(VALID_COMPARISON_ORDERS)}"
                )
            if phase.incomplete_answer_policy not in VALID_INCOMPLETE_ANSWER_POLICIES:
                raise ConfigError(
                    f"phase {phase.name!r} incomplete_answer_policy must be one of "
                    f"{sorted(VALID_INCOMPLETE_ANSWER_POLICIES)}"
                )
            if any(value < 1 for value in phase.retry_unavailable_rounds):
                raise ConfigError(
                    f"phase {phase.name!r} retry_unavailable_rounds must contain "
                    "positive integers"
                )
            if len(phase.retry_unavailable_rounds) != len(
                set(phase.retry_unavailable_rounds)
            ):
                raise ConfigError(
                    f"phase {phase.name!r} retry_unavailable_rounds must not contain "
                    "duplicates"
                )
            if phase.retry_unavailable_rounds and not phase.reuse_unavailable_answers:
                raise ConfigError(
                    f"phase {phase.name!r} retry_unavailable_rounds requires "
                    "reuse_unavailable_answers"
                )
            if phase.replay_source_targets and not phase.preauthored_probe_file:
                raise ConfigError(
                    f"phase {phase.name!r} replay_source_targets requires "
                    "preauthored_probe_file"
                )
            if phase.kind in PRIVATE_PHASE_KINDS and phase.visibility != "private":
                raise ConfigError(f"private phase {phase.name!r} must use private visibility")
            if phase.kind in PUBLIC_PHASE_KINDS and phase.visibility != "public":
                raise ConfigError(f"public phase {phase.name!r} must use public visibility")
            if phase.kind in COMPOSITE_PHASE_KINDS and phase.visibility != "private":
                raise ConfigError(f"composite phase {phase.name!r} must use private visibility")
            if phase.kind == "independent_judge_ranking" and not self.judges:
                raise ConfigError(
                    f"phase {phase.name!r} requires at least one configured judge"
                )
            if phase.preauthored_probe_file and phase.kind != "independent_judge_ranking":
                raise ConfigError(
                    f"phase {phase.name!r} preauthored_probe_file is only valid for "
                    "independent_judge_ranking"
                )
            if phase.preauthored_answer_file and phase.kind != "independent_judge_ranking":
                raise ConfigError(
                    f"phase {phase.name!r} preauthored_answer_file is only valid for "
                    "independent_judge_ranking"
                )
            if (
                phase.preauthored_answer_participants
                and not phase.preauthored_answer_file
            ):
                raise ConfigError(
                    f"phase {phase.name!r} preauthored_answer_participants requires "
                    "preauthored_answer_file"
                )
            if phase.preauthored_evidence_file and phase.kind != "independent_judge_ranking":
                raise ConfigError(
                    f"phase {phase.name!r} preauthored_evidence_file is only valid for "
                    "independent_judge_ranking"
                )
            if phase.preauthored_ranking_file and phase.kind != "independent_judge_ranking":
                raise ConfigError(
                    f"phase {phase.name!r} preauthored_ranking_file is only valid for "
                    "independent_judge_ranking"
                )
            if phase.kind in {"public_test_matrix", "public_test_evaluation"} and not phase.source_phase:
                raise ConfigError(f"phase {phase.name!r} must define source_phase")
            if phase.kind == "public_test_evaluation" and not phase.answer_phase:
                raise ConfigError(f"phase {phase.name!r} must define answer_phase")
            for field_name, referenced_phase in (
                ("source_phase", phase.source_phase),
                ("answer_phase", phase.answer_phase),
            ):
                if referenced_phase is None:
                    continue
                if referenced_phase not in phases_by_name:
                    raise ConfigError(
                        f"phase {phase.name!r} references unknown {field_name} {referenced_phase!r}"
                    )
                if referenced_phase not in phase_names_seen:
                    raise ConfigError(
                        f"phase {phase.name!r} references {field_name} {referenced_phase!r} "
                        "before that phase has run"
                    )
                referenced_spec = phases_by_name[referenced_phase]
                if (
                    field_name == "source_phase"
                    and phase.kind in {"public_test_matrix", "public_test_evaluation"}
                    and referenced_spec.visibility != "public"
                ):
                    raise ConfigError(
                        f"phase {phase.name!r} source_phase {referenced_phase!r} must be public"
                    )
            phase_names_seen.add(phase.name)
        for provider in self.providers.values():
            if provider.timeout_seconds <= 0:
                raise ConfigError(
                    f"provider {provider.name!r} timeout_seconds must be positive"
                )
            if provider.request_retries < 0:
                raise ConfigError(
                    f"provider {provider.name!r} request_retries must be non-negative"
                )
        for model in self.models.values():
            if model.provider not in self.providers:
                raise ConfigError(f"model {model.name!r} references unknown provider {model.provider!r}")
        for participant in self.participants:
            self._validate_prompt_id(participant.system_prompt, f"participant {participant.id!r}")
            if participant.model not in self.models:
                raise ConfigError(f"participant {participant.id!r} references unknown model {participant.model!r}")
        for judge in self.judges:
            self._validate_prompt_id(judge.system_prompt, f"judge {judge.id!r}")
            if judge.model not in self.models:
                raise ConfigError(f"judge {judge.id!r} references unknown model {judge.model!r}")
        if self.monitor.model and self.monitor.model not in self.models:
            raise ConfigError(f"monitor references unknown model {self.monitor.model!r}")

    def _validate_prompt_id(self, prompt_id: str, owner: str) -> None:
        from ai_council.prompts import DEFAULT_PROMPTS

        if prompt_id not in self.prompt_overrides and prompt_id not in DEFAULT_PROMPTS:
            raise ConfigError(f"{owner} references unknown prompt {prompt_id!r}")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return ExperimentConfig.from_dict(data)


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise ConfigError(f"missing required config key: {key}")
    return data[key]


def _load_named_items(items: list[dict[str, Any]], factory: Any) -> dict[str, Any]:
    loaded = {}
    for item in items:
        spec = factory(item)
        if spec.name in loaded:
            raise ConfigError(f"duplicate named config item: {spec.name}")
        loaded[spec.name] = spec
    return loaded


def _named_items(items: list[dict[str, Any]] | dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(items, dict):
        return list(items.values())
    return items


def _as_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"{field_name} must be a boolean")
