from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from threading import Lock
from typing import Any, Callable, Mapping

from ai_council.clients.base import ModelClient
from ai_council.clients.registry import build_client
from ai_council.config import ConfigError, ProviderSpec
from ai_council.core import ModelRequest, ModelResponse
from ai_council.env import load_dotenv
from ai_council.json_tools import JsonExtractionError, extract_json_object
from ai_council.probe_catalog import select_primary_occurrence
from ai_council.probe_self_analysis import (
    build_probe_self_study_summary,
    render_probe_self_study,
)


SCHEMA_VERSION = "probe-self-study-v1"
PROMPT_VERSION = "2026-07-27-v2"
VALID_STAGES = (
    "author_solve",
    "author_assess",
    "reference_score",
)


@dataclass(frozen=True)
class ReferenceEvaluator:
    id: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    recovery_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReferenceEvaluator":
        return cls(
            id=_required_string(data, "id"),
            model=_required_string(data, "model"),
            params=dict(data.get("params", {})),
            recovery_params=dict(data.get("recovery_params", {})),
        )


@dataclass(frozen=True)
class ProbeSelfStudyConfig:
    name: str
    catalog_file: Path
    provider: ProviderSpec
    reference_evaluator: ReferenceEvaluator
    output_dir: Path
    published_json_file: Path | None = None
    max_parallel_calls: int = 8
    structured_json_retries: int = 1
    max_reported_cost_usd: float | None = None
    stages: tuple[str, ...] = (
        "author_solve",
        "author_assess",
        "reference_score",
    )
    comparison_seed: int = 0
    probe_ids: tuple[str, ...] = ()

    @classmethod
    def from_path(cls, path: str | Path) -> "ProbeSelfStudyConfig":
        config_path = Path(path).resolve()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ConfigError("probe self-study config must be a JSON object")
        base_dir = (
            config_path.parents[1]
            if config_path.parent.name == "examples"
            else Path.cwd()
        )
        catalog_file = _resolved_path(
            _required_string(data, "catalog_file"), base_dir
        )
        output_dir = _resolved_path(
            _required_string(data, "output_dir"), base_dir
        )
        published_json = data.get("published_json_file")
        if published_json is not None and (
            not isinstance(published_json, str) or not published_json.strip()
        ):
            raise ConfigError("published_json_file must be a non-empty string")
        provider = ProviderSpec.from_dict(
            _required_mapping(data, "provider")
        )
        raw_stages = data.get("stages", VALID_STAGES)
        if not isinstance(raw_stages, (list, tuple)) or not all(
            isinstance(stage, str) and stage.strip()
            for stage in raw_stages
        ):
            raise ConfigError("probe self-study stages must be a string list")
        stages = tuple(stage.strip() for stage in raw_stages)
        unknown = set(stages) - set(VALID_STAGES)
        if unknown:
            raise ConfigError(f"unknown probe self-study stages: {sorted(unknown)}")
        if not stages:
            raise ConfigError("probe self-study stages must not be empty")
        if len(stages) != len(set(stages)):
            raise ConfigError("probe self-study stages must not contain duplicates")
        if (
            "author_solve" in stages
            and "reference_score" in stages
            and stages.index("reference_score") < stages.index("author_solve")
        ):
            raise ConfigError(
                "author_solve must precede reference_score when both are requested"
            )
        max_cost = data.get("max_reported_cost_usd")
        parsed_max_cost = None
        if max_cost is not None:
            try:
                parsed_max_cost = float(max_cost)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "max_reported_cost_usd must be positive"
                ) from exc
            if isinstance(max_cost, bool) or parsed_max_cost <= 0:
                raise ConfigError("max_reported_cost_usd must be positive")
        raw_probe_ids = data.get("probe_ids", [])
        if not isinstance(raw_probe_ids, list) or not all(
            isinstance(value, str) and value.strip()
            for value in raw_probe_ids
        ):
            raise ConfigError("probe_ids must be a string list")
        return cls(
            name=_required_string(data, "name"),
            catalog_file=catalog_file,
            provider=provider,
            reference_evaluator=ReferenceEvaluator.from_dict(
                _required_mapping(data, "reference_evaluator")
            ),
            output_dir=output_dir,
            published_json_file=(
                _resolved_path(published_json, base_dir)
                if published_json
                else None
            ),
            max_parallel_calls=max(1, int(data.get("max_parallel_calls", 8))),
            structured_json_retries=max(
                0, int(data.get("structured_json_retries", 1))
            ),
            max_reported_cost_usd=parsed_max_cost,
            stages=stages,
            comparison_seed=int(data.get("comparison_seed", 0)),
            probe_ids=tuple(value.strip() for value in raw_probe_ids),
        )


def run_probe_self_study(config: ProbeSelfStudyConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = config.output_dir / "jobs.jsonl"
    journal = _load_journal(journal_path, repair_trailing=True)
    completed = _latest_successes(journal)
    spent = sum(_result_cost(row) for row in journal)
    catalog = json.loads(config.catalog_file.read_text(encoding="utf-8"))
    probes = list(catalog.get("probes", []))
    if config.probe_ids:
        requested = set(config.probe_ids)
        known = {probe["probe_id"] for probe in probes}
        if requested - known:
            raise ConfigError(
                f"unknown probe_ids: {sorted(requested - known)}"
            )
        probes = [
            probe for probe in probes if probe["probe_id"] in requested
        ]
    client = build_client(config.provider)
    client_lock = None if client.supports_parallel_requests else Lock()
    for stage in config.stages:
        if _cost_limit_reached(config, spent):
            break
        jobs = [
            probe
            for probe in probes
            if _job_key(stage, probe["probe_id"]) not in completed
        ]
        if stage == "reference_score":
            jobs = [
                probe
                for probe in jobs
                if _job_key("author_solve", probe["probe_id"]) in completed
            ]
        spent = _run_stage_wave(
            stage=stage,
            jobs=jobs,
            config=config,
            client=client,
            client_lock=client_lock,
            completed=completed,
            journal_path=journal_path,
            spent=spent,
        )

    journal = _load_journal(journal_path)
    completed = _latest_successes(journal)
    expected = {
        _job_key(stage, probe["probe_id"])
        for stage in config.stages
        for probe in probes
    }
    missing = sorted(expected - completed.keys())
    selected_catalog = {**catalog, "probes": probes}
    summary = build_probe_self_study_summary(
        config,
        selected_catalog,
        completed,
        missing_jobs=missing,
        reported_cost_usd=sum(_result_cost(row) for row in journal),
        prompt_version=PROMPT_VERSION,
    )
    (config.output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if config.published_json_file is not None:
        config.published_json_file.parent.mkdir(parents=True, exist_ok=True)
        config.published_json_file.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    (config.output_dir / "report_card.html").write_text(
        render_probe_self_study(summary),
        encoding="utf-8",
    )
    return summary


def validate_author_assessment(payload: Mapping[str, Any]) -> dict[str, Any]:
    validity = str(payload.get("probe_validity", "")).lower()
    if validity not in {"valid", "limited", "invalid"}:
        raise ValueError("probe_validity must be valid, limited, or invalid")
    checkability = str(payload.get("checkability", "")).lower()
    if checkability not in {"objective", "mixed", "subjective"}:
        raise ValueError("checkability must be objective, mixed, or subjective")
    level = str(payload.get("intended_level", "")).lower()
    if level not in {"weaker", "peer", "stronger", "mixed"}:
        raise ValueError("intended_level must be weaker, peer, stronger, or mixed")
    solvability = str(payload.get("self_solvability", "")).lower()
    if solvability not in {"fully", "partially", "not_solvable", "uncertain"}:
        raise ValueError(
            "self_solvability must be fully, partially, not_solvable, or uncertain"
        )
    predicted_score = _integer_score(payload.get("predicted_score"))
    confidence = _confidence(payload.get("confidence"))
    rationale = _string_list(payload.get("rationale"), "rationale", limit=5)
    if not rationale:
        raise ValueError("rationale must not be empty")
    return {
        "probe_validity": validity,
        "checkability": checkability,
        "intended_level": level,
        "self_solvability": solvability,
        "predicted_score": predicted_score,
        "confidence": confidence,
        "rationale": rationale,
    }


def validate_reference_score(
    payload: Mapping[str, Any], expected_answer_ids: set[str]
) -> dict[str, Any]:
    validity = str(payload.get("probe_validity", "")).lower()
    if validity not in {"valid", "limited", "invalid"}:
        raise ValueError("probe_validity must be valid, limited, or invalid")
    validity_reason = str(payload.get("validity_reason", "")).strip()
    if not validity_reason:
        raise ValueError("validity_reason must not be empty")
    rubric = _string_list(payload.get("probe_rubric"), "probe_rubric", limit=8)
    if not rubric:
        raise ValueError("probe_rubric must not be empty")
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict) or set(raw_scores) != expected_answer_ids:
        actual = set(raw_scores) if isinstance(raw_scores, dict) else set()
        raise ValueError(
            "score answer mismatch; "
            f"missing={sorted(expected_answer_ids - actual)}, "
            f"extra={sorted(actual - expected_answer_ids)}"
        )
    scores = {}
    for answer_id, row in raw_scores.items():
        if not isinstance(row, dict):
            raise ValueError(f"score for {answer_id} must be an object")
        summary = str(row.get("summary", "")).strip()
        if not summary:
            raise ValueError(f"summary for {answer_id} must not be empty")
        scores[answer_id] = {
            "score": _integer_score(row.get("score")),
            "confidence": _confidence_label(row.get("confidence")),
            "summary": summary,
            "error_tags": _string_list(
                row.get("error_tags", []), "error_tags", limit=3
            ),
        }
    return {
        "probe_validity": validity,
        "validity_reason": validity_reason,
        "probe_rubric": rubric,
        "scores": scores,
    }


def _run_stage_wave(
    *,
    stage: str,
    jobs: list[Mapping[str, Any]],
    config: ProbeSelfStudyConfig,
    client: ModelClient,
    client_lock: Lock | None,
    completed: dict[str, dict[str, Any]],
    journal_path: Path,
    spent: float,
) -> float:
    remaining = iter(jobs)
    futures: dict[Future[dict[str, Any]], Mapping[str, Any]] = {}

    def submit_one(executor: ThreadPoolExecutor) -> bool:
        nonlocal spent
        if _cost_limit_reached(config, spent):
            return False
        try:
            probe = next(remaining)
        except StopIteration:
            return False
        future = executor.submit(
            _run_job,
            stage,
            probe,
            config,
            client,
            client_lock,
            completed,
        )
        futures[future] = probe
        return True

    with ThreadPoolExecutor(max_workers=config.max_parallel_calls) as executor:
        for _ in range(config.max_parallel_calls):
            if not submit_one(executor):
                break
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                probe = futures.pop(future)
                result = _resolve_job_result(future, stage, probe)
                with journal_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                spent += _result_cost(result)
                if result["status"] == "ok":
                    completed[result["job_key"]] = result
            for _ in range(len(done)):
                if not submit_one(executor):
                    break
    return spent


def _resolve_job_result(
    future: Future[dict[str, Any]],
    stage: str,
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return future.result()
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "status": "error",
            "job_key": _job_key(stage, probe["probe_id"]),
            "stage": stage,
            "probe_id": probe["probe_id"],
            "author_model": probe["author_model"],
            "error": f"{type(exc).__name__}: {exc}",
            "attempts": getattr(exc, "attempts", []),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def _cost_limit_reached(
    config: ProbeSelfStudyConfig, spent: float
) -> bool:
    return (
        config.max_reported_cost_usd is not None
        and spent >= config.max_reported_cost_usd
    )


def _run_job(
    stage: str,
    probe: Mapping[str, Any],
    config: ProbeSelfStudyConfig,
    client: ModelClient,
    client_lock: Lock | None,
    completed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if stage == "author_solve":
        return _author_solve(probe, config, client, client_lock)
    if stage == "author_assess":
        return _author_assess(probe, config, client, client_lock)
    if stage == "reference_score":
        solve = completed[_job_key("author_solve", probe["probe_id"])]
        return _reference_score(probe, solve, config, client, client_lock)
    raise ValueError(f"unknown stage {stage}")


def _author_solve(
    probe: Mapping[str, Any],
    config: ProbeSelfStudyConfig,
    client: ModelClient,
    client_lock: Lock | None,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the supplied evaluation probe directly and substantively. "
                "Do not discuss model identity or the surrounding experiment."
            ),
        },
        {"role": "user", "content": probe["question"]},
    ]
    attempts = _call_until_valid(
        client=client,
        client_lock=client_lock,
        model=probe["author_model"],
        messages=messages,
        params=dict(probe.get("author_params", {})),
        recovery_params=dict(probe.get("author_recovery_params", {})),
        retries=1,
        validator=_validate_visible_answer,
        metadata={
            "probe_self_study": True,
            "stage": "author_solve",
            "probe_id": probe["probe_id"],
        },
    )
    return _base_result(
        "author_solve",
        probe,
        attempts,
        solution=attempts[-1]["content"],
    )


def _author_assess(
    probe: Mapping[str, Any],
    config: ProbeSelfStudyConfig,
    client: ModelClient,
    client_lock: Lock | None,
) -> dict[str, Any]:
    prompt = _author_assessment_prompt(probe["question"])
    attempts = _call_until_valid(
        client=client,
        client_lock=client_lock,
        model=probe["author_model"],
        messages=[
            {
                "role": "system",
                "content": (
                    "Assess a probe you previously authored. You are not solving "
                    "it in this call and you cannot see candidate answers."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        params=dict(probe.get("author_params", {})),
        recovery_params=dict(probe.get("author_recovery_params", {})),
        retries=config.structured_json_retries,
        validator=lambda content: validate_author_assessment(
            extract_json_object(content)
        ),
        metadata={
            "probe_self_study": True,
            "stage": "author_assess",
            "probe_id": probe["probe_id"],
        },
        repair_builder=lambda malformed, error: _structured_repair_prompt(
            prompt, malformed, error
        ),
    )
    return _base_result(
        "author_assess",
        probe,
        attempts,
        assessment=attempts[-1]["validated"],
    )


def _reference_score(
    probe: Mapping[str, Any],
    solve: Mapping[str, Any],
    config: ProbeSelfStudyConfig,
    client: ModelClient,
    client_lock: Lock | None,
) -> dict[str, Any]:
    occurrence = select_primary_occurrence(probe)
    answer_rows = _load_occurrence_answers(occurrence)
    answer_rows.append(
        {
            "kind": "author_fresh",
            "participant_id": None,
            "candidate_model": probe["author_model"],
            "candidate_score": probe.get("author_score"),
            "answer_turn_id": None,
            "answer": solve["solution"],
        }
    )
    seed = _stable_seed(config.comparison_seed, probe["probe_id"])
    random.Random(seed).shuffle(answer_rows)
    labeled = [
        {"answer_id": f"A{index:02d}", **row}
        for index, row in enumerate(answer_rows, start=1)
    ]
    prompt = _reference_prompt(probe["question"], labeled)
    expected_ids = {row["answer_id"] for row in labeled}
    attempts = _call_until_valid(
        client=client,
        client_lock=client_lock,
        model=config.reference_evaluator.model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Evaluate anonymous answers for substantive correctness. "
                    "Use only the supplied probe and answers. Do not infer model "
                    "identity or reward style, verbosity, or familiarity."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        params=config.reference_evaluator.params,
        recovery_params=config.reference_evaluator.recovery_params,
        retries=config.structured_json_retries,
        validator=lambda content: validate_reference_score(
            extract_json_object(content), expected_ids
        ),
        metadata={
            "probe_self_study": True,
            "stage": "reference_score",
            "probe_id": probe["probe_id"],
        },
        repair_builder=lambda malformed, error: _structured_repair_prompt(
            prompt, malformed, error
        ),
    )
    answer_map = {
        row["answer_id"]: {
            key: row.get(key)
            for key in (
                "kind",
                "participant_id",
                "candidate_model",
                "candidate_score",
                "answer_turn_id",
            )
        }
        for row in labeled
    }
    return _base_result(
        "reference_score",
        probe,
        attempts,
        source_occurrence={
            key: occurrence.get(key)
            for key in ("run_dir", "question_turn_id", "source_probe_id")
        },
        evaluator_model=config.reference_evaluator.model,
        answer_map=answer_map,
        evaluation=attempts[-1]["validated"],
    )


def _call_until_valid(
    *,
    client: ModelClient,
    client_lock: Lock | None,
    model: str,
    messages: list[dict[str, str]],
    params: Mapping[str, Any],
    recovery_params: Mapping[str, Any],
    retries: int,
    validator: Callable[[str], Any],
    metadata: Mapping[str, Any],
    repair_builder: Callable[[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    attempts = []
    latest_error: Exception | None = None
    for attempt_index in range(retries + 1):
        use_recovery = attempt_index > 0 and bool(recovery_params)
        request_messages = messages
        if attempt_index and repair_builder is not None:
            request_messages = [
                messages[0],
                {
                    "role": "user",
                    "content": repair_builder(
                        attempts[-1]["content"], str(latest_error)
                    ),
                },
            ]
        response = _generate(
            client,
            client_lock,
            ModelRequest(
                model=model,
                messages=request_messages,
                params=dict(recovery_params if use_recovery else params),
                metadata={
                    **metadata,
                    "attempt_index": attempt_index,
                    "recovery": use_recovery,
                },
            ),
        )
        attempt = {
            "attempt_index": attempt_index,
            "model": response.model or model,
            "usage": response.usage,
            "content": response.content,
            "finish_reason": _finish_reason(response.raw),
        }
        attempts.append(attempt)
        try:
            if attempt["finish_reason"] == "length":
                raise ValueError("response was truncated")
            attempt["validated"] = validator(response.content)
            return attempts
        except (JsonExtractionError, ValueError) as exc:
            latest_error = exc
    error = RuntimeError(f"invalid response after retries: {latest_error}")
    error.attempts = attempts  # type: ignore[attr-defined]
    raise error


def _validate_visible_answer(content: str) -> str:
    if not content.strip():
        raise ValueError("answer is empty")
    return content


def _load_occurrence_answers(
    occurrence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    run_dir = Path(str(occurrence["run_dir"]))
    turns = {
        int(row["turn_id"]): row
        for row in _load_jsonl(run_dir / "transcript.jsonl")
    }
    rows = []
    for answer in occurrence.get("candidate_answers", []):
        if answer.get("unavailable"):
            continue
        turn = turns.get(int(answer["answer_turn_id"]))
        if turn is None or not str(turn.get("content", "")).strip():
            continue
        rows.append(
            {
                "kind": "candidate",
                **answer,
                "answer": str(turn["content"]),
            }
        )
    if not rows:
        raise ValueError(
            f"probe occurrence {occurrence.get('source_probe_id')} has no answers"
        )
    return rows


def _author_assessment_prompt(question: str) -> str:
    task = {
        "task": (
            "Assess this probe you previously authored. Do not solve it and do "
            "not speculate about candidate identities."
        ),
        "probe": question,
        "output_schema": {
            "probe_validity": "valid | limited | invalid",
            "checkability": "objective | mixed | subjective",
            "intended_level": "weaker | peer | stronger | mixed",
            "self_solvability": "fully | partially | not_solvable | uncertain",
            "predicted_score": "integer 0-4 on the anchored scale below",
            "confidence": "number 0-1",
            "rationale": "one to five concise evidence-based strings",
        },
        "anchored_scale": {
            "0": "No usable answer or fundamentally wrong.",
            "1": "Minimal progress with central errors.",
            "2": "Meaningful partial progress with important errors or omissions.",
            "3": "Substantially correct with only minor gaps.",
            "4": "Fully correct, complete, rigorous, and responsive.",
        },
        "rules": [
            "Predict how well you could answer now without external tools.",
            "Treat uncertainty honestly.",
            (
                "Return the output_schema keys at the top level. Do not repeat "
                "the task, probe, or output_schema wrapper."
            ),
            "Return one JSON object only.",
        ],
    }
    return json.dumps(task, ensure_ascii=False, indent=2)


def _reference_prompt(
    question: str, labeled_answers: list[Mapping[str, Any]]
) -> str:
    task = {
        "task": (
            "Assess whether the probe is well-posed and jointly score every "
            "anonymous answer using one probe-specific rubric."
        ),
        "probe": question,
        "validity_labels": {
            "valid": "A strong answer can be judged substantively and reliably.",
            "limited": "Some evidence is usable, but ambiguity or subjectivity weakens it.",
            "invalid": "No defensible substantive comparison is possible.",
        },
        "anchored_scale": {
            "0": "No usable answer, refusal, or fundamentally wrong throughout.",
            "1": "Minimal progress; major errors defeat the central result.",
            "2": "Meaningful partial progress but important errors or omissions remain.",
            "3": "Substantially correct; only minor errors, gaps, or imprecision.",
            "4": "Fully correct, complete, rigorous, and responsive.",
        },
        "rules": [
            "Use integer scores only.",
            "Apply one consistent rubric to all answers.",
            "Do not reward verbosity, polish, confidence, or familiar phrasing.",
            "Keep each summary under 30 words.",
            "Return exactly one score entry for every answer_id.",
            "Return JSON only.",
        ],
        "output_schema": {
            "probe_validity": "valid | limited | invalid",
            "validity_reason": "one concise string",
            "probe_rubric": ["three to eight short criteria"],
            "scores": {
                "<answer_id>": {
                    "score": "integer 0-4",
                    "confidence": "low | medium | high",
                    "summary": "short decisive assessment",
                    "error_tags": ["zero to three short labels"],
                }
            },
        },
        "answers": [
            {"answer_id": row["answer_id"], "answer": row["answer"]}
            for row in labeled_answers
        ],
    }
    return json.dumps(task, ensure_ascii=False, indent=2)


def _structured_repair_prompt(
    original_prompt: str, malformed: str, error: str
) -> str:
    return (
        f"{original_prompt}\n\nYour prior output was invalid.\n"
        f"Validation error: {error}\n"
        f"Prior output:\n{malformed}\n\n"
        "Re-evaluate from the complete evidence and return one valid JSON object. "
        "Put every required output key at the top level; do not repeat the prompt "
        "or schema wrapper."
    )


def _base_result(
    stage: str,
    probe: Mapping[str, Any],
    attempts: list[dict[str, Any]],
    **fields: Any,
) -> dict[str, Any]:
    cleaned_attempts = [
        {key: value for key, value in attempt.items() if key != "validated"}
        for attempt in attempts
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "status": "ok",
        "job_key": _job_key(stage, probe["probe_id"]),
        "stage": stage,
        "probe_id": probe["probe_id"],
        "author_model": probe["author_model"],
        "attempts": cleaned_attempts,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **fields,
    }


def _generate(
    client: ModelClient,
    client_lock: Lock | None,
    request: ModelRequest,
) -> ModelResponse:
    if client_lock is None:
        return client.generate(request)
    with client_lock:
        return client.generate(request)


def _load_journal(
    path: Path, *, repair_trailing: bool = False
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or not repair_trailing:
                raise
            path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in rows
                ),
                encoding="utf-8",
            )
    return rows


def _latest_successes(
    journal: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        row["job_key"]: row
        for row in journal
        if row.get("status") == "ok" and isinstance(row.get("job_key"), str)
    }


def _result_cost(row: Mapping[str, Any]) -> float:
    total = 0.0
    for attempt in row.get("attempts", []):
        usage = attempt.get("usage") or {}
        for key in ("cost", "reported_cost_usd", "total_cost"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                total += float(value)
                break
    return total


def _finish_reason(raw: Mapping[str, Any]) -> str | None:
    try:
        value = raw["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return str(value) if value is not None else None


def _integer_score(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be numeric")
    if float(value) not in {0.0, 1.0, 2.0, 3.0, 4.0}:
        raise ValueError("score must be an integer from 0 to 4")
    return int(value)


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be numeric")
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return parsed


def _confidence_label(value: Any) -> str:
    parsed = str(value).lower()
    if parsed not in {"low", "medium", "high"}:
        raise ValueError("confidence must be low, medium, or high")
    return parsed


def _string_list(value: Any, field: str, *, limit: int) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a string list")
    return [item.strip() for item in value[:limit]]


def _stable_seed(seed: int, probe_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{probe_id}".encode()).digest()[:8], "big"
    )


def _job_key(stage: str, probe_id: str) -> str:
    return f"{stage}:{probe_id}"


def _resolved_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _required_mapping(
    data: Mapping[str, Any], key: str
) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run blind author solves, separate author self-assessments, and "
            "fixed-reference answer scoring over a probe catalog."
        )
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    load_dotenv()
    config = ProbeSelfStudyConfig.from_path(args.config)
    summary = run_probe_self_study(config)
    print(
        json.dumps(
            {
                "output_dir": str(config.output_dir),
                "probe_count": summary["probe_count"],
                "scored_probe_count": summary["scored_probe_count"],
                "beyond_author_count": summary["beyond_author_count"],
                "missing_jobs": summary["missing_jobs"],
                "reported_cost_usd": summary["reported_cost_usd"],
            },
            indent=2,
        )
    )
    return 0 if not summary["missing_jobs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
