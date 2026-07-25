from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from threading import Lock
from typing import Any

from ai_council.clients.base import ModelClient
from ai_council.clients.registry import build_client
from ai_council.config import ConfigError, ProviderSpec
from ai_council.core import ModelRequest
from ai_council.json_tools import JsonExtractionError, extract_json_object


SCHEMA_VERSION = "1"
PROMPT_VERSION = "probe_answer_quality_v1"


class ScorePayloadError(ValueError):
    def __init__(self, message: str, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True)
class ScoringJudge:
    id: str
    provider: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    recovery_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoringJudge":
        return cls(
            id=_required_string(data, "id"),
            provider=_required_string(data, "provider"),
            model=_required_string(data, "model"),
            params=dict(data.get("params", {})),
            recovery_params=dict(data.get("recovery_params", {})),
        )


@dataclass(frozen=True)
class ProbeSource:
    label: str
    run_dir: Path
    question_turn_ids: tuple[int, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any], base_dir: Path) -> "ProbeSource":
        run_dir = Path(_required_string(data, "run_dir"))
        if not run_dir.is_absolute():
            run_dir = base_dir / run_dir
        raw_turn_ids = data.get("question_turn_ids")
        if (
            not isinstance(raw_turn_ids, list)
            or not raw_turn_ids
            or not all(isinstance(item, int) and item > 0 for item in raw_turn_ids)
        ):
            raise ConfigError("probe source question_turn_ids must be positive integers")
        return cls(
            label=_required_string(data, "label"),
            run_dir=run_dir,
            question_turn_ids=tuple(raw_turn_ids),
        )


@dataclass(frozen=True)
class ProbeScoringConfig:
    name: str
    providers: dict[str, ProviderSpec]
    judges: tuple[ScoringJudge, ...]
    sources: tuple[ProbeSource, ...]
    output_dir: Path
    export_path: Path | None = None
    comparison_seed: int = 0
    max_parallel_calls: int = 4
    structured_json_retries: int = 1
    max_reported_cost_usd: float | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "ProbeScoringConfig":
        config_path = Path(path).resolve()
        data = json.loads(config_path.read_text())
        if not isinstance(data, dict):
            raise ConfigError("probe scoring config must be a JSON object")
        base_dir = (
            config_path.parents[1]
            if config_path.parent.name == "examples"
            else Path.cwd()
        )
        providers = [
            ProviderSpec.from_dict(item) for item in _required_list(data, "providers")
        ]
        provider_map = {provider.name: provider for provider in providers}
        if len(provider_map) != len(providers):
            raise ConfigError("probe scoring provider names must be unique")
        judges = tuple(
            ScoringJudge.from_dict(item) for item in _required_list(data, "judges")
        )
        if not judges:
            raise ConfigError("probe scoring config requires at least one judge")
        if len({judge.id for judge in judges}) != len(judges):
            raise ConfigError("probe scoring judge ids must be unique")
        unknown = sorted({judge.provider for judge in judges} - provider_map.keys())
        if unknown:
            raise ConfigError(
                f"probe scoring judges reference unknown providers: {unknown}"
            )
        sources = tuple(
            ProbeSource.from_dict(item, base_dir)
            for item in _required_list(data, "sources")
        )
        output_dir = Path(_required_string(data, "output_dir"))
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
        raw_export_path = data.get("export_path")
        if raw_export_path is not None and (
            not isinstance(raw_export_path, str) or not raw_export_path.strip()
        ):
            raise ConfigError("probe scoring export_path must be a non-empty string")
        export_path = Path(raw_export_path) if raw_export_path else None
        if export_path is not None and not export_path.is_absolute():
            export_path = base_dir / export_path
        max_cost = data.get("max_reported_cost_usd")
        return cls(
            name=_required_string(data, "name"),
            providers=provider_map,
            judges=judges,
            sources=sources,
            output_dir=output_dir,
            export_path=export_path,
            comparison_seed=int(data.get("comparison_seed", 0)),
            max_parallel_calls=max(1, int(data.get("max_parallel_calls", 4))),
            structured_json_retries=max(
                0, int(data.get("structured_json_retries", 1))
            ),
            max_reported_cost_usd=float(max_cost) if max_cost is not None else None,
        )


@dataclass(frozen=True)
class ProbeEvidence:
    id: str
    source_label: str
    run_dir: str
    question_turn_id: int
    question: str
    answers: dict[str, str]
    unavailable_candidates: tuple[str, ...]


def load_probe_evidence(sources: tuple[ProbeSource, ...]) -> list[ProbeEvidence]:
    evidence = []
    seen_ids: set[str] = set()
    for source in sources:
        transcript_path = source.run_dir / "transcript.jsonl"
        if not transcript_path.exists():
            raise FileNotFoundError(transcript_path)
        turns = [json.loads(line) for line in transcript_path.read_text().splitlines()]
        by_id = {turn["turn_id"]: turn for turn in turns}
        for question_turn_id in source.question_turn_ids:
            question = by_id.get(question_turn_id)
            if question is None:
                raise ConfigError(
                    f"{source.run_dir} has no question turn {question_turn_id}"
                )
            answers: dict[str, str] = {}
            unavailable: set[str] = set()
            for turn in turns:
                metadata = turn.get("metadata") or {}
                if (
                    metadata.get("interaction_role") != "answer"
                    or metadata.get("question_turn_id") != question_turn_id
                ):
                    continue
                candidate_id = str(turn.get("speaker", ""))
                content = str(turn.get("content", "")).strip()
                if content:
                    answers[candidate_id] = content
                else:
                    unavailable.add(candidate_id)
            if not answers:
                raise ConfigError(
                    f"{source.run_dir} question turn {question_turn_id} has no answers"
                )
            probe_id = f"{source.label}:turn_{question_turn_id}"
            if probe_id in seen_ids:
                raise ConfigError(f"duplicate probe evidence id {probe_id!r}")
            seen_ids.add(probe_id)
            evidence.append(
                ProbeEvidence(
                    id=probe_id,
                    source_label=source.label,
                    run_dir=str(source.run_dir),
                    question_turn_id=question_turn_id,
                    question=str(question["content"]),
                    answers=answers,
                    unavailable_candidates=tuple(sorted(unavailable)),
                )
            )
    return evidence


def score_probe_answers(config: ProbeScoringConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = config.output_dir / "score_journal.jsonl"
    journal = _load_journal(journal_path, repair_trailing=True)
    existing = _latest_successful_results(journal)
    spent = sum(_result_cost(item) for item in journal)
    if (
        config.max_reported_cost_usd is not None
        and spent >= config.max_reported_cost_usd
    ):
        raise RuntimeError(
            f"existing reported spend ${spent:.4f} meets configured limit "
            f"${config.max_reported_cost_usd:.4f}"
        )

    evidence = load_probe_evidence(config.sources)
    clients = {
        judge.id: build_client(config.providers[judge.provider])
        for judge in config.judges
    }
    client_locks = {
        judge.id: Lock()
        for judge in config.judges
        if not clients[judge.id].supports_parallel_requests
    }
    jobs = [
        (
            judge,
            probe,
            any(
                item.get("job_key") == _job_key(judge.id, probe.id)
                and item.get("status") == "error"
                for item in journal
            ),
        )
        for judge in config.judges
        for probe in evidence
        if _job_key(judge.id, probe.id) not in existing
    ]
    journal_lock = Lock()
    with ThreadPoolExecutor(max_workers=config.max_parallel_calls) as executor:
        futures = {
            executor.submit(
                _score_one,
                judge,
                probe,
                clients[judge.id],
                client_locks.get(judge.id),
                config,
                start_with_recovery,
            ): (judge, probe)
            for judge, probe, start_with_recovery in jobs
        }
        for future in as_completed(futures):
            judge, probe = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "status": "error",
                    "job_key": _job_key(judge.id, probe.id),
                    "judge_id": judge.id,
                    "judge_model": judge.model,
                    "probe_id": probe.id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "attempts": (
                        exc.attempts if isinstance(exc, ScorePayloadError) else []
                    ),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            with journal_lock:
                with journal_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            if result["status"] == "ok":
                existing[result["job_key"]] = result

    expected_keys = {
        _job_key(judge.id, probe.id)
        for judge in config.judges
        for probe in evidence
    }
    missing = sorted(expected_keys - existing.keys())
    total_cost = sum(_result_cost(item) for item in _load_journal(journal_path))
    summary = build_scoring_summary(
        config,
        evidence,
        list(existing.values()),
        missing,
        reported_cost_usd=total_cost,
    )
    (config.output_dir / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    if config.export_path is not None:
        write_export_summary(summary, config.export_path)
    return summary


def build_scoring_summary(
    config: ProbeScoringConfig,
    evidence: list[ProbeEvidence],
    results: list[dict[str, Any]],
    missing_jobs: list[str] | None = None,
    reported_cost_usd: float | None = None,
) -> dict[str, Any]:
    by_probe: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") == "ok":
            by_probe.setdefault(result["probe_id"], []).append(result)
    probes = []
    for probe in evidence:
        probe_results = sorted(
            by_probe.get(probe.id, []), key=lambda item: item["judge_id"]
        )
        candidate_scores: dict[str, list[float]] = {}
        for result in probe_results:
            for candidate_id, item in result["scores"].items():
                candidate_scores.setdefault(candidate_id, []).append(
                    float(item["score"])
                )
        averaged = {
            candidate_id: mean(values)
            for candidate_id, values in candidate_scores.items()
        }
        all_scores = list(averaged.values())
        disagreement = []
        if len(probe_results) >= 2:
            common = set(probe_results[0]["scores"])
            for result in probe_results[1:]:
                common &= set(result["scores"])
            for candidate_id in common:
                values = [
                    float(result["scores"][candidate_id]["score"])
                    for result in probe_results
                ]
                disagreement.append(max(values) - min(values))
        probes.append(
            {
                "probe_id": probe.id,
                "source_label": probe.source_label,
                "run_dir": probe.run_dir,
                "question_turn_id": probe.question_turn_id,
                "question": probe.question,
                "available_answer_count": len(probe.answers),
                "unavailable_candidates": list(probe.unavailable_candidates),
                "judge_count": len(probe_results),
                "judge_results": [
                    _compact_judge_result(result) for result in probe_results
                ],
                "mean_scores": averaged,
                "mean_answer_score": mean(all_scores) if all_scores else None,
                "substantially_correct_rate": (
                    sum(value >= 3 for value in all_scores) / len(all_scores)
                    if all_scores
                    else None
                ),
                "mean_judge_score_range": (
                    mean(disagreement) if disagreement else None
                ),
            }
        )
    total_cost = (
        reported_cost_usd
        if reported_cost_usd is not None
        else sum(_result_cost(item) for item in results)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "name": config.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "judge_ids": [judge.id for judge in config.judges],
        "judge_models": {judge.id: judge.model for judge in config.judges},
        "probe_count": len(evidence),
        "completed_job_count": len(
            [item for item in results if item.get("status") == "ok"]
        ),
        "missing_jobs": missing_jobs or [],
        "reported_cost_usd": total_cost,
        "probes": probes,
    }


def validate_score_payload(
    payload: dict[str, Any], expected_candidates: set[str]
) -> dict[str, Any]:
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("score payload must contain a scores object")
    actual_candidates = set(scores)
    if actual_candidates != expected_candidates:
        missing = sorted(expected_candidates - actual_candidates)
        extra = sorted(actual_candidates - expected_candidates)
        raise ValueError(f"score candidate mismatch; missing={missing}, extra={extra}")
    normalized = {}
    for candidate_id, item in scores.items():
        if not isinstance(item, dict):
            raise ValueError(f"score for {candidate_id} must be an object")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"score for {candidate_id} must be numeric")
        if float(score) not in {0.0, 1.0, 2.0, 3.0, 4.0}:
            raise ValueError(f"score for {candidate_id} must be an integer from 0 to 4")
        confidence = str(item.get("confidence", "")).lower()
        if confidence not in {"low", "medium", "high"}:
            raise ValueError(
                f"confidence for {candidate_id} must be low, medium, or high"
            )
        summary = str(item.get("summary", "")).strip()
        if not summary:
            raise ValueError(f"summary for {candidate_id} must not be empty")
        error_tags = item.get("error_tags", [])
        if not isinstance(error_tags, list) or not all(
            isinstance(tag, str) for tag in error_tags
        ):
            raise ValueError(f"error_tags for {candidate_id} must be a string list")
        normalized[candidate_id] = {
            "score": int(score),
            "confidence": confidence,
            "summary": summary,
            "error_tags": error_tags[:3],
        }
    rubric = payload.get("probe_rubric", [])
    if not isinstance(rubric, list) or not all(
        isinstance(item, str) for item in rubric
    ):
        raise ValueError("probe_rubric must be a string list")
    return {"probe_rubric": rubric[:8], "scores": normalized}


def _score_one(
    judge: ScoringJudge,
    probe: ProbeEvidence,
    client: ModelClient,
    client_lock: Lock | None,
    config: ProbeScoringConfig,
    start_with_recovery: bool = False,
) -> dict[str, Any]:
    answer_order = sorted(probe.answers)
    stable_seed = int.from_bytes(
        hashlib.sha256(
            f"{config.comparison_seed}:{probe.id}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    random.Random(stable_seed).shuffle(answer_order)
    prompt = _scoring_prompt(probe, answer_order)
    messages = [
        {
            "role": "system",
            "content": (
                "You evaluate anonymous model answers for substantive correctness. "
                "Use only the supplied probe and answers. Do not infer identities, "
                "reward style or verbosity, or import external model rankings."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    attempts = []
    payload = None
    parse_error = None
    for attempt_index in range(config.structured_json_retries + 1):
        use_recovery = start_with_recovery or attempt_index > 0
        request_params = (
            judge.recovery_params
            if use_recovery and judge.recovery_params
            else judge.params
        )
        request_messages = messages
        if attempt_index:
            request_messages = [
                messages[0],
                {
                    "role": "user",
                    "content": _repair_prompt(
                        probe, answer_order, attempts[-1]["content"], str(parse_error)
                    ),
                },
            ]
        try:
            response = _generate(
                client,
                client_lock,
                ModelRequest(
                    model=judge.model,
                    messages=request_messages,
                    params=request_params,
                    metadata={
                        "probe_scoring": True,
                        "judge_id": judge.id,
                        "probe_id": probe.id,
                        "attempt_index": attempt_index,
                        "recovery": use_recovery,
                    },
                ),
            )
        except Exception as exc:
            if attempts:
                raise ScorePayloadError(
                    f"repair request failed: {type(exc).__name__}: {exc}",
                    attempts,
                ) from exc
            raise
        attempts.append(
            {
                "attempt_index": attempt_index,
                "model": response.model or judge.model,
                "usage": response.usage,
                "content": response.content,
                "finish_reason": _finish_reason(response.raw),
            }
        )
        try:
            payload = validate_score_payload(
                extract_json_object(response.content), set(answer_order)
            )
            break
        except (JsonExtractionError, ValueError) as exc:
            parse_error = exc
    if payload is None:
        raise ScorePayloadError(
            f"invalid score payload after retries: {parse_error}", attempts
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "status": "ok",
        "job_key": _job_key(judge.id, probe.id),
        "judge_id": judge.id,
        "judge_model": judge.model,
        "probe_id": probe.id,
        "source_label": probe.source_label,
        "question_turn_id": probe.question_turn_id,
        "answer_order": answer_order,
        "unavailable_candidates": list(probe.unavailable_candidates),
        "probe_rubric": payload["probe_rubric"],
        "scores": payload["scores"],
        "attempts": attempts,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _generate(
    client: ModelClient, client_lock: Lock | None, request: ModelRequest
):
    if client_lock is None:
        return client.generate(request)
    with client_lock:
        return client.generate(request)


def _scoring_prompt(probe: ProbeEvidence, answer_order: list[str]) -> str:
    answers = [
        {"candidate_id": candidate_id, "answer": probe.answers[candidate_id]}
        for candidate_id in answer_order
    ]
    task = {
        "task": (
            "Score every anonymous answer jointly. First derive a concise "
            "probe-specific correctness rubric, then apply the anchored scale. "
            "A score is answer quality on this probe, not an estimate of general "
            "model intelligence."
        ),
        "anchored_scale": {
            "0": "No usable answer, refusal, or fundamentally wrong throughout.",
            "1": "Minimal progress; major errors defeat the central result.",
            "2": "Meaningful partial progress but important errors or omissions remain.",
            "3": "Substantially correct; only minor errors, gaps, or imprecision.",
            "4": "Fully correct, complete, rigorous, and responsive to the key demands.",
        },
        "rules": [
            "Use integer scores only.",
            (
                "Judge correctness, reasoning, completeness, and instruction "
                "compliance only when substantively relevant."
            ),
            "Do not reward verbosity, polish, confidence, or familiar phrasing.",
            "Use all answers comparatively to keep thresholds consistent.",
            (
                "Keep each summary under 25 words and ground it in decisive "
                "evidence."
            ),
            (
                "Use at most three short error_tags; use an empty list for fully "
                "correct answers."
            ),
            "Return exactly one score entry for every supplied candidate_id.",
            "Return JSON only.",
        ],
        "output_schema": {
            "probe_rubric": ["three to eight short correctness criteria"],
            "scores": {
                "<candidate_id>": {
                    "score": "integer 0, 1, 2, 3, or 4",
                    "confidence": "low | medium | high",
                    "summary": "short evidence-grounded assessment",
                    "error_tags": ["short error label"],
                }
            },
        },
        "probe_id": probe.id,
        "probe": probe.question,
        "answers": answers,
    }
    return json.dumps(task, ensure_ascii=False, indent=2)


def _repair_prompt(
    probe: ProbeEvidence,
    answer_order: list[str],
    malformed: str,
    error: str,
) -> str:
    task = json.loads(_scoring_prompt(probe, answer_order))
    task["repair"] = {
        "instruction": (
            "Your previous output was invalid. Re-evaluate from the complete "
            "evidence above and return one valid JSON object matching the schema."
        ),
        "parse_or_validation_error": error,
        "previous_output": malformed,
    }
    return json.dumps(task, ensure_ascii=False, indent=2)


def _load_journal(
    path: Path, *, repair_trailing: bool = False
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    items = []
    for index, line in enumerate(lines):
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            if repair_trailing:
                path.write_text(
                    "".join(
                        json.dumps(item, ensure_ascii=False) + "\n"
                        for item in items
                    ),
                    encoding="utf-8",
                )
    return items


def _latest_successful_results(
    journal: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for item in journal:
        if item.get("status") == "ok" and isinstance(item.get("job_key"), str):
            results[item["job_key"]] = item
    return results


def _compact_judge_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "judge_id": result["judge_id"],
        "judge_model": result["judge_model"],
        "probe_rubric": result["probe_rubric"],
        "scores": result["scores"],
        "answer_order": result["answer_order"],
        "reported_cost_usd": _result_cost(result),
    }


def write_export_summary(summary: dict[str, Any], path: Path) -> None:
    exported = json.loads(json.dumps(summary))
    for probe in exported.get("probes", []):
        if probe.get("run_dir"):
            probe["run_dir"] = Path(probe["run_dir"]).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _finish_reason(raw: dict[str, Any]) -> str | None:
    try:
        value = raw["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return str(value) if value is not None else None


def _result_cost(result: dict[str, Any]) -> float:
    total = 0.0
    for attempt in result.get("attempts", []):
        usage = attempt.get("usage") or {}
        for key in ("cost", "reported_cost_usd", "total_cost"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                total += float(value)
                break
    return total


def _job_key(judge_id: str, probe_id: str) -> str:
    return f"{judge_id}:{probe_id}"


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value


def _required_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ConfigError(f"{key} must be a list of objects")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Jointly score archived candidate answers probe by probe."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = ProbeScoringConfig.from_path(args.config)
    summary = score_probe_answers(config)
    print(
        json.dumps(
            {
                "output_dir": str(config.output_dir),
                "completed_jobs": summary["completed_job_count"],
                "missing_jobs": summary["missing_jobs"],
                "reported_cost_usd": summary["reported_cost_usd"],
            },
            indent=2,
        )
    )
    return 0 if not summary["missing_jobs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
