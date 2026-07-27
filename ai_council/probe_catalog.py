from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping
import unicodedata


SCHEMA_VERSION = "probe-catalog-v1"


def build_probe_catalog(
    study_summary: str | Path,
    *,
    model_catalog: str | Path,
) -> dict[str, Any]:
    study_path = Path(study_summary)
    study = _load_json(study_path)
    model_catalog_path = Path(model_catalog)
    catalog = _load_json(model_catalog_path)
    model_rows = {
        row["provider_model_id"]: row for row in catalog.get("models", [])
    }
    grouped: dict[str, dict[str, Any]] = {}
    for run_record in study.get("runs", []):
        run_dir = Path(run_record["run_dir"])
        config = _load_json(run_dir / "config.json")
        turns = _load_jsonl(run_dir / "transcript.jsonl")
        judge_model, judge_params, judge_recovery = _judge_runtime(config)
        probe_annotations = {
            str(row["probe_id"]): row for row in run_record.get("probes", [])
        }
        participant_models = _participant_models(config)
        prior_scores = _participant_prior_scores(run_dir)
        for question in turns:
            metadata = question.get("metadata") or {}
            if metadata.get("interaction_role") != "question":
                continue
            content = str(question.get("content", "")).strip()
            if not content or metadata.get("finish_reason") == "length":
                continue
            probe_id = str(metadata.get("probe_id", ""))
            annotation = probe_annotations.get(probe_id)
            if annotation is None:
                continue
            stable_id = probe_catalog_id(judge_model, content)
            record = grouped.setdefault(
                stable_id,
                {
                    "probe_id": stable_id,
                    "content_sha256": _content_hash(content),
                    "author_model": judge_model,
                    "author_score": _model_score(model_rows.get(judge_model)),
                    "question": content,
                    "author_params": judge_params,
                    "author_recovery_params": judge_recovery,
                    "occurrences": [],
                },
            )
            if record["question"] != content:
                raise ValueError(f"probe hash collision for {stable_id}")
            answer_rows = []
            for answer in turns:
                answer_metadata = answer.get("metadata") or {}
                if (
                    answer_metadata.get("interaction_role") != "answer"
                    or answer_metadata.get("question_turn_id")
                    != question.get("turn_id")
                ):
                    continue
                participant_id = str(answer.get("speaker", ""))
                answer_rows.append(
                    {
                        "participant_id": participant_id,
                        "candidate_model": participant_models.get(participant_id),
                        "candidate_score": prior_scores.get(participant_id),
                        "answer_turn_id": int(answer["turn_id"]),
                        "unavailable": bool(
                            answer_metadata.get("answer_unavailable")
                        ),
                    }
                )
            record["occurrences"].append(
                {
                    "cohort": run_record["cohort"],
                    "run_dir": str(run_dir),
                    "run_name": run_record["run_name"],
                    "source_probe_id": probe_id,
                    "question_turn_id": int(question["turn_id"]),
                    "round_index": int(question.get("round_index") or 1),
                    "sequence": int(annotation.get("sequence") or 0),
                    "stage": annotation.get("stage"),
                    "question_types": list(annotation.get("question_types", [])),
                    "strategy_tags": list(annotation.get("strategy_tags", [])),
                    "transition": annotation.get("transition"),
                    "validity": annotation.get("validity"),
                    "pair_accuracy": annotation.get("pair_accuracy"),
                    "decided_pair_accuracy": annotation.get(
                        "decided_pair_accuracy"
                    ),
                    "tie_pair_rate": annotation.get("tie_pair_rate"),
                    "candidate_answers": sorted(
                        answer_rows,
                        key=lambda row: row["participant_id"],
                    ),
                }
            )
    probes = sorted(
        grouped.values(),
        key=lambda row: (
            float(row["author_score"])
            if isinstance(row["author_score"], (int, float))
            else float("inf"),
            row["author_model"],
            row["probe_id"],
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_study": str(study_path),
        "source_study_schema": study.get("schema_version"),
        "model_catalog": str(model_catalog_path),
        "model_catalog_generated_at": catalog.get("generated_at"),
        "accepted_run_count": len(study.get("runs", [])),
        "probe_occurrence_count": sum(
            len(probe["occurrences"]) for probe in probes
        ),
        "unique_probe_count": len(probes),
        "author_model_count": len({probe["author_model"] for probe in probes}),
        "probes": probes,
    }


def write_probe_catalog(catalog: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def probe_catalog_id(author_model: str, content: str) -> str:
    digest = hashlib.sha256(
        f"{author_model}\0{normalize_probe_text(content)}".encode("utf-8")
    ).hexdigest()
    return f"probe_{digest[:20]}"


def normalize_probe_text(content: str) -> str:
    normalized = unicodedata.normalize("NFKC", content)
    return re.sub(r"\s+", " ", normalized).strip()


def select_primary_occurrence(probe: Mapping[str, Any]) -> Mapping[str, Any]:
    occurrences = probe.get("occurrences", [])
    if not isinstance(occurrences, list) or not occurrences:
        raise ValueError(f"probe {probe.get('probe_id')} has no occurrences")
    return max(
        occurrences,
        key=lambda row: (
            sum(
                not answer.get("unavailable")
                for answer in row.get("candidate_answers", [])
            ),
            len(row.get("candidate_answers", [])),
            -int(row.get("question_turn_id") or 0),
        ),
    )


def _judge_runtime(
    config: Mapping[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    judges = config.get("judges", [])
    if not isinstance(judges, list) or len(judges) != 1:
        raise ValueError("probe catalog currently requires exactly one judge")
    model_ref = judges[0]["model"]
    model = _model_by_ref(config, model_ref)
    return (
        str(model["model"]),
        dict(model.get("params", {})),
        dict(model.get("recovery_params", {})),
    )


def _participant_models(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(participant["id"]): str(
            _model_by_ref(config, participant["model"])["model"]
        )
        for participant in config.get("participants", [])
    }


def _model_by_ref(
    config: Mapping[str, Any], model_ref: str
) -> Mapping[str, Any]:
    models = config.get("models", {})
    if isinstance(models, dict):
        return models[model_ref]
    for row in models:
        if row.get("name") == model_ref:
            return row
    raise ValueError(f"unknown model reference {model_ref}")


def _participant_prior_scores(run_dir: Path) -> dict[str, float]:
    path = run_dir / "analysis_summary.json"
    if not path.exists():
        return {}
    analysis = _load_json(path)
    raw = analysis.get("prior_agreement", {}).get(
        "participant_prior_scores", {}
    )
    return {
        str(participant_id): float(score)
        for participant_id, score in raw.items()
        if isinstance(score, (int, float))
    }


def _model_score(model: Mapping[str, Any] | None) -> float | None:
    if not model:
        return None
    score = model.get("intelligence_score")
    return float(score) if isinstance(score, (int, float)) else None


def _content_hash(content: str) -> str:
    return hashlib.sha256(normalize_probe_text(content).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
