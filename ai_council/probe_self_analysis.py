from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from html import escape
from statistics import mean
from typing import Any, Mapping

from ai_council.probe_catalog import select_primary_occurrence


SCHEMA_VERSION = "probe-self-study-v1"


def build_probe_self_study_summary(
    config: Any,
    catalog: Mapping[str, Any],
    completed: Mapping[str, Mapping[str, Any]],
    *,
    missing_jobs: list[str],
    reported_cost_usd: float,
    prompt_version: str,
) -> dict[str, Any]:
    rows = []
    for probe in catalog.get("probes", []):
        probe_id = probe["probe_id"]
        primary_occurrence = select_primary_occurrence(probe)
        solve = completed.get(_job_key("author_solve", probe_id))
        assess = completed.get(_job_key("author_assess", probe_id))
        reference = completed.get(_job_key("reference_score", probe_id))
        row = {
            "probe_id": probe_id,
            "author_model": probe["author_model"],
            "author_score": probe.get("author_score"),
            "question_excerpt": _excerpt(probe["question"], 220),
            "occurrence_count": len(probe.get("occurrences", [])),
            "question_types": sorted(
                {
                    label
                    for occurrence in probe.get("occurrences", [])
                    for label in occurrence.get("question_types", [])
                }
            ),
            "strategy_tags": sorted(
                {
                    label
                    for occurrence in probe.get("occurrences", [])
                    for label in occurrence.get("strategy_tags", [])
                }
            ),
            "stage": primary_occurrence.get("stage"),
            "source_run": primary_occurrence.get("run_dir"),
            "question_turn_id": primary_occurrence.get("question_turn_id"),
            "author_solve_complete": solve is not None,
            "author_assessment": assess.get("assessment") if assess else None,
        }
        if reference:
            metrics = reference_metrics(reference)
            row.update(metrics)
            assessment = row["author_assessment"]
            row["self_score_error"] = (
                int(assessment["predicted_score"])
                - int(metrics["author_reference_score"])
                if assessment
                else None
            )
        rows.append(row)
    scored = [row for row in rows if "author_reference_score" in row]
    beyond = [row for row in scored if row["beyond_author"]]
    assessed = [
        row["author_assessment"]
        for row in rows
        if isinstance(row.get("author_assessment"), Mapping)
    ]
    intended_levels = Counter(
        str(assessment["intended_level"]) for assessment in assessed
    )
    self_solvability = Counter(
        str(assessment["self_solvability"]) for assessment in assessed
    )
    job_prompt_versions = Counter(
        str(row.get("prompt_version") or "unknown")
        for row in completed.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": prompt_version,
        "job_prompt_version_counts": dict(job_prompt_versions),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "name": config.name,
        "catalog_file": str(config.catalog_file),
        "reference_evaluator": {
            "id": config.reference_evaluator.id,
            "model": config.reference_evaluator.model,
        },
        "probe_count": len(rows),
        "scored_probe_count": len(scored),
        "beyond_author_count": len(beyond),
        "beyond_author_rate": len(beyond) / len(scored) if scored else None,
        "author_assessment_count": len(assessed),
        "intended_level_counts": dict(intended_levels),
        "self_solvability_counts": dict(self_solvability),
        "self_reported_unsolvable_count": self_solvability["not_solvable"],
        "valid_probe_rate": (
            sum(row["reference_validity"] == "valid" for row in scored)
            / len(scored)
            if scored
            else None
        ),
        "mean_probe_pair_accuracy": _mean_present(
            row.get("candidate_pair_accuracy") for row in scored
        ),
        "mean_self_score_error": _mean_present(
            row.get("self_score_error") for row in scored
        ),
        "mean_absolute_self_score_error": _mean_present(
            abs(row["self_score_error"])
            for row in scored
            if isinstance(row.get("self_score_error"), (int, float))
        ),
        "reported_cost_usd": reported_cost_usd,
        "missing_jobs": missing_jobs,
        "by_author": _group_summary(rows, "author_model"),
        "by_question_type": _multilabel_summary(rows, "question_types"),
        "probes": rows,
    }


def render_probe_self_study(summary: Mapping[str, Any]) -> str:
    author_rows = "".join(
        "<tr>"
        f"<th>{escape(row['label'])}</th>"
        f"<td>{row['probe_count']}</td>"
        f"<td>{_pct(row['intended_stronger_rate'])}</td>"
        f"<td>{_pct(row['self_reported_unsolvable_rate'])}</td>"
        f"<td>{_pct(row['valid_rate'])}</td>"
        f"<td>{_pct(row['beyond_author_rate'])}</td>"
        f"<td>{_pct(row['mean_pair_accuracy'])}</td>"
        f"<td>{_number(row['mean_absolute_self_score_error'])}</td>"
        "</tr>"
        for row in summary["by_author"]
    )
    type_rows = "".join(
        "<tr>"
        f"<th>{escape(row['label'])}</th>"
        f"<td>{row['probe_count']}</td>"
        f"<td>{_pct(row['valid_rate'])}</td>"
        f"<td>{_pct(row['beyond_author_rate'])}</td>"
        f"<td>{_pct(row['mean_pair_accuracy'])}</td>"
        "</tr>"
        for row in summary["by_question_type"]
    )
    audit_rows = "".join(
        "<tr>"
        f"<td>{escape(row['author_model'])}</td>"
        f"<td><a href='../../../{escape(str(row['source_run']))}/transcript.jsonl'>"
        f"{escape(row['question_excerpt'])}</a>"
        f"<span>turn {row['question_turn_id']}</span></td>"
        f"<td>{escape(', '.join(row['question_types']) or 'unclassified')}</td>"
        f"<td>{row.get('author_reference_score', '—')}</td>"
        f"<td>{row.get('best_candidate_score', '—')}</td>"
        f"<td>{'yes' if row.get('beyond_author') else 'no'}</td>"
        "</tr>"
        for row in sorted(
            summary["probes"],
            key=lambda item: (
                not item.get("beyond_author", False),
                item["author_model"],
                item["probe_id"],
            ),
        )[:30]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Probe self-study</title>
<style>
:root{{--ink:#18201d;--muted:#5e6863;--line:#cbd3cf;--paper:#f7f9f7;--accent:#0b6b53}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{width:min(1180px,calc(100% - 32px));margin:42px auto 72px}}
h1{{margin:0 0 8px;font:700 36px/1.1 Georgia,serif}} h2{{margin:34px 0 10px;
font:700 22px/1.2 Georgia,serif}} p{{max-width:780px;color:var(--muted)}}
.metrics{{display:flex;gap:28px;flex-wrap:wrap;margin:20px 0}} .metric strong{{
display:block;color:var(--accent);font-size:28px}} .metric span{{color:var(--muted)}}
.table-wrap{{overflow-x:auto;border-top:2px solid var(--ink)}} table{{width:100%;
border-collapse:collapse;background:white}} th,td{{padding:10px 12px;border-bottom:1px
solid var(--line);text-align:left;vertical-align:top}} thead th{{font-size:12px;
text-transform:uppercase;color:var(--muted)}} tbody th{{white-space:nowrap}}
.table-wrap td span{{display:block;color:var(--muted);font-size:12px;margin-top:3px}}
.note{{font-size:13px}}
</style></head><body><main>
<h1>Can judges solve their own tests?</h1>
<p>Each probe author solved its probe in a fresh context and separately predicted
its own performance. A fixed reference evaluator then scored that solution beside
the archived anonymous candidate answers.</p>
<div class="metrics">
<div class="metric"><strong>{summary['probe_count']}</strong><span>unique probes</span></div>
<div class="metric"><strong>{_pct(summary['valid_probe_rate'])}</strong><span>reference-valid</span></div>
<div class="metric"><strong>{summary['beyond_author_count']}</strong><span>beyond-author probes</span></div>
<div class="metric"><strong>{summary['self_reported_unsolvable_count']}</strong><span>authors say unsolvable</span></div>
<div class="metric"><strong>{_number(summary['mean_absolute_self_score_error'])}</strong><span>mean absolute self-score error</span></div>
<div class="metric"><strong>${summary['reported_cost_usd']:.2f}</strong><span>reported spend</span></div>
</div>
<p class="note">“Beyond author” requires a reference-valid probe, an author score
below substantially correct (0–2), and at least one archived answer from a
candidate with a higher external intelligence score receiving 3 or 4. This is
an operational label, not proof that the author lacks the underlying capability.</p>
<h2>By probe author</h2><div class="table-wrap"><table><thead><tr>
<th>Author model</th><th>Probes</th><th>Targets stronger</th><th>Says unsolvable</th>
<th>Valid</th><th>Beyond author</th>
<th>Candidate pair accuracy</th><th>Self-score error</th></tr></thead>
<tbody>{author_rows}</tbody></table></div>
<h2>By question type</h2><div class="table-wrap"><table><thead><tr>
<th>Question type</th><th>Probes</th><th>Valid</th><th>Beyond author</th>
<th>Candidate pair accuracy</th></tr></thead><tbody>{type_rows}</tbody></table></div>
<h2>Compact audit</h2><div class="table-wrap"><table><thead><tr>
<th>Author</th><th>Probe</th><th>Types</th><th>Author</th><th>Best candidate</th>
<th>Beyond author</th></tr></thead><tbody>{audit_rows}</tbody></table></div>
</main></body></html>"""


def reference_metrics(reference: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = reference["evaluation"]
    answer_map = reference["answer_map"]
    scored = []
    author_score = None
    candidate_scores = []
    for answer_id, score_row in evaluation["scores"].items():
        mapping = answer_map[answer_id]
        score = int(score_row["score"])
        row = {**mapping, "score": score}
        scored.append(row)
        if mapping["kind"] == "author_fresh":
            author_score = score
        else:
            candidate_scores.append(row)
    if author_score is None:
        raise ValueError("reference result has no author_fresh answer")
    pair_accuracy, pair_count = candidate_pair_accuracy(candidate_scores)
    best_candidate = max((row["score"] for row in candidate_scores), default=0)
    author_external_score = next(
        (
            item.get("candidate_score")
            for item in scored
            if item["kind"] == "author_fresh"
        ),
        None,
    )
    above_author = (
        [
            row
            for row in candidate_scores
            if isinstance(row.get("candidate_score"), (int, float))
            and float(row["candidate_score"]) > float(author_external_score)
        ]
        if isinstance(author_external_score, (int, float))
        else []
    )
    validity = evaluation["probe_validity"]
    return {
        "reference_validity": validity,
        "reference_validity_reason": evaluation["validity_reason"],
        "author_reference_score": author_score,
        "best_candidate_score": best_candidate,
        "candidate_score_range": (
            max(row["score"] for row in candidate_scores)
            - min(row["score"] for row in candidate_scores)
            if candidate_scores
            else None
        ),
        "candidate_substantially_correct_rate": (
            sum(row["score"] >= 3 for row in candidate_scores)
            / len(candidate_scores)
            if candidate_scores
            else None
        ),
        "candidate_pair_accuracy": pair_accuracy,
        "candidate_pair_count": pair_count,
        "above_author_candidate_count": len(above_author),
        "above_author_success_rate": (
            sum(row["score"] >= 3 for row in above_author) / len(above_author)
            if above_author
            else None
        ),
        "stronger_candidate_succeeds": any(
            row["score"] >= 3 for row in above_author
        ),
        "beyond_author": (
            validity == "valid"
            and author_score < 3
            and any(row["score"] >= 3 for row in above_author)
        ),
    }


def candidate_pair_accuracy(
    rows: list[Mapping[str, Any]],
) -> tuple[float | None, int]:
    correct = 0.0
    count = 0
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            left_prior = left.get("candidate_score")
            right_prior = right.get("candidate_score")
            if (
                not isinstance(left_prior, (int, float))
                or not isinstance(right_prior, (int, float))
                or left_prior == right_prior
            ):
                continue
            count += 1
            predicted = int(left["score"]) - int(right["score"])
            truth = float(left_prior) - float(right_prior)
            correct += 0.5 if predicted == 0 else float(predicted * truth > 0)
    return (correct / count if count else None, count)


def _group_summary(
    rows: list[Mapping[str, Any]], key: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return [
        _summarize_group(label, values)
        for label, values in sorted(grouped.items())
    ]


def _multilabel_summary(
    rows: list[Mapping[str, Any]], key: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        labels = row.get(key) or ["unclassified"]
        for label in labels:
            grouped.setdefault(str(label), []).append(row)
    return sorted(
        (
            _summarize_group(label, values)
            for label, values in grouped.items()
        ),
        key=lambda row: (-row["probe_count"], row["label"]),
    )


def _summarize_group(
    label: str, rows: list[Mapping[str, Any]]
) -> dict[str, Any]:
    assessed = [
        row["author_assessment"]
        for row in rows
        if isinstance(row.get("author_assessment"), Mapping)
    ]
    scored = [
        row
        for row in rows
        if isinstance(row.get("reference_validity"), str)
    ]
    return {
        "label": label,
        "probe_count": len(rows),
        "scored_probe_count": len(scored),
        "intended_stronger_rate": (
            mean(
                assessment.get("intended_level") == "stronger"
                for assessment in assessed
            )
            if assessed
            else None
        ),
        "self_reported_unsolvable_rate": (
            mean(
                assessment.get("self_solvability") == "not_solvable"
                for assessment in assessed
            )
            if assessed
            else None
        ),
        "valid_rate": (
            mean(
                row["reference_validity"] == "valid"
                for row in scored
            )
            if scored
            else None
        ),
        "beyond_author_rate": (
            mean(row["beyond_author"] for row in scored)
            if scored
            else None
        ),
        "mean_pair_accuracy": _mean_present(
            row.get("candidate_pair_accuracy") for row in scored
        ),
        "mean_score_range": _mean_present(
            row.get("candidate_score_range") for row in scored
        ),
        "mean_absolute_self_score_error": _mean_present(
            abs(row["self_score_error"])
            for row in scored
            if isinstance(row.get("self_score_error"), (int, float))
        ),
    }


def _job_key(stage: str, probe_id: str) -> str:
    return f"{stage}:{probe_id}"


def _mean_present(values: Any) -> float | None:
    present = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
    ]
    return mean(present) if present else None


def _excerpt(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _number(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2f}"
