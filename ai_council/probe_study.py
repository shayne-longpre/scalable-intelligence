from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from ai_council.metrics import compute_evolution_metrics
from ai_council.oversight_analysis import pair_observations
from ai_council.taxonomy import (
    DEFAULT_TAXONOMY_PATH,
    EVALUATION_STRATEGY,
    QUESTION_TYPE,
    load_taxonomy,
    taxonomy_hits_for_entry,
)


NON_TOPICAL_TYPES = {
    "instruction_following_format_control",
    "processing_speed_efficiency",
}
ADAPTIVE_INTENT_INDICATORS = {
    "raise_difficulty": (
        "increase difficulty",
        "harder",
        "highly challenging",
        "ceiling effect",
        "saturat",
    ),
    "change_domain": (
        "change domain",
        "novel domain",
        "different domain",
        "outside ",
        "transferability",
    ),
    "retest_weakness": (
        "retest",
        "re-test",
        "replicate",
        "re-verify",
        "reliability check",
        "error consistency",
        "persists",
        "generalizes",
    ),
    "cross_domain_integration": (
        "cross-domain",
        "multi-capability",
        "integrating ",
        "synthesis across",
        "combining ",
    ),
    "adversarial_validation": (
        "adversarial",
        "edge case",
        "edge-case",
        "counterexample",
        "hidden contradiction",
        "conflicting constraint",
        "ambiguous",
    ),
    "mechanical_scoring": (
        "mechanical checkability",
        "partial credit",
        "strict format",
        "formatted output",
        "proof of correctness",
        "unambiguous",
    ),
}


def build_probe_study(
    *,
    run_specs: Sequence[Mapping[str, str]],
    catalog_path: str | Path,
    output_dir: str | Path,
    published_json_path: str | Path | None = None,
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
) -> dict[str, Any]:
    if not run_specs:
        raise ValueError("probe study requires at least one run")
    catalog = _load_json(Path(catalog_path))
    catalog_rows = {
        row["provider_model_id"]: row for row in catalog.get("models", [])
    }
    taxonomy = load_taxonomy(taxonomy_path)
    records = [
        analyze_probe_run(
            Path(spec["run_dir"]),
            cohort=spec["cohort"],
            catalog_rows=catalog_rows,
            taxonomy=taxonomy,
        )
        for spec in run_specs
    ]
    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cohort[record["cohort"]].append(record)
    summary = {
        "schema_version": "probe-evolution-study-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_file": str(catalog_path),
        "taxonomy_file": str(taxonomy_path),
        "taxonomy_version": taxonomy.get("version"),
        "run_count": len(records),
        "probe_count": sum(len(record["probes"]) for record in records),
        "cohorts": {
            cohort: summarize_cohort(rows)
            for cohort, rows in sorted(by_cohort.items())
        },
        "runs": records,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report_card.html").write_text(
        render_probe_study(summary), encoding="utf-8"
    )
    if published_json_path:
        published_path = Path(published_json_path)
        published_path.parent.mkdir(parents=True, exist_ok=True)
        published_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def analyze_probe_run(
    run_dir: Path,
    *,
    cohort: str,
    catalog_rows: Mapping[str, Mapping[str, Any]],
    taxonomy: Mapping[str, Any],
) -> dict[str, Any]:
    config = _load_json(run_dir / "config.json")
    extraction = _load_json(run_dir / "posthoc_extraction.json")
    analysis = _load_json(run_dir / "analysis_summary.json")
    judge_ref = config["judges"][0]["model"]
    judge_model = config["models"][judge_ref]["model"]
    catalog_row = catalog_rows.get(judge_model, {})
    judge_score = catalog_row.get("intelligence_score")
    if not isinstance(judge_score, (int, float)):
        raise ValueError(f"catalog score missing for judge {judge_model}")

    prior = analysis["prior_agreement"]
    scores = {
        participant_id: float(score)
        for participant_id, score in prior["participant_prior_scores"].items()
    }
    scored_participants = set(scores)
    comparisons = {
        row["probe_id"]: row for row in extraction.get("probe_comparisons", [])
    }
    harmonized_events = [
        _harmonize_event_taxonomy(event, taxonomy)
        for event in extraction.get("probe_events", [])
    ]
    harmonized_extraction = {**extraction, "probe_events": harmonized_events}
    evolution = compute_evolution_metrics(harmonized_extraction)
    annotated = {
        row["probe_id"]: row for row in evolution.get("probe_events", [])
    }
    unavailable = _unavailable_answers(run_dir)
    probes = [
        _probe_record(
            event,
            comparisons.get(event["probe_id"], {}),
            annotated.get(event["probe_id"], {}),
            scores,
            scored_participants,
            unavailable.get(event["probe_id"], set()),
            float(judge_score),
        )
        for event in harmonized_events
    ]
    opening = [probe for probe in probes if probe["stage"] == "baseline_battery"]
    adaptive = [probe for probe in probes if probe["stage"] == "adaptive_followup"]
    opening_accuracy, opening_pairs = _pooled_probe_accuracy(opening)
    adaptive_accuracy, adaptive_pairs = _pooled_probe_accuracy(adaptive)
    opening_decided_accuracy, opening_decided_pairs = (
        _pooled_decided_probe_accuracy(opening)
    )
    adaptive_decided_accuracy, adaptive_decided_pairs = (
        _pooled_decided_probe_accuracy(adaptive)
    )
    opening_tie_rate = _pooled_tie_rate(opening)
    adaptive_tie_rate = _pooled_tie_rate(adaptive)
    judgments = sorted(
        prior.get("judgments", []),
        key=lambda row: (int(row.get("round_index") or 0), row.get("phase", "")),
    )
    opening_rank_accuracy = (
        _ranking_accuracy(judgments[0]["ranking"], scores, scored_participants)
        if judgments
        else None
    )
    final_rank_accuracy = (
        _ranking_accuracy(judgments[-1]["ranking"], scores, scored_participants)
        if judgments
        else None
    )
    opening_types = {
        tag
        for probe in opening
        for tag in probe["question_types"]
        if tag not in NON_TOPICAL_TYPES
    }
    validity = Counter(probe["validity"] for probe in probes if probe["validity"])
    transitions = Counter(
        probe["transition"] for probe in probes if probe["transition"]
    )
    adaptive_rounds = adaptive_decision_records(
        extraction.get("adaptive_decisions") or []
    )
    intent_labels = sorted(
        {
            intent
            for adaptive_round in adaptive_rounds
            for intent in adaptive_round["intents"]
        }
    )
    return {
        "cohort": cohort,
        "run_dir": str(run_dir),
        "run_name": config["name"],
        "judge_model": judge_model,
        "judge_name": catalog_row.get("display_name") or _short_name(judge_model),
        "judge_score": float(judge_score),
        "candidate_count": len(scores),
        "opening_probe_count": len(opening),
        "adaptive_probe_count": len(adaptive),
        "opening_type_breadth": len(opening_types),
        "opening_question_types": sorted(opening_types),
        "opening_probe_pair_accuracy": opening_accuracy,
        "opening_probe_pair_count": opening_pairs,
        "opening_decided_pair_accuracy": opening_decided_accuracy,
        "opening_decided_pair_count": opening_decided_pairs,
        "opening_tie_pair_rate": opening_tie_rate,
        "adaptive_probe_pair_accuracy": adaptive_accuracy,
        "adaptive_probe_pair_count": adaptive_pairs,
        "adaptive_decided_pair_accuracy": adaptive_decided_accuracy,
        "adaptive_decided_pair_count": adaptive_decided_pairs,
        "adaptive_tie_pair_rate": adaptive_tie_rate,
        "opening_ranking_accuracy": opening_rank_accuracy,
        "final_ranking_accuracy": final_rank_accuracy,
        "adaptive_ranking_delta": _difference(
            final_rank_accuracy, opening_rank_accuracy
        ),
        "informative_probe_rate": (
            validity["informative"] / sum(validity.values()) if validity else None
        ),
        "validity_counts": dict(validity),
        "transition_counts": dict(transitions),
        "adaptive_rounds": adaptive_rounds,
        "adaptive_plan_action_count": sum(
            len(adaptive_round["planned_strategy"])
            for adaptive_round in adaptive_rounds
        ),
        "adaptive_intent_breadth": len(intent_labels),
        "adaptive_intents": intent_labels,
        "adaptive_target_count": sum(
            adaptive_round["target_count"] for adaptive_round in adaptive_rounds
        ),
        "adaptive_uncertain_pairs_covered": sum(
            adaptive_round["uncertain_pairs_covered"]
            for adaptive_round in adaptive_rounds
        ),
        "question_type_counts": dict(
            Counter(tag for probe in probes for tag in probe["question_types"])
        ),
        "strategy_counts": dict(
            Counter(tag for probe in probes for tag in probe["strategy_tags"])
        ),
        "probes": probes,
    }


def summarize_cohort(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: float(row["judge_score"]))
    bands = assign_score_bands(ordered)
    band_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record, band in zip(ordered, bands, strict=True):
        band_counts[band].update(record["question_type_counts"])
    metrics = (
        "opening_type_breadth",
        "opening_probe_pair_accuracy",
        "opening_decided_pair_accuracy",
        "opening_tie_pair_rate",
        "informative_probe_rate",
        "adaptive_probe_pair_accuracy",
        "adaptive_decided_pair_accuracy",
        "adaptive_tie_pair_rate",
        "adaptive_ranking_delta",
        "adaptive_plan_action_count",
        "adaptive_intent_breadth",
    )
    correlations = {
        metric: {
            "spearman_rho": spearman(
                [float(row["judge_score"]) for row in ordered],
                [row.get(metric) for row in ordered],
            ),
            "n": sum(row.get(metric) is not None for row in ordered),
        }
        for metric in metrics
    }
    return {
        "run_count": len(ordered),
        "probe_count": sum(len(row["probes"]) for row in ordered),
        "judge_score_min": min(float(row["judge_score"]) for row in ordered),
        "judge_score_max": max(float(row["judge_score"]) for row in ordered),
        "question_type_counts": _sum_counts(
            row["question_type_counts"] for row in ordered
        ),
        "strategy_counts": _sum_counts(row["strategy_counts"] for row in ordered),
        "transition_counts": _sum_counts(
            row["transition_counts"] for row in ordered
        ),
        "validity_counts": _sum_counts(row["validity_counts"] for row in ordered),
        "adaptive_intent_counts": dict(
            Counter(intent for row in ordered for intent in row["adaptive_intents"])
        ),
        "question_types_by_score_band": {
            band: dict(counts) for band, counts in band_counts.items()
        },
        "correlations": correlations,
        "runs": [
            {
                "run_dir": row["run_dir"],
                "judge_model": row["judge_model"],
                "judge_name": row["judge_name"],
                "judge_score": row["judge_score"],
                "score_band": band,
                "opening_type_breadth": row["opening_type_breadth"],
                "opening_probe_pair_accuracy": row[
                    "opening_probe_pair_accuracy"
                ],
                "opening_decided_pair_accuracy": row[
                    "opening_decided_pair_accuracy"
                ],
                "opening_tie_pair_rate": row["opening_tie_pair_rate"],
                "informative_probe_rate": row["informative_probe_rate"],
                "adaptive_probe_pair_accuracy": row[
                    "adaptive_probe_pair_accuracy"
                ],
                "adaptive_decided_pair_accuracy": row[
                    "adaptive_decided_pair_accuracy"
                ],
                "adaptive_tie_pair_rate": row["adaptive_tie_pair_rate"],
                "adaptive_ranking_delta": row["adaptive_ranking_delta"],
                "adaptive_plan_action_count": row["adaptive_plan_action_count"],
                "adaptive_intent_breadth": row["adaptive_intent_breadth"],
            }
            for row, band in zip(ordered, bands, strict=True)
        ],
    }


def assign_score_bands(records: Sequence[Mapping[str, Any]]) -> list[str]:
    count = len(records)
    if count < 3:
        return ["all"] * count
    bands = []
    for index in range(count):
        if index * 3 < count:
            bands.append("lower third")
        elif index * 3 < count * 2:
            bands.append("middle third")
        else:
            bands.append("upper third")
    return bands


def adaptive_intents(items: Sequence[str]) -> list[str]:
    text = " ".join(items).lower()
    return [
        label
        for label, indicators in ADAPTIVE_INTENT_INDICATORS.items()
        if any(indicator in text for indicator in indicators)
    ]


def adaptive_decision_records(
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for decision in decisions:
        planned_strategy = [
            str(item)
            for item in decision.get("planned_strategy", [])
            if str(item).strip()
        ]
        rationale = [
            str(item)
            for item in decision.get("follow_up_rationale", [])
            if str(item).strip()
        ]
        records.append(
            {
                "round_index": int(decision.get("round_index") or 0),
                "probe_sequence_number": int(
                    decision.get("probe_sequence_number") or 0
                ),
                "target_count": len(decision.get("actual_candidates", [])),
                "uncertain_pairs_covered": len(
                    decision.get("covered_uncertain_pairs", [])
                ),
                "planned_strategy": planned_strategy,
                "rationale": rationale,
                "intents": adaptive_intents([*rationale, *planned_strategy]),
            }
        )
    return records


def spearman(xs: Sequence[float], ys: Sequence[float | None]) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys, strict=True)
        if isinstance(y, (int, float)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return None
    x_ranks = _average_ranks([pair[0] for pair in pairs])
    y_ranks = _average_ranks([pair[1] for pair in pairs])
    x_mean, y_mean = mean(x_ranks), mean(y_ranks)
    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_ranks, y_ranks, strict=True)
    )
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in x_ranks)
        * sum((y - y_mean) ** 2 for y in y_ranks)
    )
    return numerator / denominator if denominator else None


def render_probe_study(summary: Mapping[str, Any]) -> str:
    cohort_sections = "\n".join(
        _cohort_section(name, cohort, summary["runs"])
        for name, cohort in summary["cohorts"].items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>How AI judges design intelligence tests</title>
<style>
:root{{--ink:#182026;--muted:#65727a;--line:#dce2e5;--paper:#fff;--soft:#f4f6f7}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif}} main{{max-width:1180px;
margin:auto;padding:48px 28px 72px}} h1{{font:700 46px/1.05 Georgia,serif;
max-width:850px;margin:0 0 14px}} h2{{font:700 28px/1.2 Georgia,serif;margin:52px 0 8px}}
h3{{font-size:17px;margin:30px 0 8px}} .lede{{font-size:19px;max-width:850px;color:#39464d}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);
border:1px solid var(--line);margin:28px 0}} .metric{{background:white;padding:18px}}
.metric strong{{display:block;font-size:27px}} .metric span,.note,figcaption{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:start}}
figure{{margin:22px 0}} svg{{width:100%;height:auto;background:white}}
.table-wrap{{overflow:auto;border-top:2px solid var(--ink)}} table{{width:100%;
border-collapse:collapse;min-width:760px}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);
text-align:left;vertical-align:top}} th{{font-size:12px;color:var(--muted)}}
.num{{text-align:right;font-variant-numeric:tabular-nums}} code{{font-size:12px}}
.probe{{max-width:440px}} .stage{{font-size:11px;text-transform:uppercase;color:var(--muted)}}
@media(max-width:760px){{main{{padding:30px 16px 50px}}h1{{font-size:35px}}
.grid,.metrics{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>How AI judges design intelligence tests</h1>
<p class="lede">This report follows self-authored probes from broad opening
batteries into evidence-conditioned follow-ups. "Diagnosticity" means agreement
with the external model ordering; it is not treated as a universal measure of
question difficulty.</p>
<section class="metrics">
<div class="metric"><strong>{summary['run_count']}</strong><span>judge runs</span></div>
<div class="metric"><strong>{summary['probe_count']}</strong><span>authored probes</span></div>
<div class="metric"><strong>{len(summary['cohorts'])}</strong><span>protocol cohorts</span></div>
</section>
{cohort_sections}
</main></body></html>"""


def _cohort_section(
    name: str,
    cohort: Mapping[str, Any],
    all_runs: Sequence[Mapping[str, Any]],
) -> str:
    runs = [row for row in all_runs if row["cohort"] == name]
    rows = "\n".join(
        f"<tr><td>{escape(row['judge_name'])}</td>"
        f"<td class='num'>{row['judge_score']:.1f}</td>"
        f"<td class='num'>{row['opening_type_breadth']}</td>"
        f"<td class='num'>{_pct(row['opening_probe_pair_accuracy'])}</td>"
        f"<td class='num'>{_pct(row['opening_decided_pair_accuracy'])}</td>"
        f"<td class='num'>{_pct(row['opening_tie_pair_rate'])}</td>"
        f"<td class='num'>{_pct(row['informative_probe_rate'])}</td>"
        f"<td class='num'>{_signed_pct(row['adaptive_ranking_delta'])}</td></tr>"
        for row in sorted(runs, key=lambda item: -item["judge_score"])
    )
    probe_rows = "\n".join(
        f"<tr><td>{escape(row['judge_name'])}</td>"
        f"<td>{probe['sequence']}</td><td><span class='stage'>"
        f"{escape(probe['stage_label'])}</span>"
        f"<div class='probe'>{escape(probe['title'])}</div></td>"
        f"<td>{escape(_labels(probe['question_types'], 'unclassified'))}</td>"
        f"<td>{escape(_label(probe['transition'] or 'unknown'))}</td>"
        f"<td class='num'>{_pct(probe['pair_accuracy'])}</td>"
        f"<td>{escape(probe['validity'] or 'unclassified')}</td></tr>"
        for row in sorted(runs, key=lambda item: -item["judge_score"])
        for probe in row["probes"]
    )
    return f"""<section>
<h2>{escape(_label(name))}</h2>
<p class="note">{cohort['run_count']} judge runs, {cohort['probe_count']} probes,
external-score range {cohort['judge_score_min']:.1f}-{cohort['judge_score_max']:.1f}.
Score bands below are sample-relative thirds.</p>
<div class="grid">
<figure>{_scatter_svg(runs)}
<figcaption>Opening-battery diagnosticity by judge intelligence. Correlations
are descriptive because panels are small and the external index is noisy.</figcaption></figure>
<figure>{_heatmap_svg(runs, cohort['question_type_counts'])}
<figcaption>Question-type counts for each judge. Columns show the most common
labels in this cohort; one probe may receive multiple labels.</figcaption></figure>
</div>
<figure>{_evolution_svg(runs)}
<figcaption>How each sequence moved from preplanned opening probes to
evidence-conditioned follow-up. "Deepens" retains a topical family; "broadens"
switches or adds families.</figcaption></figure>
<h3>Association with judge intelligence</h3>
{_correlation_table(cohort['correlations'])}
<h3>Judge summary</h3>
<div class="table-wrap"><table><thead><tr><th>Judge</th><th class="num">Score</th>
<th class="num">Opening breadth</th><th class="num">Tie-aware diagnosticity</th>
<th class="num">Decided-pair accuracy</th><th class="num">Opening ties</th>
<th class="num">Informative</th>
<th class="num">Adaptive ranking change</th>
</tr></thead><tbody>{rows}</tbody></table></div>
<h3>Adaptive plans</h3>
{_adaptive_plan_table(runs)}
<h3>Probe sequence</h3>
<div class="table-wrap"><table><thead><tr><th>Judge</th><th>#</th><th>Probe</th>
<th>Question types</th><th>Transition</th><th class="num">Diagnosticity</th>
<th>Judge validity</th></tr></thead><tbody>{probe_rows}</tbody></table></div>
</section>"""


def _probe_record(
    event: Mapping[str, Any],
    comparison: Mapping[str, Any],
    annotated: Mapping[str, Any],
    scores: Mapping[str, float],
    scored_participants: set[str],
    unavailable: set[str],
    judge_score: float,
) -> dict[str, Any]:
    parsed = comparison.get("parsed") or {}
    ordering = [
        participant_id
        for participant_id in parsed.get("ordering", [])
        if participant_id in scored_participants and participant_id not in unavailable
    ]
    ordering_score = score_tied_ordering(
        ordering,
        parsed.get("ties", []),
        scores,
        judge_score,
    )
    content = str(event.get("content") or "").strip()
    return {
        "probe_id": event["probe_id"],
        "sequence": int(event.get("probe_sequence_number") or 0),
        "round_index": event.get("round_index"),
        "stage": event.get("generation_stage"),
        "stage_label": (
            "opening" if event.get("generation_stage") == "baseline_battery"
            else "adaptive"
        ),
        "title": _probe_title(content),
        "question_types": [
            tag["tag"] for tag in event.get("question_type_tags", [])
        ],
        "strategy_tags": [
            tag["tag"] for tag in event.get("strategy_tags", [])
        ],
        "transition": annotated.get("transition_label"),
        "topical_transition": annotated.get("topical_transition"),
        "target_count": len(event.get("respondents", [])),
        "available_count": sum(
            participant_id not in unavailable
            for participant_id in event.get("respondents", [])
        ),
        "scored_order_count": len(ordering),
        "unavailable_count": len(unavailable),
        "pair_correct": ordering_score["pair_correct"],
        "pair_count": ordering_score["pair_count"],
        "pair_accuracy": ordering_score["pair_accuracy"],
        "decided_pair_correct": ordering_score["decided_pair_correct"],
        "decided_pair_count": ordering_score["decided_pair_count"],
        "decided_pair_accuracy": ordering_score["decided_pair_accuracy"],
        "tie_pair_count": ordering_score["tie_pair_count"],
        "tie_pair_rate": ordering_score["tie_pair_rate"],
        "validity": parsed.get("probe_validity"),
    }


def score_tied_ordering(
    ordering: Sequence[str],
    tie_groups: Sequence[Any],
    scores: Mapping[str, float],
    judge_score: float,
) -> dict[str, float | int | None]:
    observations = pair_observations(ordering, scores, judge_score)
    included = set(ordering)
    tied_pairs = {
        frozenset(pair)
        for group in tie_groups
        if isinstance(group, list)
        for pair in combinations(
            [participant_id for participant_id in group if participant_id in included],
            2,
        )
    }
    observed_pairs = {
        frozenset((row["left"], row["right"])) for row in observations
    }
    tied_pairs &= observed_pairs
    pair_correct = sum(
        0.5
        if frozenset((row["left"], row["right"])) in tied_pairs
        else float(bool(row["correct"]))
        for row in observations
    )
    decided = [
        row
        for row in observations
        if frozenset((row["left"], row["right"])) not in tied_pairs
    ]
    decided_correct = sum(bool(row["correct"]) for row in decided)
    pair_count = len(observations)
    return {
        "pair_correct": pair_correct,
        "pair_count": pair_count,
        "pair_accuracy": pair_correct / pair_count if pair_count else None,
        "decided_pair_correct": decided_correct,
        "decided_pair_count": len(decided),
        "decided_pair_accuracy": (
            decided_correct / len(decided) if decided else None
        ),
        "tie_pair_count": len(tied_pairs),
        "tie_pair_rate": len(tied_pairs) / pair_count if pair_count else None,
    }


def _harmonize_event_taxonomy(
    event: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> dict[str, Any]:
    hits = taxonomy_hits_for_entry({"content": event.get("content", "")}, taxonomy)
    return {
        **event,
        "question_type_tags": [
            {
                "tag": hit["tag"],
                "label": hit["label"],
                "matched_indicators": hit["matched_indicators"],
            }
            for hit in hits
            if hit["dimension"] == QUESTION_TYPE
        ],
        "strategy_tags": [
            {
                "tag": hit["tag"],
                "label": hit["label"],
                "matched_indicators": hit["matched_indicators"],
            }
            for hit in hits
            if hit["dimension"] == EVALUATION_STRATEGY
        ],
    }


def _unavailable_answers(run_dir: Path) -> dict[str, set[str]]:
    unavailable: dict[str, set[str]] = defaultdict(set)
    transcript_path = run_dir / "transcript.jsonl"
    for row in _load_jsonl(transcript_path):
        metadata = row.get("metadata", {})
        if metadata.get("answer_unavailable"):
            unavailable[str(metadata.get("probe_id"))].add(str(row.get("speaker")))
    return unavailable


def _ranking_accuracy(
    ranking: Sequence[str],
    scores: Mapping[str, float],
    scored_participants: set[str],
) -> float | None:
    filtered = [
        participant_id
        for participant_id in ranking
        if participant_id in scored_participants
    ]
    observations = pair_observations(filtered, scores, judge_score=0.0)
    return (
        sum(bool(row["correct"]) for row in observations) / len(observations)
        if observations
        else None
    )


def _pooled_probe_accuracy(
    probes: Sequence[Mapping[str, Any]],
) -> tuple[float | None, int]:
    correct = sum(float(probe["pair_correct"]) for probe in probes)
    pairs = sum(int(probe["pair_count"]) for probe in probes)
    return (correct / pairs if pairs else None), pairs


def _pooled_decided_probe_accuracy(
    probes: Sequence[Mapping[str, Any]],
) -> tuple[float | None, int]:
    correct = sum(float(probe["decided_pair_correct"]) for probe in probes)
    pairs = sum(int(probe["decided_pair_count"]) for probe in probes)
    return (correct / pairs if pairs else None), pairs


def _pooled_tie_rate(probes: Sequence[Mapping[str, Any]]) -> float | None:
    ties = sum(int(probe["tie_pair_count"]) for probe in probes)
    pairs = sum(int(probe["pair_count"]) for probe in probes)
    return ties / pairs if pairs else None


def _average_ranks(values: Sequence[float]) -> list[float]:
    ranked = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][1] == ranked[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for original_index, _ in ranked[index:end]:
            output[original_index] = average_rank
        index = end
    return output


def _sum_counts(rows: Sequence[Mapping[str, int]] | Any) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        total.update(row)
    return dict(total)


def _scatter_svg(runs: Sequence[Mapping[str, Any]]) -> str:
    width, height = 580, 330
    left, top, plot_width, plot_height = 52, 28, 490, 238
    scores = [float(row["judge_score"]) for row in runs]
    low, high = min(scores) - 2, max(scores) + 2

    def x(value: float) -> float:
        return left + (value - low) / (high - low) * plot_width

    def y(value: float) -> float:
        return top + (1 - value) * plot_height

    parts = []
    for tick in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        parts.append(
            f"<line x1='{left}' y1='{y(tick):.1f}' x2='{left+plot_width}' "
            f"y2='{y(tick):.1f}' stroke='#dce2e5'/>"
            f"<text x='{left-7}' y='{y(tick)+4:.1f}' text-anchor='end' "
            f"font-size='10' fill='#65727a'>{tick:.0%}</text>"
        )
    for row in runs:
        accuracy = row.get("opening_probe_pair_accuracy")
        if accuracy is None:
            continue
        cx, cy = x(float(row["judge_score"])), y(float(accuracy))
        parts.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='5' fill='#2364aa'>"
            f"<title>{escape(row['judge_name'])}: {accuracy:.1%}</title></circle>"
            f"<text x='{cx:.1f}' y='{cy-9:.1f}' text-anchor='middle' "
            f"font-size='9' fill='#39464d'>{escape(_short_label(row['judge_name']))}</text>"
        )
    parts.append(
        f"<text x='{left+plot_width/2:.1f}' y='{height-15}' text-anchor='middle' "
        f"font-size='11' fill='#65727a'>Judge intelligence score</text>"
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='Probe diagnosticity by judge intelligence'>{''.join(parts)}</svg>"
    )


def _evolution_svg(runs: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(runs, key=lambda row: -float(row["judge_score"]))
    labels = {
        "opening_probe": ("opens", "#65727a"),
        "preplanned_broadening": ("broadens", "#2a7f62"),
        "preplanned_same_area": ("same area", "#8f5d2e"),
        "preplanned_progression": ("progresses", "#7d8790"),
        "adaptive_broadening": ("broadens", "#2364aa"),
        "adaptive_deepening": ("deepens", "#c44536"),
        "adaptive_followup": ("follows up", "#7656a8"),
        "adaptive_provenance_missing": ("unknown", "#a4adb2"),
    }
    cell_width, row_height = 92, 35
    left, top = 145, 34
    max_probes = max((len(row["probes"]) for row in ordered), default=1)
    width = left + max_probes * cell_width + 14
    height = top + len(ordered) * row_height + 46
    parts = []
    for index in range(max_probes):
        parts.append(
            f"<text x='{left+index*cell_width+cell_width/2:.1f}' y='20' "
            f"text-anchor='middle' font-size='10' fill='#65727a'>"
            f"probe {index+1}</text>"
        )
    for row_index, row in enumerate(ordered):
        y = top + row_index * row_height
        parts.append(
            f"<text x='{left-8}' y='{y+21}' text-anchor='end' font-size='10' "
            f"fill='#39464d'>{escape(_short_label(row['judge_name']))}</text>"
        )
        for column, probe in enumerate(row["probes"]):
            transition = probe.get("transition") or "adaptive_provenance_missing"
            label, color = labels.get(transition, (_label(transition), "#a4adb2"))
            x = left + column * cell_width
            parts.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{cell_width-4}' "
                f"height='{row_height-5}' rx='2' fill='{color}' opacity='0.88'>"
                f"<title>{escape(row['judge_name'])}, probe {probe['sequence']}: "
                f"{escape(label)}</title></rect>"
                f"<text x='{x+(cell_width-4)/2:.1f}' y='{y+19}' text-anchor='middle' "
                f"font-size='9' fill='white'>{escape(label)}</text>"
            )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='Probe evolution by judge'>{''.join(parts)}</svg>"
    )


def _correlation_table(correlations: Mapping[str, Mapping[str, Any]]) -> str:
    labels = {
        "opening_type_breadth": "Opening topical breadth",
        "opening_probe_pair_accuracy": "Opening diagnosticity",
        "opening_decided_pair_accuracy": "Opening decided-pair accuracy",
        "opening_tie_pair_rate": "Opening tie-pair rate",
        "informative_probe_rate": "Judge-reported informative rate",
        "adaptive_probe_pair_accuracy": "Adaptive diagnosticity",
        "adaptive_decided_pair_accuracy": "Adaptive decided-pair accuracy",
        "adaptive_tie_pair_rate": "Adaptive tie-pair rate",
        "adaptive_ranking_delta": "Ranking accuracy change after adaptation",
        "adaptive_plan_action_count": "Adaptive plan actions",
        "adaptive_intent_breadth": "Adaptive plan breadth",
    }
    rows = "".join(
        f"<tr><td>{escape(labels.get(metric, _label(metric)))}</td>"
        f"<td class='num'>{_number(result.get('spearman_rho'))}</td>"
        f"<td class='num'>{int(result.get('n') or 0)}</td></tr>"
        for metric, result in correlations.items()
    )
    return (
        "<div class='table-wrap'><table><thead><tr><th>Probe property</th>"
        "<th class='num'>Spearman rho</th><th class='num'>Judges</th></tr>"
        f"</thead><tbody>{rows}</tbody></table></div>"
        "<p class='note'>These are descriptive run-level associations, not causal "
        "estimates or significance tests.</p>"
    )


def _adaptive_plan_table(runs: Sequence[Mapping[str, Any]]) -> str:
    rows = "".join(
        f"<tr><td>{escape(row['judge_name'])}</td>"
        f"<td class='num'>{adaptive_round['round_index']}</td>"
        f"<td>{escape(_labels(adaptive_round['intents'], 'none detected'))}</td>"
        f"<td class='num'>{len(adaptive_round['planned_strategy'])}</td>"
        f"<td class='num'>{adaptive_round['target_count']}</td>"
        f"<td class='num'>{adaptive_round['uncertain_pairs_covered']}</td>"
        f"<td class='probe'>{escape(' '.join(adaptive_round['rationale']))}</td></tr>"
        for row in sorted(runs, key=lambda item: -float(item["judge_score"]))
        for adaptive_round in row["adaptive_rounds"]
    )
    return (
        "<div class='table-wrap'><table><thead><tr><th>Judge</th>"
        "<th class='num'>Round</th><th>Plan signals</th>"
        "<th class='num'>Actions</th><th class='num'>Targets</th>"
        "<th class='num'>Uncertain pairs covered</th><th>Why these candidates</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _heatmap_svg(
    runs: Sequence[Mapping[str, Any]],
    total_counts: Mapping[str, int],
) -> str:
    types = [
        tag for tag, _ in sorted(
            total_counts.items(), key=lambda item: (-item[1], item[0])
        )[:10]
    ]
    ordered = sorted(runs, key=lambda row: -float(row["judge_score"]))
    cell, left, top = 31, 126, 115
    width = left + max(1, len(types)) * cell + 14
    height = top + max(1, len(ordered)) * cell + 14
    max_count = max(
        (
            int(row["question_type_counts"].get(tag, 0))
            for row in ordered
            for tag in types
        ),
        default=1,
    )
    parts = []
    for column, tag in enumerate(types):
        x = left + column * cell + cell / 2
        parts.append(
            f"<text x='{x:.1f}' y='{top-8}' transform='rotate(-55 {x:.1f} {top-8})' "
            f"text-anchor='start' font-size='9' fill='#65727a'>{escape(_label(tag))}</text>"
        )
    for row_index, row in enumerate(ordered):
        y = top + row_index * cell
        parts.append(
            f"<text x='{left-7}' y='{y+20}' text-anchor='end' font-size='10' "
            f"fill='#39464d'>{escape(_short_label(row['judge_name']))}</text>"
        )
        for column, tag in enumerate(types):
            count = int(row["question_type_counts"].get(tag, 0))
            opacity = 0.08 + 0.82 * count / max_count if count else 0.03
            parts.append(
                f"<rect x='{left+column*cell:.1f}' y='{y:.1f}' width='{cell-2}' "
                f"height='{cell-2}' fill='#2a7f62' opacity='{opacity:.2f}'>"
                f"<title>{escape(row['judge_name'])}, {escape(_label(tag))}: {count}</title>"
                f"</rect>"
                + (
                    f"<text x='{left+column*cell+(cell-2)/2:.1f}' y='{y+19}' "
                    f"text-anchor='middle' font-size='10' fill='#182026'>{count}</text>"
                    if count
                    else ""
                )
            )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='Question types by judge'>{''.join(parts)}</svg>"
    )


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def _probe_title(content: str, limit: int = 150) -> str:
    first = " ".join(content.split())
    if "**Answer limit:" in first:
        first = first.split("**Answer limit:", 1)[0].strip()
    return first if len(first) <= limit else first[: limit - 3].rstrip() + "..."


def _short_name(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].replace("-", " ").title()


def _short_label(value: str, limit: int = 17) -> str:
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


def _label(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _labels(values: Sequence[str], empty: str) -> str:
    return ", ".join(_label(value) for value in values) or empty


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _signed_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1%}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
