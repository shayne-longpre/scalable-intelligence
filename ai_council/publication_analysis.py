from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from html import escape
from itertools import combinations
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "publication-analysis-v1"
CAPABILITY_BAND_LABELS = (
    "Higher-capability authors",
    "Middle authors",
    "Lower-capability authors",
)
PROBE_TITLES = (
    ("Sol", "Pooled tests"),
    ("Sol", "Mechanics"),
    ("Sol", "Concurrency"),
    ("Sol", "Causal ID"),
    ("Fable", "Reachability"),
    ("Fable", "Language"),
    ("Fable", "Causal choice"),
    ("Fable", "Proof audit"),
)


def build_publication_analysis(
    *,
    self_study: Mapping[str, Any],
    probe_catalog: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
    probe_scores: Mapping[str, Any],
    crossed_report: Mapping[str, Any],
    research_summary: Mapping[str, Any],
    catalog_stability: Mapping[str, Any],
    oversight_synthesis: Mapping[str, Any],
    oversight_order_replay: Mapping[str, Any],
    verifier_council: Mapping[str, Any],
    catalog_order_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the publication-facing analyses from frozen experiment outputs."""
    probe_analysis = analyze_probe_authors(
        self_study=self_study,
        probe_catalog=probe_catalog,
        taxonomy=taxonomy,
    )
    answer_matrix = build_answer_matrix(
        probe_scores=probe_scores,
        crossed_report=crossed_report,
    )
    robustness = build_robustness_summary(
        research_summary=research_summary,
        catalog_stability=catalog_stability,
        oversight_synthesis=oversight_synthesis,
        crossed_report=crossed_report,
        oversight_order_replay=oversight_order_replay,
        verifier_council=verifier_council,
        catalog_order_analysis=catalog_order_analysis,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "probe_author_analysis": probe_analysis,
        "answer_matrix": answer_matrix,
        "robustness": robustness,
    }


def analyze_probe_authors(
    *,
    self_study: Mapping[str, Any],
    probe_catalog: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> dict[str, Any]:
    rows = list(self_study["probes"])
    full_questions = {
        row["probe_id"]: row["question"]
        for row in probe_catalog["probes"]
    }
    author_scores = {
        row["author_model"]: float(row["author_score"])
        for row in rows
    }
    authors = sorted(author_scores, key=lambda item: (-author_scores[item], item))
    band_authors = balanced_capability_bands(authors)
    type_labels = {
        row["id"]: row["label"] for row in taxonomy["question_types"]
    }
    strategy_labels = {
        row["id"]: row["label"] for row in taxonomy["tags"]
    }

    bands = []
    for label, members in zip(
        CAPABILITY_BAND_LABELS, band_authors, strict=True
    ):
        bands.append(
            summarize_author_band(
                label=label,
                authors=members,
                author_scores=author_scores,
                rows=rows,
                full_questions=full_questions,
            )
        )

    question_types = label_contrasts(
        rows=rows,
        band_authors=band_authors,
        label_key="question_types",
        display_labels=type_labels,
        min_occurrences=3,
    )
    strategies = label_contrasts(
        rows=rows,
        band_authors=band_authors,
        label_key="strategy_tags",
        display_labels=strategy_labels,
        min_occurrences=3,
    )
    author_solvability = [
        summarize_author(
            author=author,
            author_score=author_scores[author],
            rows=[row for row in rows if row["author_model"] == author],
        )
        for author in authors
    ]
    return {
        "probe_count": len(rows),
        "author_count": len(authors),
        "band_method": (
            "Authors are ordered by the external intelligence index and split "
            "as evenly as possible into higher, middle, and lower bands. "
            "Band rates average each author's within-author rate, giving every "
            "author equal weight."
        ),
        "bands": bands,
        "question_type_contrasts": question_types,
        "strategy_contrasts": strategies,
        "author_solvability": author_solvability,
    }


def balanced_capability_bands(
    ordered_authors: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    if len(ordered_authors) < 6:
        raise ValueError("capability comparison requires at least six authors")
    base, remainder = divmod(len(ordered_authors), 3)
    if remainder == 0:
        sizes = (base, base, base)
    elif remainder == 1:
        sizes = (base, base + 1, base)
    else:
        sizes = (base + 1, base, base + 1)
    middle_start = sizes[0]
    middle_end = middle_start + sizes[1]
    return (
        list(ordered_authors[:middle_start]),
        list(ordered_authors[middle_start:middle_end]),
        list(ordered_authors[middle_end:]),
    )


def summarize_author_band(
    *,
    label: str,
    authors: Sequence[str],
    author_scores: Mapping[str, float],
    rows: Sequence[Mapping[str, Any]],
    full_questions: Mapping[str, str],
) -> dict[str, Any]:
    author_rows = {
        author: [row for row in rows if row["author_model"] == author]
        for author in authors
    }
    metric_names = (
        "reference_valid_rate",
        "candidate_pair_accuracy",
        "objective_checkability_rate",
        "fully_self_solvable_rate",
        "multi_type_rate",
        "targets_stronger_rate",
    )
    per_author = {
        author: author_metrics(values) for author, values in author_rows.items()
    }
    return {
        "label": label,
        "author_count": len(authors),
        "probe_count": sum(len(values) for values in author_rows.values()),
        "score_min": min(author_scores[author] for author in authors),
        "score_max": max(author_scores[author] for author in authors),
        "authors": [
            {
                "model": author,
                "score": author_scores[author],
                "probe_count": len(author_rows[author]),
            }
            for author in authors
        ],
        "metrics": {
            name: mean(per_author[author][name] for author in authors)
            for name in metric_names
        },
        "mean_question_word_count": mean(
            mean(
                len(full_questions[row["probe_id"]].split())
                for row in author_rows[author]
            )
            for author in authors
        ),
    }


def author_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scored = [
        row
        for row in rows
        if isinstance(row.get("candidate_pair_accuracy"), (int, float))
    ]
    if not rows or not scored:
        raise ValueError("author metrics require scored probes")
    return {
        "reference_valid_rate": mean(
            row.get("reference_validity") == "valid" for row in scored
        ),
        "candidate_pair_accuracy": mean(
            float(row["candidate_pair_accuracy"]) for row in scored
        ),
        "objective_checkability_rate": mean(
            row["author_assessment"]["checkability"] == "objective"
            for row in rows
        ),
        "fully_self_solvable_rate": mean(
            row["author_assessment"]["self_solvability"] == "fully"
            for row in rows
        ),
        "multi_type_rate": mean(
            len(row.get("question_types", [])) > 1 for row in rows
        ),
        "targets_stronger_rate": mean(
            row["author_assessment"]["intended_level"] == "stronger"
            for row in rows
        ),
    }


def label_contrasts(
    *,
    rows: Sequence[Mapping[str, Any]],
    band_authors: Sequence[Sequence[str]],
    label_key: str,
    display_labels: Mapping[str, str],
    min_occurrences: int,
) -> list[dict[str, Any]]:
    counts = Counter(
        label for row in rows for label in row.get(label_key, [])
    )
    author_counts = {
        label: len(
            {
                row["author_model"]
                for row in rows
                if label in row.get(label_key, [])
            }
        )
        for label in counts
    }
    output = []
    for label, count in counts.items():
        if count < min_occurrences:
            continue
        rates = [
            mean(
                _author_label_rate(rows, author, label_key, label)
                for author in authors
            )
            for authors in band_authors
        ]
        output.append(
            {
                "id": label,
                "label": display_labels.get(label, _display_label(label)),
                "probe_count": count,
                "author_count": author_counts[label],
                "higher_rate": rates[0],
                "middle_rate": rates[1],
                "lower_rate": rates[2],
                "higher_minus_lower": rates[0] - rates[2],
            }
        )
    return sorted(
        output,
        key=lambda row: (-row["higher_minus_lower"], row["label"]),
    )


def summarize_author(
    *,
    author: str,
    author_score: float,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    solvability = Counter(
        row["author_assessment"]["self_solvability"] for row in rows
    )
    scored = [
        row
        for row in rows
        if isinstance(row.get("candidate_pair_accuracy"), (int, float))
    ]
    return {
        "model": author,
        "score": author_score,
        "probe_count": len(rows),
        "fully_rate": solvability["fully"] / len(rows),
        "partially_rate": solvability["partially"] / len(rows),
        "not_solvable_rate": solvability["not_solvable"] / len(rows),
        "beyond_author_rate": (
            mean(bool(row.get("beyond_author")) for row in scored)
            if scored
            else None
        ),
    }


def build_answer_matrix(
    *,
    probe_scores: Mapping[str, Any],
    crossed_report: Mapping[str, Any],
) -> dict[str, Any]:
    runs = {
        run["name"]: run for run in crossed_report["runs"]
    }
    source = runs["catalog_ladder50_gpt_5_6_sol"]
    participants = {
        row["id"]: row["provider_model_id"] for row in source["participants"]
    }
    external_scores = {
        participant_id: float(score)
        for participant_id, score in source["prior_participant_scores"].items()
    }
    directly_reported = set(source["prior_reported_score_participants"])
    probes = list(probe_scores["probes"])
    if len(probes) != len(PROBE_TITLES):
        raise ValueError(
            f"expected {len(PROBE_TITLES)} scored probes, found {len(probes)}"
        )

    columns = [
        {
            "probe_id": probe["probe_id"],
            "author": author,
            "title": title,
            "mean_answer_score": float(probe["mean_answer_score"]),
            "judge_count": int(probe["judge_count"]),
        }
        for probe, (author, title) in zip(
            probes, PROBE_TITLES, strict=True
        )
    ]
    rows = []
    aggregate_scores: dict[str, float] = {}
    for participant_id in sorted(
        external_scores,
        key=lambda item: (-external_scores[item], item),
    ):
        values = [
            (
                float(probe["mean_scores"][participant_id])
                if participant_id in probe["mean_scores"]
                else None
            )
            for probe in probes
        ]
        available = [value for value in values if value is not None]
        if available:
            aggregate_scores[participant_id] = mean(available)
        rows.append(
            {
                "participant_id": participant_id,
                "model": participants[participant_id],
                "external_score": external_scores[participant_id],
                "external_score_is_estimated": (
                    participant_id not in directly_reported
                ),
                "scores": values,
                "mean_score": mean(available) if available else None,
            }
        )
    reported_aggregate = {
        key: value
        for key, value in aggregate_scores.items()
        if key in directly_reported
    }
    reported_scores = {
        key: external_scores[key] for key in reported_aggregate
    }
    return {
        "columns": columns,
        "rows": rows,
        "stats": {
            "probe_count": len(columns),
            "candidate_count": len(rows),
            "reported_candidate_count": len(reported_aggregate),
            "score_spearman": spearman(
                list(reported_aggregate.values()),
                [reported_scores[key] for key in reported_aggregate],
            ),
            "pairwise_accuracy": score_pairwise_accuracy(
                reported_aggregate,
                reported_scores,
            ),
            "jointly_scored_probe_count": sum(
                column["judge_count"] > 1 for column in columns
            ),
        },
    }


def build_robustness_summary(
    *,
    research_summary: Mapping[str, Any],
    catalog_stability: Mapping[str, Any],
    oversight_synthesis: Mapping[str, Any],
    crossed_report: Mapping[str, Any],
    oversight_order_replay: Mapping[str, Any],
    verifier_council: Mapping[str, Any],
    catalog_order_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    rq1 = research_summary["rq1_catalog_ranking"]
    rq2 = research_summary["rq2_oversight_frontier"]
    anchor_rows = []
    for judge in rq1["judges"]:
        all_scores = {
            row["provider_model_id"]: float(row["external_score"])
            for row in judge["rank_table"]
        }
        direct_scores = {
            row["provider_model_id"]: float(row["external_score"])
            for row in judge["rank_table"]
            if not row["external_score_is_estimated"]
        }
        anchor_rows.append(
            {
                "judge": judge["judge_name"],
                "direct_candidate_count": len(direct_scores),
                "direct_accuracy": pairwise_accuracy(
                    judge["ranking_models"], direct_scores
                ),
                "all_candidate_count": len(all_scores),
                "all_accuracy": pairwise_accuracy(
                    judge["ranking_models"], all_scores
                ),
            }
        )

    runs = {run["name"]: run for run in crossed_report["runs"]}
    sol_own = runs["catalog_ladder50_gpt_5_6_sol"]
    fable_own = runs["catalog_ladder50_claude_fable_5"]
    fable_on_sol = runs[
        "catalog_ladder50_cross_fable_judges_sol_evidence_complete"
    ]
    sol_on_fable = runs[
        "catalog_ladder50_cross_sol_judges_fable_evidence_complete"
    ]
    same_evidence = [
        {
            "evidence": "Sol-authored",
            "kendall_tau": kendall_order(
                _checkpoint_ranking(sol_own, 6),
                _checkpoint_ranking(fable_on_sol, 6),
            ),
        },
        {
            "evidence": "Fable-authored",
            "kendall_tau": kendall_order(
                _checkpoint_ranking(fable_own, 6),
                _checkpoint_ranking(sol_on_fable, 6),
            ),
        },
    ]
    same_judge = [
        {
            "judge": "Sol",
            "kendall_tau": kendall_order(
                _checkpoint_ranking(sol_own, 6),
                _checkpoint_ranking(sol_on_fable, 6),
            ),
        },
        {
            "judge": "Fable",
            "kendall_tau": kendall_order(
                _checkpoint_ranking(fable_own, 6),
                _checkpoint_ranking(fable_on_sol, 6),
            ),
        },
    ]

    catalog_order = _catalog_order_summary(
        source_run=fable_on_sol,
        catalog_order_analysis=catalog_order_analysis,
    )
    oversight_order = oversight_order_replay["aggregate"]
    pooled_oversight = oversight_synthesis["pooled"]
    matched_extension = next(
        row
        for row in oversight_synthesis["studies"]
        if row["study"] == "oversight_frontier_v4_matched_extension"
    )
    conditions = {
        row["battery"]: row for row in verifier_council["conditions"]
    }
    return {
        "primary_catalog_pairwise_accuracy": rq1["mean_pairwise_accuracy"],
        "external_anchor_sensitivity": anchor_rows,
        "battery_replication": {
            "mean_rank_tau": catalog_stability["mean_rank_replication_tau"],
            "mean_top5_overlap": catalog_stability["mean_top5_overlap"],
            "baseline_probe_count": catalog_stability["judges"][0][
                "baseline_opening_probe_count"
            ],
            "replication_probe_count": catalog_stability["judges"][0][
                "replication_opening_probe_count"
            ],
            "judges": [
                {
                    "judge": row["judge_name"],
                    "baseline_accuracy": row[
                        "baseline_opening_pairwise_accuracy"
                    ],
                    "replication_accuracy": row[
                        "replication_opening_pairwise_accuracy"
                    ],
                    "rank_tau": row["rank_replication_tau"],
                    "top5_overlap": row["top5_overlap"],
                }
                for row in catalog_stability["judges"]
            ],
        },
        "answer_order": {
            "catalog": catalog_order,
            "oversight_panels": {
                "condition_count": oversight_order["condition_count"],
                "mean_kendall_tau": oversight_order["mean_kendall_tau"],
                "top_rank_stable_count": oversight_order[
                    "top_rank_stable_count"
                ],
                "exact_evidence": oversight_order["all_evidence_exact"],
            },
        },
        "evidence_and_interpreter": {
            "same_evidence": same_evidence,
            "same_judge": same_judge,
        },
        "probe_budget": {
            "catalog_judges": [
                {
                    "judge": row["judge_name"],
                    "checkpoints": row["replication_checkpoints"],
                }
                for row in catalog_stability["judges"]
            ],
            "oversight_improved": rq2["adaptive_improved_count"],
            "oversight_unchanged": rq2["adaptive_unchanged_count"],
            "oversight_worsened": rq2["adaptive_worsened_count"],
        },
        "oversight_uncertainty": {
            "panel_count": rq2["panel_count"],
            "judge_count": rq2["judge_count"],
            "superior_recognition_rate": rq2[
                "superior_recognition_rate"
            ],
            "margin_bins": rq2["superior_by_margin"],
            "judge_bands": rq2["judge_bands"],
        },
        "missing_evidence": {
            "fresh_catalog_opening_expected_answers": catalog_stability[
                "replication_answer_availability"
            ]["opening_expected"],
            "fresh_catalog_opening_missing_answers": catalog_stability[
                "replication_answer_availability"
            ]["opening_unavailable"],
            "fresh_catalog_adaptive_expected_answers": catalog_stability[
                "replication_answer_availability"
            ]["adaptive_expected"],
            "fresh_catalog_adaptive_missing_answers": catalog_stability[
                "replication_answer_availability"
            ]["adaptive_unavailable"],
            "pooled_oversight_expected_answers": pooled_oversight[
                "candidate_answer_count"
            ],
            "pooled_oversight_missing_answers": pooled_oversight[
                "unavailable_answer_count"
            ],
            "matched_extension_missing_answers": matched_extension[
                "unavailable_answer_count"
            ],
        },
        "council_scope": {
            "panel_count_per_battery": 4,
            "member_count": 3,
            "ordinary_anchor_accuracy": conditions["ordinary"][
                "anchor_pairwise_accuracy"
            ],
            "ordinary_council_accuracy": conditions["ordinary"][
                "council_pairwise_accuracy"
            ],
            "verifier_anchor_accuracy": conditions["verifier"][
                "anchor_pairwise_accuracy"
            ],
            "verifier_council_accuracy": conditions["verifier"][
                "council_pairwise_accuracy"
            ],
            "mean_interaction": verifier_council["matched_effects"][
                "mean_interaction"
            ],
        },
    }


def render_probe_type_contrast_svg(summary: Mapping[str, Any]) -> str:
    rows = list(summary["question_type_contrasts"])
    width, left, right, top, row_height = 1200, 335, 96, 138, 39
    height = top + len(rows) * row_height + 76
    plot_width = width - left - right

    def x(value: float) -> float:
        return left + plot_width * value

    grid = "".join(
        f"<line x1='{x(value):.1f}' y1='{top - 18}' "
        f"x2='{x(value):.1f}' y2='{height - 58}' stroke='#e1e6e7'/>"
        f"<text x='{x(value):.1f}' y='{height - 28}' text-anchor='middle' "
        f"font-size='12' fill='#5c6970'>{value:.0%}</text>"
        for value in (0, 0.25, 0.5, 0.75, 1)
    )
    marks = []
    for index, row in enumerate(rows):
        y = top + index * row_height
        high = float(row["higher_rate"])
        low = float(row["lower_rate"])
        marks.append(
            f"<text x='{left - 18}' y='{y + 4}' text-anchor='end' "
            f"font-size='13' fill='#182026'>{escape(row['label'])}</text>"
            f"<line x1='{x(low):.1f}' y1='{y}' x2='{x(high):.1f}' y2='{y}' "
            f"stroke='#bac3c6' stroke-width='2'/>"
            f"<circle cx='{x(low):.1f}' cy='{y}' r='6' fill='#b94736'/>"
            f"<circle cx='{x(high):.1f}' cy='{y}' r='6' fill='#247665'/>"
            f"<text x='{width - 18}' y='{y + 4}' text-anchor='end' "
            f"font-size='11' fill='#5c6970'>n={row['probe_count']}</text>"
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg"
viewBox="0 0 {width} {height}" role="img"
aria-label="Question types used by higher and lower capability probe authors">
<rect width="100%" height="100%" fill="#fff"/>
<text x="40" y="45" font-family="Georgia,serif" font-size="29"
fill="#182026">Stronger authors concentrated on quantitative stress tests</text>
<text x="40" y="76" font-family="system-ui,sans-serif" font-size="14"
fill="#5c6970">Mean within-author share of probes; every author has equal weight. Labels may overlap.</text>
<circle cx="42" cy="108" r="6" fill="#247665"/>
<text x="55" y="113" font-family="system-ui,sans-serif" font-size="13"
fill="#182026">Higher-capability authors</text>
<circle cx="232" cy="108" r="6" fill="#b94736"/>
<text x="245" y="113" font-family="system-ui,sans-serif" font-size="13"
fill="#182026">Lower-capability authors</text>
{grid}{''.join(marks)}
</svg>"""


def render_probe_design_profile_svg(summary: Mapping[str, Any]) -> str:
    bands = {row["label"]: row for row in summary["bands"]}
    high = bands["Higher-capability authors"]["metrics"]
    middle = bands["Middle authors"]["metrics"]
    low = bands["Lower-capability authors"]["metrics"]
    metrics = (
        ("reference_valid_rate", "Reference-valid"),
        ("objective_checkability_rate", "Author calls objectively checkable"),
        ("fully_self_solvable_rate", "Author says fully self-solvable"),
        ("multi_type_rate", "Spans multiple question types"),
        ("targets_stronger_rate", "Explicitly targets stronger models"),
        ("candidate_pair_accuracy", "Candidate pair discrimination"),
    )
    width, height = 1140, 500
    left, right, top, row_height = 380, 80, 140, 50
    plot_width = width - left - right

    def x(value: float) -> float:
        return left + plot_width * value

    grid = "".join(
        f"<line x1='{x(value):.1f}' y1='{top - 24}' "
        f"x2='{x(value):.1f}' y2='{height - 66}' stroke='#e1e6e7'/>"
        f"<text x='{x(value):.1f}' y='{height - 30}' text-anchor='middle' "
        f"font-size='12' fill='#5c6970'>{value:.0%}</text>"
        for value in (0, 0.25, 0.5, 0.75, 1)
    )
    marks = []
    for index, (key, label) in enumerate(metrics):
        y = top + index * row_height
        values = (float(high[key]), float(middle[key]), float(low[key]))
        marks.append(
            f"<text x='{left - 20}' y='{y + 5}' text-anchor='end' "
            f"font-size='14' fill='#182026'>{escape(label)}</text>"
            f"<line x1='{x(min(values)):.1f}' y1='{y}' "
            f"x2='{x(max(values)):.1f}' y2='{y}' stroke='#bac3c6' "
            f"stroke-width='2'/>"
            f"<circle cx='{x(values[0]):.1f}' cy='{y}' r='7' fill='#247665'/>"
            f"<circle cx='{x(values[1]):.1f}' cy='{y}' r='6' fill='#9b6a18'/>"
            f"<circle cx='{x(values[2]):.1f}' cy='{y}' r='7' fill='#b94736'/>"
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg"
viewBox="0 0 {width} {height}" role="img"
aria-label="Probe design and performance by author capability band">
<rect width="100%" height="100%" fill="#fff"/>
<text x="40" y="45" font-family="Georgia,serif" font-size="29"
fill="#182026">The larger difference was execution, not ambition</text>
<text x="40" y="76" font-family="system-ui,sans-serif" font-size="14"
fill="#5c6970">Author-balanced descriptive rates across 147 probes from 11 authors.</text>
<circle cx="42" cy="108" r="6" fill="#247665"/><text x="55" y="113"
font-size="13" fill="#182026">Higher</text>
<circle cx="125" cy="108" r="6" fill="#9b6a18"/><text x="138" y="113"
font-size="13" fill="#182026">Middle</text>
<circle cx="205" cy="108" r="6" fill="#b94736"/><text x="218" y="113"
font-size="13" fill="#182026">Lower</text>
{grid}{''.join(marks)}
</svg>"""


def render_author_solvability_svg(summary: Mapping[str, Any]) -> str:
    rows = list(summary["author_solvability"])
    width, left, right, top, row_height = 1200, 350, 170, 128, 47
    height = top + len(rows) * row_height + 70
    plot_width = width - left - right

    def short_name(model: str) -> str:
        return model.split("/", 1)[-1].replace("-", " ").title()

    marks = []
    for index, row in enumerate(rows):
        y = top + index * row_height
        full = float(row["fully_rate"])
        partial = float(row["partially_rate"])
        unsolved = float(row["not_solvable_rate"])
        x0 = left
        widths = (
            plot_width * full,
            plot_width * partial,
            plot_width * unsolved,
        )
        marks.append(
            f"<text x='{left - 18}' y='{y + 5}' text-anchor='end' "
            f"font-size='13' fill='#182026'>{escape(short_name(row['model']))}</text>"
            f"<text x='34' y='{y + 5}' font-size='11' fill='#5c6970'>"
            f"{row['score']:.1f}</text>"
            f"<rect x='{x0:.1f}' y='{y - 10}' width='{widths[0]:.1f}' "
            f"height='20' fill='#247665'/>"
            f"<rect x='{x0 + widths[0]:.1f}' y='{y - 10}' "
            f"width='{widths[1]:.1f}' height='20' fill='#d8b66b'/>"
            f"<rect x='{x0 + widths[0] + widths[1]:.1f}' y='{y - 10}' "
            f"width='{widths[2]:.1f}' height='20' fill='#b94736'/>"
            f"<text x='{width - 22}' y='{y + 5}' text-anchor='end' "
            f"font-size='11' fill='#5c6970'>"
            f"{row['probe_count']} probes · {row['beyond_author_rate']:.0%} surpassed</text>"
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg"
viewBox="0 0 {width} {height}" role="img"
aria-label="Probe self-solvability by author model">
<rect width="100%" height="100%" fill="#fff"/>
<text x="40" y="44" font-family="Georgia,serif" font-size="29"
fill="#182026">Authors rarely tried to write problems they could not solve</text>
<text x="40" y="75" font-family="system-ui,sans-serif" font-size="14"
fill="#5c6970">Author self-assessment, ordered by external capability. “Surpassed” is the operational beyond-author rate.</text>
<rect x="40" y="93" width="13" height="13" fill="#247665"/><text x="60" y="104" font-size="12" fill="#182026">Fully</text>
<rect x="115" y="93" width="13" height="13" fill="#d8b66b"/><text x="135" y="104" font-size="12" fill="#182026">Partly</text>
<rect x="195" y="93" width="13" height="13" fill="#b94736"/><text x="215" y="104" font-size="12" fill="#182026">Not solvable</text>
{''.join(marks)}
</svg>"""


def render_answer_matrix_svg(
    summary: Mapping[str, Any],
    display_names: Mapping[str, str],
) -> str:
    columns = list(summary["columns"])
    rows = list(summary["rows"])
    left, top, row_height, cell_width = 390, 170, 21, 88
    width = max(1260, left + cell_width * len(columns) + 90)
    height = top + row_height * len(rows) + 95
    if any(len(row["scores"]) != len(columns) for row in rows):
        raise ValueError("every answer-matrix row must match the column count")
    palette = (
        "#b94736",
        "#c96a54",
        "#d98f74",
        "#e8b99d",
        "#eee1bf",
        "#cfddbd",
        "#9fc8a9",
        "#62a28d",
        "#247665",
    )

    headers = []
    for index, column in enumerate(columns):
        x = left + index * cell_width + cell_width / 2
        headers.append(
            f"<text x='{x:.1f}' y='122' text-anchor='middle' "
            f"font-size='11' fill='#b94736'>{escape(column['author'])}</text>"
            f"<text x='{x:.1f}' y='140' text-anchor='middle' "
            f"font-size='11' fill='#182026'>{escape(column['title'])}</text>"
            f"<text x='{x:.1f}' y='156' text-anchor='middle' "
            f"font-size='10' fill='#5c6970'>μ {column['mean_answer_score']:.1f}</text>"
        )
    marks = []
    for row_index, row in enumerate(rows):
        y = top + row_index * row_height
        name = display_names.get(row["model"], row["model"])
        name = name.split(": ", 1)[-1]
        estimated = "*" if row["external_score_is_estimated"] else ""
        marks.append(
            f"<text x='34' y='{y + 4}' font-size='10' fill='#7b878c'>"
            f"{row_index + 1}</text>"
            f"<text x='{left - 72}' y='{y + 4}' text-anchor='end' "
            f"font-size='11' fill='#182026'>{escape(name)}</text>"
            f"<text x='{left - 18}' y='{y + 4}' text-anchor='end' "
            f"font-size='10' fill='#5c6970'>{row['external_score']:.1f}{estimated}</text>"
        )
        for column_index, value in enumerate(row["scores"]):
            x = left + column_index * cell_width
            if value is None:
                fill, text_color, label = "#e6eaeb", "#6c787e", "—"
            else:
                bucket = max(0, min(8, round(float(value) * 2)))
                fill = palette[bucket]
                text_color = "#fff" if bucket in (0, 1, 7, 8) else "#182026"
                label = f"{value:.1f}"
            marks.append(
                f"<rect x='{x + 2:.1f}' y='{y - 9}' "
                f"width='{cell_width - 4}' height='18' fill='{fill}'/>"
                f"<text x='{x + cell_width / 2:.1f}' y='{y + 4}' "
                f"text-anchor='middle' font-size='10' fill='{text_color}'>"
                f"{label}</text>"
            )
    legend = []
    for index, value in enumerate((0, 1, 2, 3, 4)):
        bucket = value * 2
        x = left + index * 56
        legend.append(
            f"<rect x='{x}' y='{height - 54}' width='34' height='16' "
            f"fill='{palette[bucket]}'/>"
            f"<text x='{x + 17}' y='{height - 22}' text-anchor='middle' "
            f"font-size='10' fill='#5c6970'>{value}</text>"
        )
    dividers = "".join(
        f"<line x1='{left + index * cell_width}' y1='108' "
        f"x2='{left + index * cell_width}' y2='{height - 70}' "
        "stroke='#87949a' stroke-width='2'/>"
        for index in range(1, len(columns))
        if columns[index]["author"] != columns[index - 1]["author"]
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg"
viewBox="0 0 {width} {height}" role="img"
aria-label="Answer quality of {len(rows)} candidate models on {len(columns)} model-authored probes">
<rect width="100%" height="100%" fill="#fff"/>
<text x="40" y="45" font-family="Georgia,serif" font-size="29"
fill="#182026">The same model can look different under different probes</text>
<text x="40" y="76" font-family="system-ui,sans-serif" font-size="14"
fill="#5c6970">Rows follow the external ladder. Scores range from 0 (unusable) to 4 (fully correct and rigorous).</text>
<text x='{left - 72}' y='151' text-anchor='end' font-size='11'
fill='#5c6970'>Model</text><text x='{left - 18}' y='151' text-anchor='end'
font-size='11' fill='#5c6970'>Index</text>
{''.join(headers)}
{dividers}
{''.join(marks)}
<text x='{left - 18}' y='{height - 44}' text-anchor='end' font-size='11'
fill='#5c6970'>Answer score</text>{''.join(legend)}
<text x='{left + 320}' y='{height - 36}' font-size='11' fill='#5c6970'>* estimated external anchor</text>
</svg>"""


def render_experiment_design_svg() -> str:
    width, height = 1320, 530
    panels = (
        (
            35,
            "1 · Catalog ladder",
            "Can a strong judge recover a broad ordering?",
            "One judge writes ten probes",
            "50 anonymous candidates answer",
            "The judge ranks all 50",
            "50",
        ),
        (
            455,
            "2 · Oversight frontier",
            "Can a judge recognize a model above itself?",
            "The judge is also an anonymous candidate",
            "Eight peers sit above, near, and below it",
            "We test who is placed above anonymous self",
            "9",
        ),
        (
            875,
            "3 · Independent council",
            "Can several judges use the same evidence better?",
            "Archived probes and answers are frozen",
            "Three judges rank them independently",
            "Pairwise majority combines their rankings",
            "3",
        ),
    )
    content = []
    for x, title, question, first, second, third, count in panels:
        center = x + 185
        dots = "".join(
            f"<circle cx='{center - 50 + (index % 5) * 25}' "
            f"cy='{258 + (index // 5) * 24}' r='6' fill='#b94736'/>"
            for index in range(10 if count == "50" else int(count))
        )
        if count == "50":
            dots += (
                f"<text x='{center + 82}' y='274' font-size='13' "
                f"fill='#5c6970'>× 5 rows</text>"
            )
        judges = (
            "".join(
                f"<circle cx='{center - 35 + index * 35}' cy='357' "
                f"r='14' fill='#247665'/>"
                for index in range(3)
            )
            if count == "3"
            else f"<circle cx='{center}' cy='357' r='17' fill='#247665'/>"
        )
        content.append(
            f"<rect x='{x}' y='28' width='385' height='465' rx='6' "
            f"fill='#f3f5f5' stroke='#d8dee1'/>"
            f"<text x='{x + 24}' y='69' font-family='Georgia,serif' "
            f"font-size='23' fill='#182026'>{escape(title)}</text>"
            f"<text x='{x + 24}' y='101' font-size='13' fill='#5c6970'>"
            f"{escape(question)}</text>"
            f"<circle cx='{center}' cy='156' r='20' fill='#247665'/>"
            f"<text x='{center}' y='161' text-anchor='middle' font-size='13' "
            f"font-weight='700' fill='#fff'>J</text>"
            f"<text x='{center}' y='195' text-anchor='middle' font-size='12' "
            f"fill='#182026'>{escape(first)}</text>"
            f"<path d='M {center} 207 V 226' stroke='#87949a' "
            f"stroke-width='2' marker-end='url(#arrow)'/>"
            f"{dots}"
            f"<text x='{center}' y='322' text-anchor='middle' font-size='12' "
            f"fill='#182026'>{escape(second)}</text>"
            f"<path d='M {center} 330 V 338' stroke='#87949a' "
            f"stroke-width='2' marker-end='url(#arrow)'/>"
            f"{judges}"
            f"<text x='{center}' y='401' text-anchor='middle' font-size='12' "
            f"fill='#182026'>{escape(third)}</text>"
            f"<line x1='{x + 55}' y1='428' x2='{x + 330}' y2='428' "
            f"stroke='#9b6a18' stroke-width='3'/>"
            f"<circle cx='{x + 80}' cy='451' r='8' fill='#9b6a18'/>"
            f"<circle cx='{x + 145}' cy='451' r='8' fill='#9b6a18'/>"
            f"<circle cx='{x + 210}' cy='451' r='8' fill='#9b6a18'/>"
            f"<circle cx='{x + 275}' cy='451' r='8' fill='#9b6a18'/>"
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg"
viewBox="0 0 {width} {height}" role="img"
aria-label="The catalog ladder, oversight frontier, and independent council experiments">
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="5" refY="3"
orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#87949a"/></marker></defs>
<rect width="100%" height="100%" fill="#fff"/>
{''.join(content)}
</svg>"""


def render_robustness_markdown(summary: Mapping[str, Any]) -> str:
    battery = summary["battery_replication"]
    order = summary["answer_order"]
    evidence = summary["evidence_and_interpreter"]
    budget = summary["probe_budget"]
    budget_by_judge = {
        row["judge"]: row for row in budget["catalog_judges"]
    }
    sol_budget = budget_by_judge["Sol"]["checkpoints"]
    fable_budget = budget_by_judge["Fable"]["checkpoints"]
    uncertainty = summary["oversight_uncertainty"]
    missing = summary["missing_evidence"]
    council = summary["council_scope"]
    anchors = summary["external_anchor_sensitivity"]
    direct_candidate_count = min(
        row["direct_candidate_count"] for row in anchors
    )
    anchor_rows = "\n".join(
        f"| {row['judge']} | {row['direct_candidate_count']} | "
        f"{row['direct_accuracy']:.1%} | {row['all_candidate_count']} | "
        f"{row['all_accuracy']:.1%} |"
        for row in anchors
    )
    battery_rows = "\n".join(
        f"| {row['judge']} | {row['baseline_accuracy']:.1%} | "
        f"{row['replication_accuracy']:.1%} | {row['rank_tau']:.2f} | "
        f"{row['top5_overlap']:.0%} |"
        for row in battery["judges"]
    )
    return f"""# Robustness And Sensitivity Appendix

This appendix gathers the checks that qualify the headline results. It is
generated from the same frozen JSON artifacts as the public article; no
additional model calls are involved.

## External Reference And Estimated Anchors

The primary catalog metric uses {direct_candidate_count} candidates with directly reported external
scores. Three deliberately weak anchors have estimated scores and are excluded
from that endpoint.

| Judge | Direct models | Direct pair accuracy | All models | Accuracy with estimates |
| --- | ---: | ---: | ---: | ---: |
{anchor_rows}

Including the estimated lower-tail anchors raises measured accuracy slightly
because those candidates are easy to separate. It does not drive the primary
{summary['primary_catalog_pairwise_accuracy']:.1%} result.

## Opening Probe Budget

| Judge | {battery['baseline_probe_count']} probes | {battery['replication_probe_count']} probes | Rank tau | Top-five overlap |
| --- | ---: | ---: | ---: | ---: |
{battery_rows}

The mean rank agreement is Kendall {battery['mean_rank_tau']:.2f}, with
{battery['mean_top5_overlap']:.0%} mean top-five overlap. Aggregate accuracy is
higher with the larger opening battery; exact frontier ordering still varies.

## Answer Presentation Order

The 50-model catalog replay changed only answer order. Its ranking agreed with
the original at Kendall {order['catalog']['kendall_tau']:.2f}; pairwise accuracy
moved from {order['catalog']['source_accuracy']:.1%} to
{order['catalog']['replay_accuracy']:.1%}. The top three stayed fixed and
{order['catalog']['top10_overlap']} of the top ten overlapped.

{order['oversight_panels']['condition_count']} smaller oversight panels were
also replayed under new answer orders with
exact evidence identity. Mean rank agreement was Kendall
{order['oversight_panels']['mean_kendall_tau']:.2f}, and the top candidate
remained fixed in all {order['oversight_panels']['condition_count']} panels.
Order is therefore a real source of individual-rank uncertainty, but it did not
erase the broad signal.

## Probe Battery Versus Evidence Interpreter

With the same archived evidence, judge-to-judge Kendall agreement was
{evidence['same_evidence'][0]['kendall_tau']:.2f} on Sol-authored evidence and
{evidence['same_evidence'][1]['kendall_tau']:.2f} on Fable-authored evidence.
For the same judge across different batteries, agreement was
{evidence['same_judge'][0]['kendall_tau']:.2f} for Sol and
{evidence['same_judge'][1]['kendall_tau']:.2f} for Fable. In this cross-over,
the chosen evidence changed rankings more than swapping the interpreter of
fixed evidence.

## Probe Count And Adaptive Follow-Ups

In the catalog experiment, Sol moved from
{sol_budget[0]['pairwise_accuracy']:.1%} after
{sol_budget[0]['probe_count']} probes to
{sol_budget[-1]['pairwise_accuracy']:.1%} after
two follow-ups. Fable moved from
{fable_budget[0]['pairwise_accuracy']:.1%} to
{fable_budget[-1]['pairwise_accuracy']:.1%}.
Across {uncertainty['panel_count']} oversight panels, one follow-up improved
{budget['oversight_improved']} rankings, left
{budget['oversight_unchanged']} unchanged, and worsened
{budget['oversight_worsened']}. More evidence was not monotonically better.

## Oversight Uncertainty

The pooled superior-recognition estimate is
{uncertainty['superior_recognition_rate']:.1%} across
{uncertainty['panel_count']} panels and {uncertainty['judge_count']} judges.
Wilson intervals are wide within margin cells; panel-bootstrap intervals for
the standardized sub-ten-point rates are:

| Judge band | Rate | Panel-bootstrap 95% interval |
| --- | ---: | ---: |
{chr(10).join(
    f"| {row['label'].title()} | {row['subten_standardized_rate']:.1%} | "
    f"{row['bootstrap_95_low']:.1%}–{row['bootstrap_95_high']:.1%} |"
    for row in uncertainty['judge_bands']
)}

The observed judge-capability pattern is suggestive, not causal: judge,
provider, panel, and probe battery remain entangled.

## Missing Evidence And Provider Failures

The primary catalog opening contains
{missing['fresh_catalog_opening_expected_answers']} candidate answers, with
{missing['fresh_catalog_opening_missing_answers']} unavailable. The adaptive
rounds contain {missing['fresh_catalog_adaptive_expected_answers']} answer
cells, with {missing['fresh_catalog_adaptive_missing_answers']} unavailable. The
pooled oversight study retains {missing['pooled_oversight_missing_answers']} of
{missing['pooled_oversight_expected_answers']} unavailable answer cells
({missing['pooled_oversight_missing_answers'] / missing['pooled_oversight_expected_answers']:.1%});
the matched extension has none. Repairs reused successful evidence and preserve
lineage. Provider incompatibility is treated as missing evidence, never as an
intelligence score.

## Council Scope

The three-member council improved ordinary-battery accuracy from
{council['ordinary_anchor_accuracy']:.1%} to
{council['ordinary_council_accuracy']:.1%} across
{council['panel_count_per_battery']} panels. On
verifier-oriented evidence it moved from
{council['verifier_anchor_accuracy']:.1%} to
{council['verifier_council_accuracy']:.1%}. The interaction was
{council['mean_interaction']:+.1%}; the interventions were not complementary.
This is a fixed-composition pilot, not evidence for every possible council.

## Remaining Limits

- The external intelligence index is a reference measurement, not ground truth.
- Nearby frontier scores and exact model settings are uncertain.
- Taxonomy labels are post-hoc and multi-label; they describe behavior but do
  not yet select reliably diagnostic probes.
- One fixed evaluator scored the 147-probe corpus.
- A single self-solve attempt cannot establish that a problem is impossible for
  its author.
- Free discussion remains a qualitative extension, not a quantitative claim
  about machine social order.

Primary artifacts:
`data/research_question_synthesis.json`,
`data/catalog_ladder50_opening10_stability.json`,
`data/oversight_frontier_v1_order_replay_results.json`,
`data/probe_effectiveness_results.json`, and
`data/verifier_council_matched_v1_results.json`.
"""


def _catalog_order_summary(
    *,
    source_run: Mapping[str, Any],
    catalog_order_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    source_ranking = _checkpoint_ranking(source_run, 4)
    replay_judgment = catalog_order_analysis["prior_agreement"]["judgments"][-1]
    replay_ranking = replay_judgment["ranking"]
    direct_scores = {
        participant_id: float(
            source_run["prior_participant_scores"][participant_id]
        )
        for participant_id in source_run["prior_reported_score_participants"]
    }
    source_position = {
        participant_id: rank
        for rank, participant_id in enumerate(source_ranking, start=1)
    }
    replay_position = {
        participant_id: rank
        for rank, participant_id in enumerate(replay_ranking, start=1)
    }
    displacements = sorted(
        abs(source_position[item] - replay_position[item])
        for item in source_ranking
    )
    return {
        "source_accuracy": _checkpoint(source_run, 4)["pairwise_accuracy"],
        "replay_accuracy": pairwise_accuracy(
            replay_ranking,
            direct_scores,
        ),
        "kendall_tau": kendall_order(source_ranking, replay_ranking),
        "top3_overlap": len(set(source_ranking[:3]) & set(replay_ranking[:3])),
        "top5_overlap": len(set(source_ranking[:5]) & set(replay_ranking[:5])),
        "top10_overlap": len(
            set(source_ranking[:10]) & set(replay_ranking[:10])
        ),
        "median_absolute_displacement": median(displacements),
        "mean_absolute_displacement": mean(displacements),
    }


def _checkpoint(run: Mapping[str, Any], probe_count: int) -> Mapping[str, Any]:
    return next(
        row
        for row in run["probe_budget_results"]
        if int(row["probe_count"]) == probe_count
    )


def _checkpoint_ranking(
    run: Mapping[str, Any], probe_count: int
) -> list[str]:
    return list(_checkpoint(run, probe_count)["ranking"])


def _author_label_rate(
    rows: Sequence[Mapping[str, Any]],
    author: str,
    label_key: str,
    label: str,
) -> float:
    author_rows = [row for row in rows if row["author_model"] == author]
    return mean(label in row.get(label_key, []) for row in author_rows)


def pairwise_accuracy(
    ordering: Sequence[str],
    scores: Mapping[str, float],
) -> float:
    if len(set(ordering)) != len(ordering):
        raise ValueError("ordering contains duplicate participants")
    positions = {item: index for index, item in enumerate(ordering)}
    comparable = [item for item in ordering if item in scores]
    outcomes = []
    for left, right in combinations(comparable, 2):
        if scores[left] == scores[right]:
            continue
        outcomes.append(
            (positions[left] < positions[right])
            == (scores[left] > scores[right])
        )
    return mean(outcomes) if outcomes else 0.0


def score_pairwise_accuracy(
    predictions: Mapping[str, float],
    scores: Mapping[str, float],
) -> float:
    outcomes = []
    for left, right in combinations(
        sorted(predictions.keys() & scores.keys()), 2
    ):
        if scores[left] == scores[right]:
            continue
        prediction_delta = predictions[left] - predictions[right]
        if prediction_delta == 0:
            outcomes.append(0.5)
        else:
            outcomes.append(
                (prediction_delta > 0) == (scores[left] > scores[right])
            )
    return mean(outcomes) if outcomes else 0.0


def ranked(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + end + 1) / 2
        for offset in range(index, end):
            result[order[offset]] = average_rank
        index = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks = ranked(left)
    right_ranks = ranked(right)
    left_mean = mean(left_ranks)
    right_mean = mean(right_ranks)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks, strict=True)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left_ranks)
    right_variance = sum((value - right_mean) ** 2 for value in right_ranks)
    denominator = (left_variance * right_variance) ** 0.5
    return numerator / denominator if denominator else 0.0


def kendall_order(left: Sequence[str], right: Sequence[str]) -> float:
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError("rankings must not contain duplicate participants")
    if set(left) != set(right) or len(left) < 2:
        return 0.0
    right_position = {item: index for index, item in enumerate(right)}
    outcomes = []
    for first, second in combinations(left, 2):
        outcomes.append(right_position[first] < right_position[second])
    concordant = sum(outcomes)
    discordant = len(outcomes) - concordant
    return (
        (concordant - discordant) / len(outcomes) if outcomes else 0.0
    )


def write_publication_outputs(
    *,
    summary: Mapping[str, Any],
    output_json: Path,
    figure_dir: Path,
    robustness_markdown: Path,
    display_names: Mapping[str, str],
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    probe_summary = summary["probe_author_analysis"]
    figures = {
        "probe-types-by-capability.svg": render_probe_type_contrast_svg(
            probe_summary
        ),
        "probe-design-by-capability.svg": render_probe_design_profile_svg(
            probe_summary
        ),
        "probe-self-solvability.svg": render_author_solvability_svg(
            probe_summary
        ),
        "model-probe-solvability.svg": render_answer_matrix_svg(
            summary["answer_matrix"],
            display_names,
        ),
        "experiment-designs.svg": render_experiment_design_svg(),
    }
    for filename, contents in figures.items():
        (figure_dir / filename).write_text(contents, encoding="utf-8")
    robustness_markdown.parent.mkdir(parents=True, exist_ok=True)
    robustness_markdown.write_text(
        render_robustness_markdown(summary["robustness"]),
        encoding="utf-8",
    )


def _display_label(value: str) -> str:
    return value.replace("_", " ").title()
