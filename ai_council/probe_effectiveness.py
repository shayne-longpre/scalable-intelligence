from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import random
from statistics import mean, median
from typing import Any, Callable, Mapping, Sequence

from ai_council.probe_catalog import select_primary_occurrence
from ai_council.probe_study import spearman


SCHEMA_VERSION = "probe-effectiveness-v1"
FEATURE_PRIOR_WEIGHT = 5


def build_probe_effectiveness_report(
    *,
    self_study_path: str | Path,
    probe_catalog_path: str | Path,
    probe_evolution_path: str | Path,
    output_dir: str | Path,
    published_json_path: str | Path | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260728,
) -> dict[str, Any]:
    self_study_path = Path(self_study_path)
    probe_catalog_path = Path(probe_catalog_path)
    probe_evolution_path = Path(probe_evolution_path)
    self_study = _load_json(self_study_path)
    catalog = _load_json(probe_catalog_path)
    evolution = _load_json(probe_evolution_path)
    rows = _probe_rows(self_study, catalog)

    held_out = [
        held_out_label_analysis(
            rows,
            name=name,
            feature_filter=feature_filter,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
        )
        for index, (name, feature_filter) in enumerate(
            (
                ("Question types", lambda value: value.startswith("type:")),
                (
                    "Strategies",
                    lambda value: value.startswith(("strategy:", "stage:")),
                ),
                ("All recorded labels", lambda value: True),
            )
        )
    ]
    question_type_effects = author_centered_label_effects(
        rows,
        label_key="question_types",
        min_probes=8,
        min_authors=3,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "self_study_file": str(self_study_path),
        "probe_catalog_file": str(probe_catalog_path),
        "probe_evolution_file": str(probe_evolution_path),
        "probe_count": len(rows),
        "author_count": len({row["author_model"] for row in rows}),
        "reference_valid_rate": mean(
            row["reference_validity"] == "valid" for row in rows
        ),
        "mean_candidate_pair_accuracy": mean(
            row["candidate_pair_accuracy"] for row in rows
        ),
        "author_score_pair_accuracy_spearman": spearman(
            [row["author_score"] for row in rows],
            [row["candidate_pair_accuracy"] for row in rows],
        ),
        "question_type_effects": question_type_effects,
        "held_out_label_prediction": held_out,
        "by_stage": _group_summary(rows, "stage"),
        "by_intended_level": _assessment_group_summary(
            rows, "intended_level"
        ),
        "by_checkability": _assessment_group_summary(rows, "checkability"),
        "by_self_solvability": _assessment_group_summary(
            rows, "self_solvability"
        ),
        "dynamics": _dynamics_summary(evolution),
        "most_diagnostic_examples": _diagnostic_examples(rows, reverse=True),
        "least_diagnostic_examples": _diagnostic_examples(rows, reverse=False),
        "beyond_author_count": sum(row["beyond_author"] for row in rows),
        "valid_self_reported_unsolvable_count": sum(
            row["reference_validity"] == "valid"
            and row["author_assessment"]["self_solvability"]
            == "not_solvable"
            for row in rows
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "question-type-effects.svg").write_text(
        render_question_type_effects_svg(question_type_effects),
        encoding="utf-8",
    )
    (output_dir / "held-out-labels.svg").write_text(
        render_held_out_labels_svg(held_out),
        encoding="utf-8",
    )
    if published_json_path is not None:
        published_path = Path(published_json_path)
        published_path.parent.mkdir(parents=True, exist_ok=True)
        published_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return summary


def held_out_label_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    feature_filter: Callable[[str], bool],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    predictions = []
    for test in rows:
        train = [
            row
            for row in rows
            if row["author_model"] != test["author_model"]
        ]
        global_mean = mean(
            float(row["candidate_pair_accuracy"]) for row in train
        )
        feature_scores = []
        for feature in filter(feature_filter, test["features"]):
            observed = [
                float(row["candidate_pair_accuracy"])
                for row in train
                if feature in row["features"]
            ]
            if observed:
                feature_scores.append(
                    (
                        sum(observed)
                        + FEATURE_PRIOR_WEIGHT * global_mean
                    )
                    / (len(observed) + FEATURE_PRIOR_WEIGHT)
                )
        predictions.append(
            mean(feature_scores) if feature_scores else global_mean
        )

    author_concordance = []
    pair_correct = 0.0
    pair_count = 0
    for author in sorted({row["author_model"] for row in rows}):
        indices = [
            index
            for index, row in enumerate(rows)
            if row["author_model"] == author
        ]
        correct = 0.0
        count = 0
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                prediction_delta = predictions[left] - predictions[right]
                if prediction_delta == 0:
                    continue
                outcome_delta = (
                    float(rows[left]["candidate_pair_accuracy"])
                    - float(rows[right]["candidate_pair_accuracy"])
                )
                correct += (
                    0.5
                    if outcome_delta == 0
                    else float(prediction_delta * outcome_delta > 0)
                )
                count += 1
        if count:
            author_concordance.append(correct / count)
            pair_correct += correct
            pair_count += count

    if not author_concordance:
        raise ValueError(
            "held-out analysis requires at least one author with "
            "two probes receiving distinct predictions"
        )

    rng = random.Random(bootstrap_seed)
    bootstrap = sorted(
        mean(rng.choice(author_concordance) for _ in author_concordance)
        for _ in range(bootstrap_samples)
    )
    high, low, equal = [], [], []
    for author in sorted({row["author_model"] for row in rows}):
        indices = [
            index
            for index, row in enumerate(rows)
            if row["author_model"] == author
        ]
        midpoint = median(predictions[index] for index in indices)
        high.extend(index for index in indices if predictions[index] > midpoint)
        low.extend(index for index in indices if predictions[index] < midpoint)
        equal.extend(
            index for index in indices if predictions[index] == midpoint
        )
    return {
        "name": name,
        "probe_count": len(rows),
        "author_count": len(author_concordance),
        "global_spearman": spearman(
            predictions,
            [float(row["candidate_pair_accuracy"]) for row in rows],
        ),
        "within_author_pair_concordance": (
            pair_correct / pair_count if pair_count else None
        ),
        "mean_author_concordance": mean(author_concordance),
        "bootstrap_95_low": bootstrap[
            int(0.025 * len(bootstrap))
        ],
        "bootstrap_95_high": bootstrap[
            max(0, int(0.975 * len(bootstrap)) - 1)
        ],
        "high_prediction_probe_count": len(high),
        "high_prediction_mean_accuracy": _index_mean(rows, high),
        "low_prediction_probe_count": len(low),
        "low_prediction_mean_accuracy": _index_mean(rows, low),
        "median_tie_probe_count": len(equal),
    }


def author_centered_label_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_key: str,
    min_probes: int,
    min_authors: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    authors = sorted({str(row["author_model"]) for row in rows})
    by_author = {
        author: [row for row in rows if row["author_model"] == author]
        for author in authors
    }
    labels = sorted(
        {
            label
            for row in rows
            for label in (row.get(label_key) or ["unclassified"])
        }
    )
    output = []
    for label_index, label in enumerate(labels):
        author_effects = []
        probe_count = 0
        for author, author_rows in by_author.items():
            selected = [
                row
                for row in author_rows
                if label in (row.get(label_key) or ["unclassified"])
            ]
            if not selected:
                continue
            probe_count += len(selected)
            author_effects.append(
                mean(
                    float(row["candidate_pair_accuracy"])
                    for row in selected
                )
                - mean(
                    float(row["candidate_pair_accuracy"])
                    for row in author_rows
                )
            )
        if probe_count < min_probes or len(author_effects) < min_authors:
            continue
        rng = random.Random(bootstrap_seed + label_index)
        bootstrap = sorted(
            mean(rng.choice(author_effects) for _ in author_effects)
            for _ in range(bootstrap_samples)
        )
        output.append(
            {
                "label": label,
                "probe_count": probe_count,
                "author_count": len(author_effects),
                "author_centered_accuracy_effect": mean(author_effects),
                "bootstrap_95_low": bootstrap[
                    int(0.025 * len(bootstrap))
                ],
                "bootstrap_95_high": bootstrap[
                    max(0, int(0.975 * len(bootstrap)) - 1)
                ],
            }
        )
    return sorted(
        output,
        key=lambda row: -row["author_centered_accuracy_effect"],
    )


def render_question_type_effects_svg(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    width = 1120
    left, right, top, row_height = 335, 100, 115, 42
    height = top + row_height * len(rows) + 74
    plot_width = width - left - right
    domain = 0.14

    def x(value: float) -> float:
        return left + plot_width * (float(value) + domain) / (2 * domain)

    grid = "".join(
        f"<line x1='{x(value):.1f}' y1='{top - 18}' "
        f"x2='{x(value):.1f}' y2='{height - 58}' "
        f"stroke='{'#88949a' if value == 0 else '#e2e6e8'}'/>"
        f"<text x='{x(value):.1f}' y='{height - 28}' text-anchor='middle' "
        f"font-size='12' fill='#5e6a71'>{value:+.0%}</text>"
        for value in (-0.1, -0.05, 0.0, 0.05, 0.1)
    )
    marks = []
    for index, row in enumerate(rows):
        y = top + index * row_height
        effect = float(row["author_centered_accuracy_effect"])
        low = max(-domain, float(row["bootstrap_95_low"]))
        high = min(domain, float(row["bootstrap_95_high"]))
        color = "#247665" if effect >= 0 else "#b94736"
        marks.append(
            f"<text x='{left - 14}' y='{y + 4}' text-anchor='end' "
            f"font-size='13' fill='#182026'>{escape(_label(row['label']))}</text>"
            f"<line x1='{x(low):.1f}' y1='{y}' x2='{x(high):.1f}' y2='{y}' "
            f"stroke='{color}' stroke-width='2' opacity='.6'/>"
            f"<circle cx='{x(effect):.1f}' cy='{y}' r='6' fill='{color}'/>"
            f"<text x='{width - 18}' y='{y + 4}' text-anchor='end' "
            f"font-size='11' fill='#5e6a71'>n={row['probe_count']}, "
            f"{row['author_count']} authors</text>"
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg"
viewBox="0 0 {width} {height}" role="img"
aria-label="Author-centered diagnostic accuracy by question type">
<rect width="100%" height="100%" fill="#fff"/>
<text x="40" y="45" font-family="Georgia,serif" font-size="29"
fill="#182026">Some question families look promising, but uncertainty is wide</text>
<text x="40" y="75" font-family="system-ui,sans-serif" font-size="14"
fill="#5e6a71">Difference from each author's own mean probe accuracy; author bootstrap intervals.</text>
{grid}{''.join(marks)}
</svg>"""


def render_held_out_labels_svg(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    width, height = 1040, 390
    left, top, plot_width, plot_height = 250, 115, 670, 190

    def x(value: float) -> float:
        return left + plot_width * (float(value) - 0.35) / 0.3

    grid = "".join(
        f"<line x1='{x(value):.1f}' y1='{top - 20}' "
        f"x2='{x(value):.1f}' y2='{top + plot_height}' "
        f"stroke='{'#88949a' if value == 0.5 else '#e2e6e8'}'/>"
        f"<text x='{x(value):.1f}' y='{top + plot_height + 28}' "
        f"text-anchor='middle' font-size='12' fill='#5e6a71'>{value:.0%}</text>"
        for value in (0.4, 0.5, 0.6)
    )
    marks = []
    for index, row in enumerate(rows):
        y = top + 30 + index * 58
        value = float(row["mean_author_concordance"])
        low = float(row["bootstrap_95_low"])
        high = float(row["bootstrap_95_high"])
        marks.append(
            f"<text x='{left - 18}' y='{y + 5}' text-anchor='end' "
            f"font-size='14' fill='#182026'>{escape(row['name'])}</text>"
            f"<line x1='{x(low):.1f}' y1='{y}' x2='{x(high):.1f}' y2='{y}' "
            "stroke='#2364aa' stroke-width='3' opacity='.6'/>"
            f"<circle cx='{x(value):.1f}' cy='{y}' r='7' fill='#2364aa'/>"
            f"<text x='{x(value) + 14:.1f}' y='{y + 5}' "
            f"font-size='12' fill='#182026'>{value:.1%}</text>"
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg"
viewBox="0 0 {width} {height}" role="img"
aria-label="Held-out prediction of probe effectiveness from taxonomy labels">
<rect width="100%" height="100%" fill="#fff"/>
<text x="42" y="45" font-family="Georgia,serif" font-size="29"
fill="#182026">Taxonomy labels did not reliably select better probes</text>
<text x="42" y="75" font-family="system-ui,sans-serif" font-size="14"
fill="#5e6a71">Labels learned from other authors; ability to order a held-out author's probes. Chance is 50%.</text>
{grid}{''.join(marks)}
</svg>"""


def _probe_rows(
    self_study: Mapping[str, Any],
    catalog: Mapping[str, Any],
) -> list[dict[str, Any]]:
    catalog_by_id = {
        row["probe_id"]: row for row in catalog.get("probes", [])
    }
    rows = []
    for row in self_study.get("probes", []):
        if not isinstance(row.get("candidate_pair_accuracy"), (int, float)):
            continue
        author_score = row.get("author_score")
        assessment = row.get("author_assessment")
        if not isinstance(author_score, (int, float)) or not isinstance(
            assessment, Mapping
        ):
            continue
        catalog_row = catalog_by_id.get(row["probe_id"])
        occurrence = (
            select_primary_occurrence(catalog_row) if catalog_row else {}
        )
        question_types = list(row.get("question_types") or ["unclassified"])
        strategy_tags = list(row.get("strategy_tags") or ["unclassified"])
        features = [
            *(f"type:{value}" for value in question_types),
            *(f"strategy:{value}" for value in strategy_tags),
            f"stage:{row.get('stage')}",
            f"checkability:{assessment.get('checkability')}",
            f"intended_level:{assessment.get('intended_level')}",
            f"self_solvability:{assessment.get('self_solvability')}",
        ]
        rows.append(
            {
                **row,
                "author_score": float(author_score),
                "candidate_pair_accuracy": float(
                    row["candidate_pair_accuracy"]
                ),
                "question_types": question_types,
                "strategy_tags": strategy_tags,
                "transition": occurrence.get("transition"),
                "features": features,
            }
        )
    return rows


def _group_summary(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return [
        _summarize_rows(label, values)
        for label, values in sorted(grouped.items())
    ]


def _assessment_group_summary(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["author_assessment"].get(key) or "unknown")].append(
            row
        )
    return [
        _summarize_rows(label, values)
        for label, values in sorted(grouped.items())
    ]


def _summarize_rows(
    label: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "label": label,
        "probe_count": len(rows),
        "mean_pair_accuracy": mean(
            float(row["candidate_pair_accuracy"]) for row in rows
        ),
        "reference_valid_rate": mean(
            row["reference_validity"] == "valid" for row in rows
        ),
        "beyond_author_rate": mean(bool(row["beyond_author"]) for row in rows),
        "mean_score_range": mean(
            float(row["candidate_score_range"]) for row in rows
        ),
    }


def _dynamics_summary(evolution: Mapping[str, Any]) -> dict[str, Any]:
    cohort = evolution["cohorts"]["frontier_fixed_protocol"]
    transitions = cohort["transition_counts"]
    adaptive = {
        key: value
        for key, value in transitions.items()
        if key.startswith("adaptive_")
    }
    opening = {
        key: value
        for key, value in transitions.items()
        if key.startswith("preplanned_")
    }
    return {
        "fixed_protocol_run_count": cohort["run_count"],
        "opening_transition_counts": opening,
        "adaptive_transition_counts": adaptive,
        "adaptive_intent_counts": cohort["adaptive_intent_counts"],
    }


def _diagnostic_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    reverse: bool,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row["reference_validity"] == "valid"
        and int(row["candidate_pair_count"]) >= 10
    ]
    selected = sorted(
        eligible,
        key=lambda row: (
            float(row["candidate_pair_accuracy"]),
            int(row["candidate_pair_count"]),
        ),
        reverse=reverse,
    )[:5]
    return [
        {
            key: row.get(key)
            for key in (
                "probe_id",
                "author_model",
                "author_score",
                "question_excerpt",
                "question_types",
                "strategy_tags",
                "stage",
                "candidate_pair_accuracy",
                "candidate_pair_count",
                "beyond_author",
                "source_run",
                "question_turn_id",
            )
        }
        for row in selected
    ]


def _index_mean(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> float | None:
    if not indices:
        return None
    return mean(
        float(rows[index]["candidate_pair_accuracy"]) for index in indices
    )


def _label(value: str) -> str:
    return value.replace("_", " ").title()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
