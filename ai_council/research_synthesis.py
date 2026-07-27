from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping, Sequence

from ai_council.oversight_synthesis import (
    MARGIN_BINS,
    superior_observations,
    wilson_interval,
)
from ai_council.probe_study import spearman


SCHEMA_VERSION = "research-question-synthesis-v1"
JUDGE_BANDS = ("lower third", "middle third", "upper third")


def build_research_synthesis(
    *,
    catalog_stability_path: str | Path,
    oversight_synthesis_path: str | Path,
    output_dir: str | Path,
    published_json_path: str | Path | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260728,
) -> dict[str, Any]:
    stability_path = Path(catalog_stability_path)
    oversight_path = Path(oversight_synthesis_path)
    stability = _load_json(stability_path)
    replication = _load_json(Path(stability["replication_summary"]))
    oversight = _load_json(oversight_path)
    conditions = _load_oversight_conditions(oversight)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_stability_file": str(stability_path),
        "oversight_synthesis_file": str(oversight_path),
        "rq1_catalog_ranking": summarize_catalog_ranking(
            replication,
            stability,
        ),
        "rq2_oversight_frontier": summarize_oversight_frontier(
            oversight,
            conditions,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "oversight-frontier.svg").write_text(
        render_oversight_frontier_svg(summary["rq2_oversight_frontier"]),
        encoding="utf-8",
    )
    if published_json_path is not None:
        published_path = Path(published_json_path)
        published_path.parent.mkdir(parents=True, exist_ok=True)
        published_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def summarize_catalog_ranking(
    report: Mapping[str, Any],
    stability: Mapping[str, Any],
) -> dict[str, Any]:
    judges = [_catalog_judge_row(run) for run in report["runs"]]
    if len(judges) < 2:
        raise ValueError("catalog synthesis requires at least two judges")
    model_sets = {frozenset(row["ranking_models"]) for row in judges}
    if len(model_sets) != 1:
        raise ValueError("catalog judges must rank the same model roster")

    gap_labels = [
        row["label"] for row in judges[0]["pairwise_accuracy_by_score_gap"]
    ]
    gap_rows = []
    for label in gap_labels:
        cells = [
            next(
                cell
                for cell in judge["pairwise_accuracy_by_score_gap"]
                if cell["label"] == label
            )
            for judge in judges
        ]
        pair_count = sum(int(cell["pair_count"]) for cell in cells)
        correct = sum(float(cell["correct_pairs"]) for cell in cells)
        gap_rows.append(
            {
                "label": label,
                "correct_pairs": correct,
                "pair_count": pair_count,
                "accuracy": correct / pair_count if pair_count else None,
            }
        )

    primary = max(judges, key=lambda row: row["pairwise_accuracy"])
    stability_by_model = {
        row["judge_model"]: row for row in stability["judges"]
    }
    for judge in judges:
        prior = stability_by_model.get(judge["judge_model"])
        judge["rank_replication_tau"] = (
            prior.get("rank_replication_tau") if prior else None
        )
        judge["top5_replication_overlap"] = (
            prior.get("top5_overlap") if prior else None
        )

    return {
        "candidate_count": judges[0]["candidate_count"],
        "reported_score_candidate_count": judges[0][
            "reported_score_candidate_count"
        ],
        "judge_count": len(judges),
        "opening_probe_count": 5,
        "judges": judges,
        "mean_pairwise_accuracy": mean(
            row["pairwise_accuracy"] for row in judges
        ),
        "interjudge_kendall_tau": kendall_order(
            judges[0]["ranking_models"],
            judges[1]["ranking_models"],
        ),
        "interjudge_top5_overlap": len(
            set(judges[0]["ranking_models"][:5])
            & set(judges[1]["ranking_models"][:5])
        ),
        "mean_rank_replication_tau": stability[
            "mean_rank_replication_tau"
        ],
        "mean_top5_replication_overlap": stability["mean_top5_overlap"],
        "pairwise_accuracy_by_score_gap": gap_rows,
        "primary_judge_model": primary["judge_model"],
        "primary_judge_name": primary["judge_name"],
        "primary_pairwise_accuracy": primary["pairwise_accuracy"],
        "primary_ranking": primary["rank_table"],
        "replication_reported_cost_usd": stability[
            "replication_reported_cost_usd"
        ],
    }


def summarize_oversight_frontier(
    synthesis: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    observations = superior_observations(conditions, stage="opening")
    score_bands = _judge_score_bands(observations)
    for row in observations:
        row["judge_band"] = score_bands[float(row["judge_score"])]

    margin_weights = _subten_margin_weights(observations)
    band_rows = []
    for band in JUDGE_BANDS:
        selected = [
            row for row in observations if row["judge_band"] == band
        ]
        cells = _margin_cells(selected)
        standardized = _standardized_rate(cells, margin_weights)
        low, high = _panel_bootstrap_interval(
            selected,
            margin_weights,
            samples=bootstrap_samples,
            seed=bootstrap_seed + JUDGE_BANDS.index(band),
        )
        band_rows.append(
            {
                "label": band,
                "judge_score_min": min(
                    row["judge_score"] for row in selected
                ),
                "judge_score_max": max(
                    row["judge_score"] for row in selected
                ),
                "panel_count": len(
                    {
                        (row["study"], row["condition_id"])
                        for row in selected
                    }
                ),
                "cells": cells,
                "subten_standardized_rate": standardized,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )

    panel_rows = []
    for condition in conditions:
        total = int(condition["opening"]["superior_total"])
        if not total:
            continue
        panel_rows.append(
            {
                "judge_score": float(condition["judge_external_score"]),
                "recognition_rate": (
                    int(condition["opening"]["superior_recognized"]) / total
                ),
            }
        )
    pooled = synthesis["pooled"]
    return {
        "panel_count": pooled["condition_count"],
        "judge_count": pooled["judge_count"],
        "candidate_pair_accuracy": pooled["opening_pairs"]["accuracy"],
        "candidate_pair_correct": pooled["opening_pairs"]["correct"],
        "candidate_pair_count": pooled["opening_pairs"]["pair_count"],
        "superior_recognized": pooled["opening_superior_recognized"],
        "superior_total": pooled["opening_superior_total"],
        "superior_recognition_rate": (
            pooled["opening_superior_recognized"]
            / pooled["opening_superior_total"]
        ),
        "superior_by_margin": pooled["opening_superior_by_margin"],
        "judge_bands": band_rows,
        "panel_level_judge_score_spearman": spearman(
            [row["judge_score"] for row in panel_rows],
            [row["recognition_rate"] for row in panel_rows],
        ),
        "adaptive_improved_count": pooled["adaptive_improved_count"],
        "adaptive_unchanged_count": pooled["adaptive_unchanged_count"],
        "adaptive_worsened_count": pooled["adaptive_worsened_count"],
        "reported_cost_usd": pooled["reported_cost_usd"],
    }


def kendall_order(left: Sequence[str], right: Sequence[str]) -> float:
    if (
        len(left) != len(right)
        or set(left) != set(right)
        or len(set(left)) != len(left)
        or len(left) < 2
    ):
        raise ValueError("rankings must contain the same unique items")
    positions = {item: index for index, item in enumerate(right)}
    concordant = 0
    discordant = 0
    for index, first in enumerate(left):
        for second in left[index + 1 :]:
            if positions[first] < positions[second]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (concordant + discordant)


def render_oversight_frontier_svg(summary: Mapping[str, Any]) -> str:
    width, height = 1120, 610
    left, top, plot_width, plot_height = 105, 145, 900, 350
    labels = [row[0] for row in MARGIN_BINS]
    x_positions = [
        left + index * plot_width / (len(labels) - 1)
        for index in range(len(labels))
    ]
    colors = {
        "lower third": "#b94736",
        "middle third": "#9b6a18",
        "upper third": "#247665",
    }

    def y(value: float) -> float:
        return top + plot_height * (1 - (value - 0.2) / 0.8)

    grid = []
    for value in (0.2, 0.4, 0.6, 0.8, 1.0):
        y_value = y(value)
        grid.append(
            f"<line x1='{left}' y1='{y_value:.1f}' "
            f"x2='{left + plot_width}' y2='{y_value:.1f}' "
            "stroke='#dce2e5'/>"
            f"<text x='{left - 14}' y='{y_value + 4:.1f}' "
            "text-anchor='end' font-size='12' fill='#5e6a71'>"
            f"{value:.0%}</text>"
        )
    series = []
    for band in summary["judge_bands"]:
        color = colors[band["label"]]
        points = []
        for x_value, cell in zip(x_positions, band["cells"], strict=True):
            if cell["rate"] is None:
                continue
            y_value = y(float(cell["rate"]))
            points.append((x_value, y_value))
            if cell["wilson_95_low"] is not None:
                low = y(float(cell["wilson_95_low"]))
                high = y(float(cell["wilson_95_high"]))
                series.append(
                    f"<line x1='{x_value:.1f}' y1='{high:.1f}' "
                    f"x2='{x_value:.1f}' y2='{low:.1f}' "
                    f"stroke='{color}' stroke-width='1.5' opacity='.55'/>"
                )
            series.append(
                f"<circle cx='{x_value:.1f}' cy='{y_value:.1f}' r='6' "
                f"fill='{color}'><title>{escape(band['label'])}: "
                f"{cell['recognized']}/{cell['total']}</title></circle>"
            )
        if len(points) > 1:
            path = " ".join(
                f"{'M' if index == 0 else 'L'} {x_value:.1f} {y_value:.1f}"
                for index, (x_value, y_value) in enumerate(points)
            )
            series.insert(
                0,
                f"<path d='{path}' fill='none' stroke='{color}' "
                "stroke-width='3'/>",
            )
    x_labels = "".join(
        f"<text x='{x_value:.1f}' y='{top + plot_height + 34}' "
        "text-anchor='middle' font-size='13' fill='#182026'>"
        f"{escape(label)} points</text>"
        for x_value, label in zip(x_positions, labels, strict=True)
    )
    legend = "".join(
        f"<circle cx='{left + 175 * index}' cy='555' r='5' fill='{colors[band]}'/>"
        f"<text x='{left + 175 * index + 13}' y='559' "
        f"font-size='13' fill='#182026'>{escape(band)}</text>"
        for index, band in enumerate(JUDGE_BANDS)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg"
viewBox="0 0 {width} {height}" role="img"
aria-label="Recognition of stronger candidates by judge capability and margin">
<rect width="100%" height="100%" fill="#fff"/>
<text x="55" y="52" font-family="Georgia,serif" font-size="30"
fill="#182026">Can judges recognize models above their own level?</text>
<text x="55" y="82" font-family="system-ui,sans-serif" font-size="14"
fill="#5e6a71">Opening five-probe judgment; points show observed rates and Wilson intervals.</text>
{''.join(grid)}
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}"
y2="{top + plot_height}" stroke="#182026" stroke-width="1.5"/>
{x_labels}{''.join(series)}
<text x="24" y="{top + plot_height / 2}" font-family="system-ui,sans-serif"
font-size="13" fill="#182026" text-anchor="middle"
transform="rotate(-90 24 {top + plot_height / 2})">Stronger candidate ranked above self</text>
{legend}
</svg>"""


def _catalog_judge_row(run: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = next(
        row
        for row in run["probe_budget_results"]
        if int(row["probe_count"]) == 5
    )
    participants = {
        row["id"]: row["provider_model_id"] for row in run["participants"]
    }
    scores = {
        participants[participant_id]: float(score)
        for participant_id, score in run["prior_participant_scores"].items()
    }
    reported = {
        participants[participant_id]
        for participant_id in run["prior_reported_score_participants"]
    }
    ranking_models = [
        participants[participant_id] for participant_id in checkpoint["ranking"]
    ]
    external_order = sorted(scores, key=lambda model: (-scores[model], model))
    external_rank = {
        model: index for index, model in enumerate(external_order, start=1)
    }
    rank_table = [
        {
            "judged_rank": judged_rank,
            "provider_model_id": model,
            "external_score": scores[model],
            "external_rank": external_rank[model],
            "rank_error": judged_rank - external_rank[model],
            "external_score_is_estimated": model not in reported,
        }
        for judged_rank, model in enumerate(ranking_models, start=1)
    ]
    judge_model = run["judges"][0]["provider_model_id"]
    return {
        "judge_model": judge_model,
        "judge_name": _short_judge_name(judge_model),
        "run_dir": run["run_dir"],
        "candidate_count": len(ranking_models),
        "reported_score_candidate_count": len(reported),
        "pairwise_accuracy": float(checkpoint["pairwise_accuracy"]),
        "kendall_tau": float(checkpoint["kendall_tau"]),
        "spearman_rho": float(checkpoint["spearman_rho"]),
        "confidence": checkpoint.get("confidence"),
        "ranking_models": ranking_models,
        "pairwise_accuracy_by_score_gap": checkpoint[
            "pairwise_accuracy_by_score_gap"
        ],
        "rank_table": rank_table,
        "largest_reported_rank_errors": sorted(
            (
                row
                for row in rank_table
                if not row["external_score_is_estimated"]
            ),
            key=lambda row: (-abs(row["rank_error"]), row["judged_rank"]),
        )[:5],
    }


def _load_oversight_conditions(
    synthesis: Mapping[str, Any],
) -> list[dict[str, Any]]:
    conditions = []
    for path in synthesis["source_results"]:
        study = _load_json(Path(path))
        conditions.extend(
            {**condition, "study": study["study"]}
            for condition in study["conditions"]
        )
    return conditions


def _judge_score_bands(
    observations: Sequence[Mapping[str, Any]],
) -> dict[float, str]:
    scores = sorted({float(row["judge_score"]) for row in observations})
    return {
        score: JUDGE_BANDS[min(2, index * 3 // len(scores))]
        for index, score in enumerate(scores)
    }


def _margin_cells(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cells = []
    for label, lower, upper in MARGIN_BINS:
        selected = [
            row
            for row in observations
            if float(row["margin"]) >= lower
            and (upper is None or float(row["margin"]) < upper)
        ]
        recognized = sum(bool(row["recognized"]) for row in selected)
        low, high = wilson_interval(recognized, len(selected))
        cells.append(
            {
                "label": label,
                "recognized": recognized,
                "total": len(selected),
                "rate": recognized / len(selected) if selected else None,
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )
    return cells


def _subten_margin_weights(
    observations: Sequence[Mapping[str, Any]],
) -> list[float]:
    counts = []
    for _, lower, upper in MARGIN_BINS[:3]:
        counts.append(
            sum(
                float(row["margin"]) >= lower
                and float(row["margin"]) < float(upper)
                for row in observations
            )
        )
    total = sum(counts)
    return [count / total for count in counts]


def _standardized_rate(
    cells: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
) -> float | None:
    rates = [cell["rate"] for cell in cells[:3]]
    if any(rate is None for rate in rates):
        return None
    return sum(
        weight * float(rate)
        for weight, rate in zip(weights, rates, strict=True)
    )


def _panel_bootstrap_interval(
    observations: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[(str(row["study"]), str(row["condition_id"]))].append(row)
    panels = list(grouped.values())
    if not panels:
        return None, None
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sample = [
            row
            for _ in panels
            for row in rng.choice(panels)
        ]
        estimate = _standardized_rate(_margin_cells(sample), weights)
        if estimate is not None:
            estimates.append(estimate)
    if not estimates:
        return None, None
    estimates.sort()
    return (
        estimates[int(0.025 * len(estimates))],
        estimates[max(0, int(0.975 * len(estimates)) - 1)],
    )


def _short_judge_name(model: str) -> str:
    if "sol" in model.lower():
        return "Sol"
    if "fable" in model.lower():
        return "Fable"
    return model.rsplit("/", 1)[-1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
