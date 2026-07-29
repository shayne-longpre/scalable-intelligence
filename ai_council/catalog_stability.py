from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai_council.rankings import kendall_tau_between


COLORS = {"Sol": "#2364aa", "Fable": "#c44536"}


def build_catalog_stability_report(
    *,
    baseline_summary_path: str | Path,
    replication_summary_path: str | Path,
    output_dir: str | Path,
    published_json_path: str | Path | None = None,
) -> dict[str, Any]:
    baseline = _load_json(Path(baseline_summary_path))
    replication = _load_json(Path(replication_summary_path))
    baseline_runs = {_judge_key(run): run for run in baseline["runs"]}
    replication_runs = {_judge_key(run): run for run in replication["runs"]}
    if baseline_runs.keys() != replication_runs.keys():
        raise ValueError("baseline and replication judges must match")
    comparisons = [
        compare_judge_runs(baseline_runs[key], replication_runs[key])
        for key in sorted(baseline_runs)
    ]
    replication_availability = [
        _answer_availability(run) for run in replication_runs.values()
    ]
    summary = {
        "schema_version": "catalog-ladder-stability-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_summary": str(baseline_summary_path),
        "replication_summary": str(replication_summary_path),
        "judges": comparisons,
        "mean_rank_replication_tau": _mean(
            row["rank_replication_tau"] for row in comparisons
        ),
        "mean_top5_overlap": _mean(row["top5_overlap"] for row in comparisons),
        "baseline_reported_cost_usd": sum(
            float(run.get("reported_cost_usd") or 0)
            for run in baseline["runs"]
        ),
        "replication_reported_cost_usd": sum(
            float(run.get("reported_cost_usd") or 0)
            for run in replication["runs"]
        ),
        "runtime_sensitive_replication_count": sum(
            bool(run.get("repair", {}).get("runtime_sensitive"))
            for run in replication["runs"]
        ),
        "replication_answer_availability": {
            key: sum(row[key] for row in replication_availability)
            for key in (
                "opening_expected",
                "opening_unavailable",
                "adaptive_expected",
                "adaptive_unavailable",
            )
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "evidence_trajectory": output_dir / "evidence-trajectory.svg",
        "rank_replication": output_dir / "rank-replication.svg",
        "gap_stability": output_dir / "gap-stability.svg",
    }
    figures["evidence_trajectory"].write_text(
        evidence_trajectory_svg(comparisons), encoding="utf-8"
    )
    figures["rank_replication"].write_text(
        rank_replication_svg(comparisons), encoding="utf-8"
    )
    figures["gap_stability"].write_text(
        gap_stability_svg(comparisons), encoding="utf-8"
    )
    summary["figures"] = {name: str(path) for name, path in figures.items()}
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report_card.html").write_text(
        render_catalog_stability(summary, figures), encoding="utf-8"
    )
    if published_json_path:
        published_path = Path(published_json_path)
        published_path.parent.mkdir(parents=True, exist_ok=True)
        published_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def compare_judge_runs(
    baseline: Mapping[str, Any],
    replication: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_route = _judge_key(baseline)
    replication_route = _judge_key(replication)
    if baseline_route != replication_route:
        raise ValueError("judge routes differ")
    baseline_models = _participant_models(baseline)
    replication_models = _participant_models(replication)
    baseline_opening = baseline["probe_budget_results"][0]
    replication_opening = replication["probe_budget_results"][0]
    baseline_ranking = _model_ranking(
        baseline_opening["ranking"], baseline_models
    )
    replication_ranking = _model_ranking(
        replication_opening["ranking"], replication_models
    )
    shared = set(baseline_ranking) & set(replication_ranking)
    reported_baseline = {
        baseline_models[participant_id]
        for participant_id in baseline.get(
            "prior_reported_score_participants", []
        )
        if participant_id in baseline_models
    }
    reported_replication = {
        replication_models[participant_id]
        for participant_id in replication.get(
            "prior_reported_score_participants", []
        )
        if participant_id in replication_models
    }
    reported = shared & reported_baseline & reported_replication
    baseline_reported = [model for model in baseline_ranking if model in reported]
    replication_reported = [
        model for model in replication_ranking if model in reported
    ]
    baseline_positions = {
        model: index for index, model in enumerate(baseline_reported, 1)
    }
    replication_positions = {
        model: index for index, model in enumerate(replication_reported, 1)
    }
    return {
        "judge_model": baseline_route,
        "judge_name": _judge_name(baseline_route),
        "baseline_run": baseline["run_dir"],
        "replication_run": replication["run_dir"],
        "replication_repair": replication.get("repair", {}),
        "reported_candidate_count": len(reported),
        "rank_replication_tau": kendall_tau_between(
            baseline_reported, replication_reported
        ),
        "top5_overlap": (
            len(set(baseline_reported[:5]) & set(replication_reported[:5]))
            / min(5, len(reported))
            if reported
            else 0.0
        ),
        "baseline_opening_probe_count": int(baseline_opening["probe_count"]),
        "replication_opening_probe_count": int(
            replication_opening["probe_count"]
        ),
        "baseline_opening_pairwise_accuracy": baseline_opening[
            "pairwise_accuracy"
        ],
        "replication_opening_pairwise_accuracy": replication_opening[
            "pairwise_accuracy"
        ],
        "opening_pairwise_accuracy_delta": (
            replication_opening["pairwise_accuracy"]
            - baseline_opening["pairwise_accuracy"]
        ),
        "baseline_checkpoints": _checkpoint_rows(baseline),
        "replication_checkpoints": _checkpoint_rows(replication),
        "baseline_opening_gap_accuracy": baseline_opening.get(
            "pairwise_accuracy_by_score_gap", []
        ),
        "replication_opening_gap_accuracy": replication_opening.get(
            "pairwise_accuracy_by_score_gap", []
        ),
        "rank_points": [
            {
                "model": model,
                "baseline_rank": baseline_positions[model],
                "replication_rank": replication_positions[model],
            }
            for model in baseline_reported
        ],
    }


def render_catalog_stability(
    summary: Mapping[str, Any],
    figures: Mapping[str, Path],
) -> str:
    rows = "".join(
        f"<tr><td>{escape(row['judge_name'])}</td>"
        f"<td class='num'>{row['rank_replication_tau']:.3f}</td>"
        f"<td class='num'>{row['top5_overlap']:.0%}</td>"
        f"<td class='num'>{row['baseline_opening_pairwise_accuracy']:.1%}</td>"
        f"<td class='num'>{row['replication_opening_pairwise_accuracy']:.1%}</td>"
        f"<td class='num'>{row['opening_pairwise_accuracy_delta']:+.1%}</td></tr>"
        for row in summary["judges"]
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Catalog ladder stability</title><style>
:root{{--ink:#182026;--muted:#65727a;--line:#dce2e5}}*{{box-sizing:border-box}}
body{{margin:0;color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif}}
main{{max-width:1160px;margin:auto;padding:48px 28px 70px}}h1{{font:700 44px/1.08 Georgia,serif;
margin:0 0 14px}}h2{{font:700 27px/1.2 Georgia,serif;margin-top:46px}}
.lede,figcaption,.note{{color:var(--muted)}}.metrics{{display:grid;
grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);
border:1px solid var(--line);margin:28px 0}}.metric{{background:white;padding:18px}}
.metric strong{{font-size:27px;display:block}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}figure{{margin:22px 0}}
img{{width:100%;height:auto}}.table-wrap{{overflow:auto;border-top:2px solid var(--ink)}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px 10px;
border-bottom:1px solid var(--line);
text-align:left}}th{{font-size:12px;color:var(--muted)}}.num{{text-align:right;
font-variant-numeric:tabular-nums}}@media(max-width:760px){{main{{padding:30px 16px}}
.grid,.metrics{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>Does the catalog ladder persist with a larger opening battery?</h1>
<p class="lede">The same two judges evaluated the same anonymous 50-model roster.
The comparison uses each run's opening judgment, before adaptive follow-ups, and
the same external model index.</p>
<section class="metrics">
<div class="metric"><strong>{summary['mean_rank_replication_tau']:.2f}</strong>
<span>mean old-new Kendall tau</span></div>
<div class="metric"><strong>{summary['mean_top5_overlap']:.0%}</strong>
<span>mean top-five overlap</span></div>
<div class="metric"><strong>${summary['replication_reported_cost_usd']:.2f}</strong>
<span>new provider-reported spend</span></div></section>
<h2>Outcome stability</h2><div class="table-wrap"><table><thead><tr><th>Judge</th>
<th class="num">Old-new tau</th><th class="num">Top-5 overlap</th>
<th class="num">Old opening accuracy</th><th class="num">New opening accuracy</th>
<th class="num">Change</th></tr></thead><tbody>{rows}</tbody></table></div>
<figure><img src="{escape(figures['evidence_trajectory'].name)}" alt="Accuracy by probe count">
<figcaption>Accuracy as each judge accumulates evidence. Dashed lines are the
earlier run; solid lines are the larger opening battery.</figcaption></figure>
<div class="grid"><figure><img src="{escape(figures['rank_replication'].name)}"
alt="Old rank versus new rank"><figcaption>Each point is one model with a directly
reported external score. The diagonal is exact rank replication.</figcaption></figure>
<figure><img src="{escape(figures['gap_stability'].name)}"
alt="Accuracy by capability gap"><figcaption>Final discrimination by external
capability gap. Large gaps should remain easier in both runs.</figcaption></figure></div>
<p class="note">{summary['runtime_sensitive_replication_count']} replication
run(s) used recovery parameters or a recorded model-specific override. Such
answers remain visible as runtime sensitivities rather than silent replacements.</p>
</main></body></html>"""


def evidence_trajectory_svg(comparisons: Sequence[Mapping[str, Any]]) -> str:
    width, height = 920, 470
    left, top, plot_width, plot_height = 75, 55, 770, 330
    parts = _chart_grid(left, top, plot_width, plot_height, 0.5, 0.9)
    parts.extend(_series_legend(505, 24))
    for row in comparisons:
        color = COLORS.get(row["judge_name"], "#39464d")
        for key, dashed in (
            ("baseline_checkpoints", True),
            ("replication_checkpoints", False),
        ):
            points = [
                (
                    _scale(item["probe_count"], 4, 7, left, left + plot_width),
                    _scale(
                        item["pairwise_accuracy"],
                        0.5,
                        0.9,
                        top + plot_height,
                        top,
                    ),
                )
                for item in row[key]
            ]
            dash = " stroke-dasharray='7 6'" if dashed else ""
            parts.append(
                f"<polyline points='{' '.join(f'{x:.1f},{y:.1f}' for x,y in points)}' "
                f"fill='none' stroke='{color}' stroke-width='3'{dash}/>"
            )
            for x, y in points:
                parts.append(
                    f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}'/>"
                )
    for tick in range(4, 8):
        x = _scale(tick, 4, 7, left, left + plot_width)
        parts.append(
            f"<text x='{x:.1f}' y='{top+plot_height+25}' text-anchor='middle' "
            f"font-size='11' fill='#65727a'>{tick}</text>"
        )
    parts.append(
        f"<text x='{left+plot_width/2:.1f}' y='{height-18}' text-anchor='middle' "
        f"font-size='12' fill='#65727a'>Cumulative probes</text>"
    )
    return _svg(width, height, "Pairwise accuracy by cumulative probes", parts)


def rank_replication_svg(comparisons: Sequence[Mapping[str, Any]]) -> str:
    width, height = 760, 390
    panel_width, panel_height, top = 280, 275, 58
    parts = []
    for panel, row in enumerate(comparisons):
        left = 55 + panel * 365
        count = row["reported_candidate_count"]
        parts.append(
            f"<text x='{left}' y='28' font-size='15' font-weight='700' "
            f"fill='#182026'>{escape(row['judge_name'])}</text>"
        )
        parts.append(
            f"<line x1='{left}' y1='{top}' x2='{left+panel_width}' "
            f"y2='{top+panel_height}' stroke='#9aa5aa' stroke-dasharray='5 5'/>"
        )
        for point in row["rank_points"]:
            x = _scale(point["baseline_rank"], 1, count, left, left + panel_width)
            y = _scale(
                point["replication_rank"],
                1,
                count,
                top,
                top + panel_height,
            )
            parts.append(
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.4' "
                f"fill='{COLORS.get(row['judge_name'], '#39464d')}' opacity='.75'>"
                f"<title>{escape(point['model'])}: old {point['baseline_rank']}, "
                f"new {point['replication_rank']}</title></circle>"
            )
    return _svg(width, height, "Old rank versus replication rank", parts)


def gap_stability_svg(comparisons: Sequence[Mapping[str, Any]]) -> str:
    width, height = 760, 390
    left, top, plot_width, plot_height = 58, 45, 650, 270
    parts = _chart_grid(left, top, plot_width, plot_height, 0.45, 1.0)
    parts.extend(_series_legend(365, 22))
    labels = [
        row["label"]
        for row in comparisons[0]["baseline_opening_gap_accuracy"]
    ]
    for row in comparisons:
        color = COLORS.get(row["judge_name"], "#39464d")
        for key, dashed in (
            ("baseline_opening_gap_accuracy", True),
            ("replication_opening_gap_accuracy", False),
        ):
            points = []
            for index, item in enumerate(row[key]):
                x = left + (index + 0.5) * plot_width / len(labels)
                y = _scale(
                    item["accuracy"], 0.45, 1.0, top + plot_height, top
                )
                points.append((x, y))
            dash = " stroke-dasharray='7 6'" if dashed else ""
            parts.append(
                f"<polyline points='{' '.join(f'{x:.1f},{y:.1f}' for x,y in points)}' "
                f"fill='none' stroke='{color}' stroke-width='3'{dash}/>"
            )
            for x, y in points:
                parts.append(
                    f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}'/>"
                )
    for index, label in enumerate(labels):
        x = left + (index + 0.5) * plot_width / len(labels)
        parts.append(
            f"<text x='{x:.1f}' y='{top+plot_height+25}' text-anchor='middle' "
            f"font-size='11' fill='#65727a'>{escape(label)}</text>"
        )
    return _svg(width, height, "Accuracy by external capability gap", parts)


def _chart_grid(
    left: float,
    top: float,
    width: float,
    height: float,
    low: float,
    high: float,
) -> list[str]:
    parts = []
    for index in range(5):
        value = low + (high - low) * index / 4
        y = _scale(value, low, high, top + height, top)
        parts.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{left+width}' y2='{y:.1f}' "
            f"stroke='#dce2e5'/><text x='{left-8}' y='{y+4:.1f}' text-anchor='end' "
            f"font-size='10' fill='#65727a'>{value:.0%}</text>"
        )
    return parts


def _series_legend(x: float, y: float) -> list[str]:
    return [
        f"<line x1='{x}' y1='{y}' x2='{x+24}' y2='{y}' stroke='{COLORS['Sol']}' "
        f"stroke-width='3'/><text x='{x+30}' y='{y+4}' font-size='10' "
        f"fill='#65727a'>Sol</text>",
        f"<line x1='{x+75}' y1='{y}' x2='{x+99}' y2='{y}' "
        f"stroke='{COLORS['Fable']}' stroke-width='3'/>"
        f"<text x='{x+105}' y='{y+4}' font-size='10' fill='#65727a'>Fable</text>",
        f"<line x1='{x+175}' y1='{y}' x2='{x+199}' y2='{y}' stroke='#65727a' "
        f"stroke-width='3' stroke-dasharray='7 6'/>"
        f"<text x='{x+205}' y='{y+4}' font-size='10' fill='#65727a'>earlier</text>",
        f"<line x1='{x+270}' y1='{y}' x2='{x+294}' y2='{y}' stroke='#65727a' "
        f"stroke-width='3'/><text x='{x+300}' y='{y+4}' font-size='10' "
        f"fill='#65727a'>five-opening-probe</text>",
    ]


def _svg(width: int, height: int, label: str, parts: Sequence[str]) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' "
        f"role='img' aria-label='{escape(label)}'>{''.join(parts)}</svg>"
    )


def _checkpoint_rows(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "probe_count": int(row["probe_count"]),
            "pairwise_accuracy": float(row["pairwise_accuracy"]),
            "kendall_tau": float(row["kendall_tau"]),
        }
        for row in run["probe_budget_results"]
    ]


def _participant_models(run: Mapping[str, Any]) -> dict[str, str]:
    return {
        row["id"]: row["provider_model_id"] for row in run["participants"]
    }


def _answer_availability(run: Mapping[str, Any]) -> dict[str, int]:
    rows = _jsonl(Path(run["run_dir"]) / "transcript.jsonl")
    answers = [
        row
        for row in rows
        if (row.get("metadata") or {}).get("interaction_role") == "answer"
    ]
    opening = [row for row in answers if int(row.get("round_index") or 0) == 1]
    adaptive = [row for row in answers if int(row.get("round_index") or 0) > 1]
    return {
        "opening_expected": len(opening),
        "opening_unavailable": sum(
            (row.get("metadata") or {}).get("answer_unavailable") is True
            for row in opening
        ),
        "adaptive_expected": len(adaptive),
        "adaptive_unavailable": sum(
            (row.get("metadata") or {}).get("answer_unavailable") is True
            for row in adaptive
        ),
    }


def _model_ranking(
    ranking: Sequence[str],
    participant_models: Mapping[str, str],
) -> list[str]:
    return [
        participant_models[participant_id]
        for participant_id in ranking
        if participant_id in participant_models
    ]


def _judge_key(run: Mapping[str, Any]) -> str:
    return run["judges"][0]["provider_model_id"]


def _judge_name(model_id: str) -> str:
    lowered = model_id.lower()
    if "sol" in lowered:
        return "Sol"
    if "fable" in lowered:
        return "Fable"
    return model_id.rsplit("/", 1)[-1]


def _scale(
    value: float,
    low: float,
    high: float,
    output_low: float,
    output_high: float,
) -> float:
    return output_low + (value - low) / (high - low) * (
        output_high - output_low
    )


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
