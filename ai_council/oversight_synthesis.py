from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai_council.oversight_replication import combine_accuracy, self_relative_summary


MARGIN_BINS = (
    ("<2", 0.0, 2.0),
    ("2-5", 2.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10+", 10.0, None),
)


def build_frontier_synthesis(
    *,
    result_paths: Sequence[str | Path],
    output_dir: str | Path,
    published_json_path: str | Path | None = None,
) -> dict[str, Any]:
    if len(result_paths) < 2:
        raise ValueError("frontier synthesis requires at least two study results")
    studies = [_load_json(path) for path in result_paths]
    conditions = [
        {**condition, "study": study["study"]}
        for study in studies
        for condition in study["conditions"]
    ]
    opening_superior = superior_observations(conditions, stage="opening")
    final_superior = superior_observations(conditions, stage="final")
    summary = {
        "schema_version": "oversight-frontier-synthesis-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_results": [str(path) for path in result_paths],
        "studies": [_study_summary(study) for study in studies],
        "pooled": {
            "condition_count": len(conditions),
            "judge_count": len({row["judge_model"] for row in conditions}),
            "opening_pairs": combine_accuracy(
                [row["opening"]["pairs"]["overall"] for row in conditions]
            ),
            "final_pairs": combine_accuracy(
                [row["final"]["pairs"]["overall"] for row in conditions]
            ),
            "opening_superior_recognized": sum(
                row["recognized"] for row in opening_superior
            ),
            "opening_superior_total": len(opening_superior),
            "opening_superior_by_margin": summarize_superior_margins(
                opening_superior
            ),
            "opening_superior_by_judge_band": summarize_judge_margin_matrix(
                opening_superior
            ),
            "superior_recognized": sum(
                row["recognized"] for row in final_superior
            ),
            "superior_total": len(final_superior),
            "superior_by_margin": summarize_superior_margins(final_superior),
            "self_relative": self_relative_summary(conditions),
            "adaptive_improved_count": sum(
                row["adaptive_delta_pair_accuracy"] > 0 for row in conditions
            ),
            "adaptive_unchanged_count": sum(
                row["adaptive_delta_pair_accuracy"] == 0 for row in conditions
            ),
            "adaptive_worsened_count": sum(
                row["adaptive_delta_pair_accuracy"] < 0 for row in conditions
            ),
            "model_calls": sum(row["model_calls"] for row in conditions),
            "reported_cost_usd": sum(
                row["reported_cost_usd"] for row in conditions
            ),
            "unavailable_answer_count": sum(
                row["unavailable_answer_count"] for row in conditions
            ),
            "candidate_answer_count": sum(
                probe["target_count"]
                for row in conditions
                for probe in row["probes"]
            ),
            "structured_repair_count": sum(
                row["structured_repair_count"] for row in conditions
            ),
            "visible_text_retry_count": sum(
                row["visible_text_retry_count"] for row in conditions
            ),
        },
        "judges": summarize_judges(
            conditions,
            final_superior,
            opening_superior=opening_superior,
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report_card.html").write_text(
        render_frontier_synthesis(summary, repo_prefix="../../.."),
        encoding="utf-8",
    )
    if published_json_path:
        published_path = Path(published_json_path)
        published_path.parent.mkdir(parents=True, exist_ok=True)
        published_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def superior_observations(
    conditions: Iterable[Mapping[str, Any]],
    *,
    stage: str = "final",
) -> list[dict[str, Any]]:
    if stage not in {"opening", "final"}:
        raise ValueError("superior observation stage must be opening or final")
    observations = []
    for condition in conditions:
        ranking = condition[stage]["ranking"]
        positions = {
            participant_id: index for index, participant_id in enumerate(ranking)
        }
        self_participant = condition["self_participant"]
        judge_score = float(condition["judge_external_score"])
        for participant_id, raw_score in condition["participant_scores"].items():
            score = float(raw_score)
            if score <= judge_score:
                continue
            observations.append(
                {
                    "study": condition["study"],
                    "condition_id": condition["id"],
                    "judge_model": condition["judge_model"],
                    "judge_short_name": condition["judge_short_name"],
                    "judge_score": judge_score,
                    "candidate_model": condition["participant_models"][participant_id],
                    "candidate_score": score,
                    "margin": score - judge_score,
                    "recognized": positions[participant_id] < positions[self_participant],
                    "stage": stage,
                }
            )
    return observations


def summarize_superior_margins(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for label, lower, upper in MARGIN_BINS:
        selected = [
            row
            for row in observations
            if row["margin"] >= lower
            and (upper is None or row["margin"] < upper)
        ]
        recognized = sum(bool(row["recognized"]) for row in selected)
        low, high = wilson_interval(recognized, len(selected))
        rows.append(
            {
                "label": label,
                "recognized": recognized,
                "total": len(selected),
                "rate": recognized / len(selected) if selected else None,
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )
    return rows


def summarize_judge_margin_matrix(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    judge_scores = sorted({float(row["judge_score"]) for row in observations})
    if not judge_scores:
        return []
    score_bands: dict[float, str] = {}
    band_labels = ("lower third", "middle third", "upper third")
    for index, score in enumerate(judge_scores):
        band_index = min(2, index * 3 // len(judge_scores))
        score_bands[score] = band_labels[band_index]

    rows = []
    for band in band_labels:
        band_scores = [
            score for score, label in score_bands.items() if label == band
        ]
        cells = []
        for label, lower, upper in MARGIN_BINS:
            selected = [
                row
                for row in observations
                if score_bands[float(row["judge_score"])] == band
                and float(row["margin"]) >= lower
                and (upper is None or float(row["margin"]) < upper)
            ]
            recognized = sum(bool(row["recognized"]) for row in selected)
            cells.append(
                {
                    "label": label,
                    "recognized": recognized,
                    "total": len(selected),
                    "rate": recognized / len(selected) if selected else None,
                }
            )
        rows.append(
            {
                "label": band,
                "judge_score_min": min(band_scores) if band_scores else None,
                "judge_score_max": max(band_scores) if band_scores else None,
                "cells": cells,
            }
        )
    return rows


def summarize_judges(
    conditions: Sequence[Mapping[str, Any]],
    superior: Sequence[Mapping[str, Any]],
    *,
    opening_superior: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    superior_by_judge: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    opening_by_judge: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for condition in conditions:
        grouped[condition["judge_model"]].append(condition)
    for observation in superior:
        superior_by_judge[observation["judge_model"]].append(observation)
    for observation in opening_superior or []:
        opening_by_judge[observation["judge_model"]].append(observation)

    rows = []
    for judge_model, judge_conditions in grouped.items():
        superior_rows = superior_by_judge[judge_model]
        opening_rows = opening_by_judge[judge_model]
        opening_pair_metrics = [
            condition["opening"]["pairs"]["overall"]
            for condition in judge_conditions
        ]
        final_pair_metrics = [
            condition["final"]["pairs"]["overall"] for condition in judge_conditions
        ]
        opening_above_metrics = [
            next(
                row
                for row in condition["opening"]["pairs"]["by_relative_position"]
                if row["label"] == "both above"
            )
            for condition in judge_conditions
        ]
        final_above_metrics = [
            next(
                row
                for row in condition["final"]["pairs"]["by_relative_position"]
                if row["label"] == "both above"
            )
            for condition in judge_conditions
        ]
        rows.append(
            {
                "judge_model": judge_model,
                "judge_short_name": judge_conditions[0]["judge_short_name"],
                "judge_external_score": judge_conditions[0][
                    "judge_external_score"
                ],
                "condition_count": len(judge_conditions),
                "superior_recognized": sum(
                    bool(row["recognized"]) for row in superior_rows
                ),
                "superior_total": len(superior_rows),
                "opening_superior_recognized": sum(
                    bool(row["recognized"]) for row in opening_rows
                ),
                "opening_superior_total": len(opening_rows),
                "unique_superior_models": len(
                    {row["candidate_model"] for row in superior_rows}
                ),
                "opening_pairs": combine_accuracy(opening_pair_metrics),
                "final_pairs": combine_accuracy(final_pair_metrics),
                "opening_both_above_pairs": combine_accuracy(
                    opening_above_metrics
                ),
                "final_both_above_pairs": combine_accuracy(
                    final_above_metrics
                ),
                "adaptive_improved_count": sum(
                    row["adaptive_delta_pair_accuracy"] > 0
                    for row in judge_conditions
                ),
                "unavailable_answer_count": sum(
                    row["unavailable_answer_count"] for row in judge_conditions
                ),
                "candidate_answer_count": sum(
                    probe["target_count"]
                    for row in judge_conditions
                    for probe in row["probes"]
                ),
            }
        )
    return sorted(rows, key=lambda row: -row["judge_external_score"])


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.96,
) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - spread), min(1.0, center + spread)


def render_frontier_synthesis(
    summary: Mapping[str, Any],
    *,
    repo_prefix: str = "../..",
) -> str:
    pooled = summary["pooled"]
    judge_rows = "".join(
        "<tr>"
        f"<th>{escape(row['judge_short_name'])}</th>"
        f"<td class='num'>{row['judge_external_score']:.1f}</td>"
        f"<td class='num'>{row['condition_count']}</td>"
        f"<td class='num'>{row['opening_superior_recognized']}/"
        f"{row['opening_superior_total']}</td>"
        f"<td class='num'>{row['superior_recognized']}/{row['superior_total']}</td>"
        f"<td class='num'>{row['unique_superior_models']}</td>"
        f"<td class='num'>{_pct(row['opening_pairs']['accuracy'])}</td>"
        f"<td class='num'>{_pct(row['final_pairs']['accuracy'])}</td>"
        f"<td class='num'>{_pct(row['opening_both_above_pairs']['accuracy'])}</td>"
        f"<td class='num'>{row['adaptive_improved_count']}/{row['condition_count']}</td>"
        f"<td class='num'>{row['unavailable_answer_count']}/{row['candidate_answer_count']}</td>"
        "</tr>"
        for row in summary["judges"]
    )
    final_margins = {
        row["label"]: row for row in pooled["superior_by_margin"]
    }
    margin_rows = "".join(
        "<tr>"
        f"<th>{escape(row['label'])}</th>"
        f"<td class='num'>{row['recognized']}/{row['total']}</td>"
        f"<td class='num'>{_pct(row['rate'])}</td>"
        f"<td class='num'>{_pct(final_margins[row['label']]['rate'])}</td>"
        f"<td class='num'>{_interval(row)}</td>"
        "</tr>"
        for row in pooled["opening_superior_by_margin"]
    )
    study_rows = "".join(
        "<tr>"
        f"<th>{escape(row['study'])}</th>"
        f"<td class='num'>{row['condition_count']}</td>"
        f"<td class='num'>{row['candidate_count_text']}</td>"
        f"<td class='num'>{row['opening_superior_recognized']}/"
        f"{row['opening_superior_total']}</td>"
        f"<td class='num'>{row['superior_recognized']}/{row['superior_total']}</td>"
        f"<td class='num'>{_pct(row['final_pair_accuracy'])}</td>"
        f"<td class='num'>{row['unavailable_answer_count']}/{row['candidate_answer_count']}</td>"
        f"<td class='num'>${row['reported_cost_usd']:.2f}</td>"
        "</tr>"
        for row in summary["studies"]
    )
    return f"""<!doctype html>
<html lang="en"><head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Oversight frontier synthesis</title>
  <style>
    :root {{ --ink:#172026; --muted:#637078; --line:#d9dfe2; --soft:#f4f6f6;
      --blue:#276b9c; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#fff; color:var(--ink);
      font:15px/1.5 system-ui,sans-serif; }}
    main {{ max-width:1160px; margin:0 auto; padding:48px 26px 72px; }}
    h1 {{ margin:0 0 10px; max-width:900px; font:700 42px/1.08 Georgia,serif; }}
    h2 {{ margin:42px 0 12px; font:700 25px/1.2 Georgia,serif; }}
    p {{ max-width:860px; color:var(--muted); }}
    a {{ color:var(--blue); text-underline-offset:2px; }}
    .resources {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin:18px 0 4px;
      font-size:13px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
      border-block:1px solid var(--line); margin:28px 0; }}
    .metric {{ padding:18px 12px; }}
    .metric strong {{ display:block; font:700 28px/1.1 Georgia,serif; }}
    .metric span {{ color:var(--muted); font-size:13px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
    .grid > * {{ min-width:0; }}
    figure {{ margin:0; border-top:2px solid var(--ink); padding-top:10px; }}
    svg {{ width:100%; height:auto; display:block; background:var(--soft); }}
    figcaption,.small {{ color:var(--muted); font-size:13px; }}
    .table-wrap {{ max-width:100%; overflow:auto; border-top:2px solid var(--ink); }}
    table {{ width:100%; border-collapse:collapse; min-width:700px; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; }}
    thead th {{ color:var(--muted); font-size:12px; white-space:nowrap; }}
    .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
    @media (max-width:760px) {{
      main {{ padding:30px 16px 52px; }} h1 {{ font-size:34px; }}
      .metrics {{ grid-template-columns:1fr 1fr; }} .grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head><body><main>
  <h1>Can models recognize intelligence above their own?</h1>
  <p>{pooled['condition_count']} anonymous judge panels across
  {pooled['judge_count']} distinct judges. Every judge authored five common
  opening probes and one targeted follow-up. Catalog scores are an external,
  noisy reference rather than ground truth.</p>
  <nav class="resources" aria-label="Study resources">
    <a href="{repo_prefix}/docs/oversight_frontier_design.md">Protocol</a>
    <a href="{repo_prefix}/docs/site/taxonomy.html">Taxonomy</a>
    <a href="{repo_prefix}/docs/site/models.html">Model catalog</a>
    <a href="{repo_prefix}/docs/pilot_analysis_oversight_matched_20260727.md">Analysis</a>
    <a href="{repo_prefix}/data/oversight_frontier_synthesis_matched_results.json">Results JSON</a>
  </nav>
  <section class="metrics">
    <div class="metric"><strong>{_pct(pooled['opening_pairs']['accuracy'])}</strong>
      <span>all pair orderings after five probes</span></div>
    <div class="metric"><strong>{pooled['opening_superior_recognized']}/{pooled['opening_superior_total']}</strong>
      <span>stronger candidates above judge after five probes</span></div>
    <div class="metric"><strong>{pooled['adaptive_improved_count']}/{pooled['condition_count']}</strong>
      <span>panels improved by follow-up</span></div>
    <div class="metric"><strong>${pooled['reported_cost_usd']:.2f}</strong>
      <span>provider-reported study spend</span></div>
  </section>

  <div class="grid">
    <figure>{_margin_svg(pooled['opening_superior_by_margin'])}
      <figcaption>Primary five-probe recognition of candidates above the judge,
      grouped by external-score lead. Bars show Wilson 95% intervals.</figcaption>
    </figure>
    <div>
      <h2>Capability margin</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>Lead over judge</th><th class="num">Opening</th>
          <th class="num">Opening rate</th><th class="num">Final rate</th>
          <th class="num">Opening 95% interval</th></tr></thead>
        <tbody>{margin_rows}</tbody>
      </table></div>
      <p class="small">Bins are disjoint. Counts are candidate appearances, not
      independent draws; the interval is descriptive and does not model clustering
      by judge or candidate.</p>
    </div>
  </div>

  <h2>Judge capability and candidate margin</h2>
  <figure>{_judge_margin_svg(pooled['opening_superior_by_judge_band'])}
    <figcaption>Five-probe superior-recognition rates by judge-score third and
    candidate lead. Cells show recognized appearances over total appearances.
    Sparse cells are descriptive, not a fitted oversight threshold.</figcaption>
  </figure>

  <h2>Judge coverage</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Judge</th><th class="num">Score</th><th class="num">Panels</th>
      <th class="num">Opening recognized</th><th class="num">Final recognized</th>
      <th class="num">Unique stronger</th>
      <th class="num">Opening pairs</th><th class="num">Final pairs</th>
      <th class="num">Opening both above</th>
      <th class="num">Adaptive helped</th><th class="num">Missing answers</th></tr></thead>
    <tbody>{judge_rows}</tbody>
  </table></div>
  <p class="small">“Stronger recognized” asks whether each externally stronger
  candidate was ranked above the judge's anonymous self. “Both above” tests
  whether a judge could order two candidates that both exceeded it.</p>

  <h2>Study waves</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Wave</th><th class="num">Panels</th><th class="num">Candidates</th>
      <th class="num">Opening recognized</th><th class="num">Final recognized</th>
      <th class="num">All pairs</th>
      <th class="num">Missing answers</th><th class="num">Spend</th></tr></thead>
    <tbody>{study_rows}</tbody>
  </table></div>
  <p class="small">Across all waves, {pooled['unavailable_answer_count']} of
  {pooled['candidate_answer_count']} routed answers remained unavailable,
  {pooled['visible_text_retry_count']} visible-text retries were attempted, and
  {pooled['structured_repair_count']} private JSON outputs required bounded
  same-model repair. Missing answers remain visible to the judge as missing
  evidence, never as low-quality answers.</p>
</main></body></html>
"""


def _study_summary(study: Mapping[str, Any]) -> dict[str, Any]:
    candidate_counts = sorted(
        {int(row["candidate_count"]) for row in study["conditions"]}
    )
    return {
        "study": study["study"],
        "condition_count": len(study["conditions"]),
        "candidate_count_text": (
            str(candidate_counts[0])
            if len(candidate_counts) == 1
            else f"{candidate_counts[0]}-{candidate_counts[-1]}"
        ),
        "superior_recognized": study["aggregate"]["superior_recognized"],
        "superior_total": study["aggregate"]["superior_total"],
        "opening_superior_recognized": sum(
            row["opening"]["superior_recognized"]
            for row in study["conditions"]
        ),
        "opening_superior_total": sum(
            row["opening"]["superior_total"] for row in study["conditions"]
        ),
        "final_pair_accuracy": study["aggregate"]["final_pair_accuracy"],
        "unavailable_answer_count": study["aggregate"]["unavailable_answer_count"],
        "candidate_answer_count": sum(
            probe["target_count"]
            for condition in study["conditions"]
            for probe in condition["probes"]
        ),
        "reported_cost_usd": study["aggregate"]["reported_cost_usd"],
    }


def _margin_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    width, height = 520, 300
    left, right, top, bottom = 48, 20, 24, 48
    plot_width, plot_height = width - left - right, height - top - bottom

    def y(value: float) -> float:
        return top + (1 - value) * plot_height

    parts = []
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f"<line x1='{left}' y1='{y(tick):.1f}' x2='{width-right}' "
            f"y2='{y(tick):.1f}' stroke='#d9dfe2'/>"
            f"<text x='{left-7}' y='{y(tick)+4:.1f}' text-anchor='end' "
            f"font-size='11' fill='#637078'>{tick:.0%}</text>"
        )
    bar_width = plot_width / len(rows) * 0.52
    for index, row in enumerate(rows):
        x = left + (index + 0.5) * plot_width / len(rows)
        rate = row["rate"] or 0.0
        high = row["wilson_95_high"] or 0.0
        low = row["wilson_95_low"] or 0.0
        parts.extend(
            [
                f"<rect x='{x-bar_width/2:.1f}' y='{y(rate):.1f}' "
                f"width='{bar_width:.1f}' height='{y(0)-y(rate):.1f}' fill='#276b9c'/>",
                f"<line x1='{x:.1f}' y1='{y(high):.1f}' x2='{x:.1f}' "
                f"y2='{y(low):.1f}' stroke='#172026' stroke-width='2'/>",
                f"<line x1='{x-5:.1f}' y1='{y(high):.1f}' x2='{x+5:.1f}' "
                f"y2='{y(high):.1f}' stroke='#172026'/>",
                f"<line x1='{x-5:.1f}' y1='{y(low):.1f}' x2='{x+5:.1f}' "
                f"y2='{y(low):.1f}' stroke='#172026'/>",
                f"<text x='{x:.1f}' y='{height-18}' text-anchor='middle' "
                f"font-size='11' fill='#637078'>{escape(row['label'])}</text>",
                f"<text x='{x:.1f}' y='{max(top + 12, y(rate)-7):.1f}' "
                f"text-anchor='middle' font-size='11' fill='#172026'>"
                f"{row['recognized']}/{row['total']}</text>",
            ]
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='Recognition of stronger candidates by capability margin'>"
        f"{''.join(parts)}</svg>"
    )


def _judge_margin_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = [label for label, _, _ in MARGIN_BINS]
    cell_width, cell_height = 116, 62
    left, top = 132, 48
    width = left + len(columns) * cell_width + 20
    height = top + len(rows) * cell_height + 28
    parts = []
    for column, label in enumerate(columns):
        parts.append(
            f"<text x='{left + (column + 0.5) * cell_width:.1f}' y='29' "
            f"text-anchor='middle' font-size='12' fill='#637078'>{escape(label)}</text>"
        )
    for row_index, row in enumerate(reversed(rows)):
        y = top + row_index * cell_height
        score_range = (
            f"{row['judge_score_min']:.1f}-{row['judge_score_max']:.1f}"
            if row["judge_score_min"] is not None
            else "n/a"
        )
        parts.append(
            f"<text x='{left-10}' y='{y+25}' text-anchor='end' font-size='12' "
            f"fill='#172026'>{escape(row['label'].title())}</text>"
            f"<text x='{left-10}' y='{y+42}' text-anchor='end' font-size='10' "
            f"fill='#637078'>{score_range}</text>"
        )
        for column, cell in enumerate(row["cells"]):
            x = left + column * cell_width
            rate = cell["rate"]
            opacity = 0.04 if rate is None else 0.10 + 0.65 * rate
            label = (
                "n/a"
                if rate is None
                else f"{cell['recognized']}/{cell['total']} · {rate:.0%}"
            )
            parts.append(
                f"<rect x='{x+2}' y='{y+2}' width='{cell_width-4}' "
                f"height='{cell_height-4}' fill='#276b9c' opacity='{opacity:.2f}'/>"
                f"<text x='{x+cell_width/2:.1f}' y='{y+36}' text-anchor='middle' "
                f"font-size='12' fill='#172026'>{label}</text>"
            )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        "aria-label='Superior recognition by judge capability and candidate margin'>"
        f"{''.join(parts)}</svg>"
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _interval(row: Mapping[str, Any]) -> str:
    low = row["wilson_95_low"]
    high = row["wilson_95_high"]
    return "n/a" if low is None else f"{low:.1%}-{high:.1%}"


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
