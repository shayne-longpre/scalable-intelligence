from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def combine_accuracy(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    correct = sum(int(row["correct"]) for row in rows)
    pair_count = sum(int(row["pair_count"]) for row in rows)
    return {
        "correct": correct,
        "pair_count": pair_count,
        "accuracy": correct / pair_count if pair_count else None,
    }


def build_replication_report(
    *,
    first_result_path: str | Path,
    replication_result_path: str | Path,
    order_replay_path: str | Path,
    output_dir: str | Path,
    published_json_path: str | Path | None = None,
) -> dict[str, Any]:
    first = _load_json(first_result_path)
    replication = _load_json(replication_result_path)
    order_replay = _load_json(order_replay_path)
    first_conditions = {row["id"]: row for row in first["conditions"]}
    replication_conditions = {row["id"]: row for row in replication["conditions"]}
    condition_ids = [
        row["id"] for row in first["conditions"] if row["id"] in replication_conditions
    ]

    judge_comparisons = [
        _compare_judge(first_conditions[condition_id], replication_conditions[condition_id])
        for condition_id in condition_ids
    ]
    gap_rows = _combine_labeled_metrics(
        first["aggregate"]["final_pairs"]["by_score_gap"],
        replication["aggregate"]["final_pairs"]["by_score_gap"],
    )
    relative_rows = _combine_labeled_metrics(
        first["aggregate"]["final_pairs"]["by_relative_position"],
        replication["aggregate"]["final_pairs"]["by_relative_position"],
    )
    final = combine_accuracy(
        [
            first["aggregate"]["final_pairs"]["overall"],
            replication["aggregate"]["final_pairs"]["overall"],
        ]
    )
    opening = combine_accuracy(
        [
            condition["opening"]["pairs"]["overall"]
            for condition in [*first["conditions"], *replication["conditions"]]
        ]
    )
    question_types = _question_type_comparison(first, replication)
    self_relative = self_relative_summary(
        [*first["conditions"], *replication["conditions"]]
    )
    summary = {
        "schema_version": "oversight-replication-report-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "first_study": first["study"],
        "replication_study": replication["study"],
        "judge_comparisons": judge_comparisons,
        "pooled": {
            "opening_pairs": opening,
            "final_pairs": final,
            "by_score_gap": gap_rows,
            "by_relative_position": relative_rows,
            "superior_recognized": (
                first["aggregate"]["superior_recognized"]
                + replication["aggregate"]["superior_recognized"]
            ),
            "superior_total": (
                first["aggregate"]["superior_total"]
                + replication["aggregate"]["superior_total"]
            ),
            "self_relative_correct": (
                first["aggregate"]["self_relative_correct"]
                + replication["aggregate"]["self_relative_correct"]
            ),
            "self_relative_total": (
                first["aggregate"]["self_relative_total"]
                + replication["aggregate"]["self_relative_total"]
            ),
            "adaptive_improved_count": (
                first["aggregate"]["adaptive_improved_count"]
                + replication["aggregate"]["adaptive_improved_count"]
            ),
            "adaptive_unchanged_count": (
                first["aggregate"]["adaptive_unchanged_count"]
                + replication["aggregate"]["adaptive_unchanged_count"]
            ),
            "adaptive_worsened_count": (
                first["aggregate"]["adaptive_worsened_count"]
                + replication["aggregate"]["adaptive_worsened_count"]
            ),
            "reported_cost_usd": (
                first["aggregate"]["reported_cost_usd"]
                + replication["aggregate"]["reported_cost_usd"]
            ),
            "model_calls": (
                first["aggregate"]["model_calls"]
                + replication["aggregate"]["model_calls"]
            ),
            "self_relative_by_score_gap": self_relative["by_score_gap"],
            "superior_recognition_by_minimum_gap": self_relative[
                "superior_recognition_by_minimum_gap"
            ],
        },
        "question_types": question_types,
        "order_replay": order_replay["aggregate"],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report_card.html").write_text(
        render_replication_report(summary), encoding="utf-8"
    )
    if published_json_path:
        published_path = Path(published_json_path)
        published_path.parent.mkdir(parents=True, exist_ok=True)
        published_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def render_replication_report(summary: Mapping[str, Any]) -> str:
    pooled = summary["pooled"]
    order = summary["order_replay"]
    judge_rows = "".join(
        "<tr>"
        f"<th>{escape(row['judge_short_name'])}</th>"
        f"<td>{row['judge_external_score']:.1f}</td>"
        f"<td>{row['first_final_accuracy']:.1%}</td>"
        f"<td>{row['replication_final_accuracy']:.1%}</td>"
        f"<td>{row['final_accuracy_delta']:+.1%}</td>"
        f"<td>{row['first_superior_recognized']}/{row['first_superior_total']}</td>"
        f"<td>{row['replication_superior_recognized']}/{row['replication_superior_total']}</td>"
        f"<td>{row['first_self_relative_correct']}/6</td>"
        f"<td>{row['replication_self_relative_correct']}/6</td>"
        "</tr>"
        for row in summary["judge_comparisons"]
    )
    gap_rows = "".join(
        "<tr>"
        f"<th>{escape(row['label'])}</th>"
        f"<td>{row['first']['correct']}/{row['first']['pair_count']} "
        f"({row['first']['accuracy']:.1%})</td>"
        f"<td>{row['replication']['correct']}/{row['replication']['pair_count']} "
        f"({row['replication']['accuracy']:.1%})</td>"
        f"<td><strong>{row['pooled']['correct']}/{row['pooled']['pair_count']} "
        f"({row['pooled']['accuracy']:.1%})</strong></td>"
        "</tr>"
        for row in pooled["by_score_gap"]
    )
    self_gap_rows = "".join(
        "<tr>"
        f"<th>{escape(row['label'])}</th>"
        f"<td>{row['correct']}/{row['pair_count']}</td>"
        f"<td>{row['accuracy']:.1%}</td>"
        "</tr>"
        for row in pooled["self_relative_by_score_gap"]
    )
    superior_rows = "".join(
        "<tr>"
        f"<th>{escape(row['label'])}</th>"
        f"<td>{row['recognized']}/{row['total']}</td>"
        f"<td>{row['rate']:.1%}</td>"
        "</tr>"
        for row in pooled["superior_recognition_by_minimum_gap"]
    )
    type_rows = "".join(
        "<tr>"
        f"<th>{escape(_short_type(row['label']))}</th>"
        f"<td>{row['first_count']}</td><td>{row['replication_count']}</td>"
        f"<td>{row['total_count']}</td>"
        "</tr>"
        for row in summary["question_types"][:10]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Oversight frontier replication</title>
  <style>
    :root {{ --ink:#172026; --muted:#637078; --line:#d9dfe2; --soft:#f4f6f6;
      --blue:#276b9c; --red:#b84a45; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); font:15px/1.5 system-ui,sans-serif; }}
    main {{ max-width:1180px; margin:0 auto; padding:48px 26px 72px; }}
    h1 {{ margin:0 0 10px; max-width:850px; font:700 42px/1.08 Georgia,serif; }}
    h2 {{ margin:42px 0 12px; font:700 25px/1.2 Georgia,serif; }}
    p {{ max-width:850px; color:var(--muted); }}
    a {{ color:var(--blue); text-underline-offset:2px; }}
    .resources {{ display:flex; flex-wrap:wrap; gap:8px 18px; margin:18px 0 4px;
      font-size:13px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
      border-block:1px solid var(--line); margin:28px 0; }}
    .metric {{ padding:18px 12px; }}
    .metric strong {{ display:block; font:700 28px/1.1 Georgia,serif; }}
    .metric span {{ color:var(--muted); font-size:13px; }}
    .grid {{ display:grid; grid-template-columns:1.1fr .9fr; gap:24px; }}
    .grid > * {{ min-width:0; }}
    figure {{ margin:0; border-top:2px solid var(--ink); padding-top:10px; }}
    svg {{ width:100%; height:auto; display:block; background:var(--soft); }}
    figcaption {{ margin-top:7px; color:var(--muted); font-size:13px; }}
    .table-wrap {{ overflow:auto; border-top:2px solid var(--ink); }}
    table {{ width:100%; border-collapse:collapse; min-width:700px; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; }}
    thead th {{ color:var(--muted); font-size:12px; white-space:nowrap; }}
    .note {{ border-left:3px solid var(--blue); background:var(--soft);
      padding:12px 14px; color:var(--ink); }}
    @media (max-width:760px) {{
      main {{ padding:30px 16px 52px; }} h1 {{ font-size:34px; }}
      .metrics {{ grid-template-columns:1fr 1fr; }} .grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body><main>
  <h1>Can weaker models recognize stronger ones?</h1>
  <p>Two independent panels tested the same six judges. Each judge wrote five
  common probes, ranked seven anonymous candidates including itself, then used
  one targeted follow-up. External scores are a noisy reference, not ground truth.</p>
  <nav class="resources" aria-label="Study resources">
    <a href="../oversight_frontier_design.md">Protocol</a>
    <a href="taxonomy.html">Taxonomy</a>
    <a href="models.html">Model catalog</a>
    <a href="../../data/oversight_frontier_v2_probe_audit.json">Probe audit</a>
    <a href="../../data/oversight_frontier_replication_results.json">Results JSON</a>
  </nav>
  <section class="metrics">
    <div class="metric"><strong>{pooled['final_pairs']['accuracy']:.1%}</strong><span>pooled pair ordering</span></div>
    <div class="metric"><strong>{pooled['superior_recognized']}/{pooled['superior_total']}</strong><span>stronger candidates above judge</span></div>
    <div class="metric"><strong>{pooled['self_relative_correct']}/{pooled['self_relative_total']}</strong><span>all candidate-vs-self relations</span></div>
    <div class="metric"><strong>{pooled['by_score_gap'][-1]['pooled']['accuracy']:.1%}</strong><span>pairs separated by 10+ points</span></div>
  </section>

  <div class="grid">
    <figure>{_gap_svg(pooled['by_score_gap'])}
      <figcaption>Large capability gaps were consistently easier than local
      ordering. The two panels disagree most on near-ties.</figcaption></figure>
    <figure>{_judge_svg(summary['judge_comparisons'])}
      <figcaption>Condition-level accuracy moved substantially with the candidate
      panel. Judge identity alone does not define a stable frontier.</figcaption></figure>
  </div>

  <h2>Judge by judge</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Judge</th><th>Score</th><th>Panel 1</th><th>Panel 2</th><th>Change</th>
    <th>Stronger P1</th><th>Stronger P2</th><th>Self P1</th><th>Self P2</th></tr></thead>
    <tbody>{judge_rows}</tbody>
  </table></div>

  <h2>Capability gap</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>External-score gap</th><th>Panel 1</th><th>Panel 2</th><th>Pooled</th></tr></thead>
    <tbody>{gap_rows}</tbody>
  </table></div>

  <div class="grid">
    <div><h2>Candidate versus judge</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>Absolute score gap</th><th>Correct</th><th>Accuracy</th></tr></thead>
        <tbody>{self_gap_rows}</tbody>
      </table></div>
    </div>
    <div><h2>Recognizing a stronger model</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>Minimum lead over judge</th><th>Recognized</th><th>Rate</th></tr></thead>
        <tbody>{superior_rows}</tbody>
      </table></div>
    </div>
  </div>

  <h2>What judges tested</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Question type</th><th>Panel 1</th><th>Panel 2</th><th>Total</th></tr></thead>
    <tbody>{type_rows}</tbody>
  </table></div>

  <h2>Robustness checks</h2>
  <p class="note">The exact-evidence order replay changed all
  {order['comparison_count']} answer presentations. Mean rank agreement was
  Kendall τ={order['mean_kendall_tau']:.2f}; all three judges retained the same
  top candidate, but the two lower-capability judges each fell to τ=0.52.
  Presentation order is therefore a real uncertainty, especially below the
  strongest judge.</p>
  <p>Across both panels, the adaptive probe improved
  {pooled['adaptive_improved_count']} rankings, left
  {pooled['adaptive_unchanged_count']} unchanged, and worsened
  {pooled['adaptive_worsened_count']}. Pooled accuracy moved from
  {pooled['opening_pairs']['accuracy']:.1%} to {pooled['final_pairs']['accuracy']:.1%}.
  The mechanism is operationally useful, but its effect is not uniformly positive.</p>
</main></body></html>
"""


def _compare_judge(
    first: Mapping[str, Any],
    replication: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": first["id"],
        "judge_short_name": first["judge_short_name"],
        "judge_external_score": first["judge_external_score"],
        "first_final_accuracy": first["final"]["pairs"]["overall"]["accuracy"],
        "replication_final_accuracy": replication["final"]["pairs"]["overall"]["accuracy"],
        "final_accuracy_delta": (
            replication["final"]["pairs"]["overall"]["accuracy"]
            - first["final"]["pairs"]["overall"]["accuracy"]
        ),
        "first_superior_recognized": first["final"]["superior_recognized"],
        "first_superior_total": first["final"]["superior_total"],
        "replication_superior_recognized": replication["final"]["superior_recognized"],
        "replication_superior_total": replication["final"]["superior_total"],
        "first_self_relative_correct": first["final"]["self_relative_correct"],
        "replication_self_relative_correct": replication["final"]["self_relative_correct"],
        "first_adaptive_delta": first["adaptive_delta_pair_accuracy"],
        "replication_adaptive_delta": replication["adaptive_delta_pair_accuracy"],
    }


def _combine_labeled_metrics(
    first: Sequence[Mapping[str, Any]],
    replication: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    replication_by_label = {row["label"]: row for row in replication}
    return [
        {
            "label": row["label"],
            "first": dict(row),
            "replication": dict(replication_by_label[row["label"]]),
            "pooled": combine_accuracy([row, replication_by_label[row["label"]]]),
        }
        for row in first
        if row["label"] in replication_by_label
    ]


def _question_type_comparison(
    first: Mapping[str, Any],
    replication: Mapping[str, Any],
) -> list[dict[str, Any]]:
    first_counts = Counter(
        {
            label: sum(
                condition["question_type_counts"].get(label, 0)
                for condition in first["conditions"]
            )
            for label in {
                label
                for condition in first["conditions"]
                for label in condition["question_type_counts"]
            }
        }
    )
    replication_counts = Counter(
        {
            label: sum(
                condition["question_type_counts"].get(label, 0)
                for condition in replication["conditions"]
            )
            for label in {
                label
                for condition in replication["conditions"]
                for label in condition["question_type_counts"]
            }
        }
    )
    labels = first_counts.keys() | replication_counts.keys()
    rows = [
        {
            "label": label,
            "first_count": first_counts[label],
            "replication_count": replication_counts[label],
            "total_count": first_counts[label] + replication_counts[label],
        }
        for label in labels
    ]
    return sorted(rows, key=lambda row: (-row["total_count"], row["label"]))


def self_relative_summary(
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observations = []
    for condition in conditions:
        ranking = condition["final"]["ranking"]
        positions = {
            participant_id: index for index, participant_id in enumerate(ranking)
        }
        self_participant = condition["self_participant"]
        judge_score = float(condition["judge_external_score"])
        for participant_id, score in condition["participant_scores"].items():
            if participant_id == self_participant:
                continue
            score = float(score)
            candidate_above = positions[participant_id] < positions[self_participant]
            observations.append(
                {
                    "score_gap": abs(score - judge_score),
                    "superior_gap": score - judge_score,
                    "correct": candidate_above == (score > judge_score),
                    "recognized": score > judge_score and candidate_above,
                }
            )

    bins = (
        ("<2", 0.0, 2.0),
        ("2-5", 2.0, 5.0),
        ("5-10", 5.0, 10.0),
        ("10+", 10.0, None),
    )
    by_gap = []
    for label, lower, upper in bins:
        rows = [
            row
            for row in observations
            if row["score_gap"] >= lower
            and (upper is None or row["score_gap"] < upper)
        ]
        correct = sum(row["correct"] for row in rows)
        by_gap.append(
            {
                "label": label,
                "correct": correct,
                "pair_count": len(rows),
                "accuracy": correct / len(rows) if rows else None,
            }
        )

    superior = []
    for threshold in (0.0, 2.0, 5.0, 10.0):
        rows = [
            row
            for row in observations
            if row["superior_gap"] > 0
            and row["superior_gap"] >= threshold
        ]
        recognized = sum(row["recognized"] for row in rows)
        superior.append(
            {
                "label": ">0 points" if threshold == 0 else f"{threshold:g}+ points",
                "recognized": recognized,
                "total": len(rows),
                "rate": recognized / len(rows) if rows else None,
            }
        )
    return {
        "by_score_gap": by_gap,
        "superior_recognition_by_minimum_gap": superior,
    }


def _gap_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    width, height = 520, 300
    left, right, top, bottom = 48, 20, 24, 48
    plot_w, plot_h = width - left - right, height - top - bottom

    def y(value: float) -> float:
        return top + (1 - value) * plot_h

    parts = []
    for tick in (0.4, 0.6, 0.8, 1.0):
        parts.append(
            f"<line x1='{left}' y1='{y(tick):.1f}' x2='{width-right}' y2='{y(tick):.1f}' stroke='#d9dfe2'/>"
            f"<text x='{left-7}' y='{y(tick)+4:.1f}' text-anchor='end' font-size='11' fill='#637078'>{tick:.0%}</text>"
        )
    colors = {"first": "#9aa5ab", "replication": "#276b9c", "pooled": "#172026"}
    point_sets: dict[str, list[str]] = {key: [] for key in colors}
    for index, row in enumerate(rows):
        x = left + (index + 0.5) * plot_w / len(rows)
        parts.append(
            f"<text x='{x:.1f}' y='{height-18}' text-anchor='middle' font-size='11' fill='#637078'>{escape(row['label'])}</text>"
        )
        for key in colors:
            point_sets[key].append(f"{x:.1f},{y(row[key]['accuracy']):.1f}")
    for key, color in colors.items():
        points = " ".join(point_sets[key])
        parts.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2'/>")
        parts.extend(
            f"<circle cx='{point.split(',')[0]}' cy='{point.split(',')[1]}' r='4' fill='{color}'/>"
            for point in point_sets[key]
        )
    parts.append(
        "<text x='286' y='14' font-size='11' fill='#9aa5ab'>Panel 1</text>"
        "<text x='354' y='14' font-size='11' fill='#276b9c'>Panel 2</text>"
        "<text x='425' y='14' font-size='11' fill='#172026'>Pooled</text>"
    )
    return f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Accuracy by capability gap'>{''.join(parts)}</svg>"


def _judge_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    width, height = 520, 300
    left, right, top, bottom = 64, 22, 24, 50
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(value: float) -> float:
        return left + (value - 0.4) / 0.6 * plot_w

    parts = []
    for tick in (0.4, 0.6, 0.8, 1.0):
        px = x(tick)
        parts.append(
            f"<line x1='{px:.1f}' y1='{top}' x2='{px:.1f}' y2='{height-bottom}' stroke='#d9dfe2'/>"
            f"<text x='{px:.1f}' y='{height-20}' text-anchor='middle' font-size='11' fill='#637078'>{tick:.0%}</text>"
        )
    for index, row in enumerate(rows):
        py = top + (index + 0.5) * plot_h / len(rows)
        first_x = x(row["first_final_accuracy"])
        second_x = x(row["replication_final_accuracy"])
        parts.append(
            f"<text x='{left-7}' y='{py+4:.1f}' text-anchor='end' font-size='10' fill='#637078'>{escape(row['judge_short_name'])}</text>"
            f"<line x1='{first_x:.1f}' y1='{py:.1f}' x2='{second_x:.1f}' y2='{py:.1f}' stroke='#aeb8bd' stroke-width='2'/>"
            f"<circle cx='{first_x:.1f}' cy='{py:.1f}' r='5' fill='#9aa5ab'/>"
            f"<circle cx='{second_x:.1f}' cy='{py:.1f}' r='5' fill='#276b9c'/>"
        )
    return f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Judge accuracy across panels'>{''.join(parts)}</svg>"


def _short_type(label: str) -> str:
    return label.replace("_", " ").replace("reasoning", "").strip().title()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
