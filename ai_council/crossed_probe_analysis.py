from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from ai_council.analysis import analyze_run


def build_crossed_probe_report(
    *,
    study_path: str | Path,
    runs_root: str | Path,
    output_dir: str | Path,
    published_json_path: str | Path | None = None,
) -> dict[str, Any]:
    study_path = Path(study_path)
    study = _load_json(study_path)
    run_dirs = discover_crossed_runs(
        study_path,
        runs_root,
    )
    missing = [
        condition["id"]
        for condition in study["conditions"]
        if condition["id"] not in run_dirs
    ]
    if missing:
        raise ValueError(f"completed crossed runs are missing: {missing}")

    catalog = _load_json(study["catalog"])
    display_names = {
        row["provider_model_id"]: row.get("display_name")
        for row in catalog.get("models", [])
    }
    cells = [
        _condition_cell(
            condition,
            run_dirs[condition["id"]],
            study["catalog"],
            display_names,
        )
        for condition in study["conditions"]
    ]
    summary = {
        "schema_version": "crossed-probe-report-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study": study["name"],
        "research_question": study["research_question"],
        "study_file": str(study_path),
        "catalog_file": study["catalog"],
        "cells": cells,
        **summarize_crossed_cells(cells),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report_card.html").write_text(
        render_crossed_probe_report(summary),
        encoding="utf-8",
    )
    if published_json_path:
        published_path = Path(published_json_path)
        published_path.parent.mkdir(parents=True, exist_ok=True)
        published_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


def discover_crossed_runs(
    study_path: str | Path,
    runs_root: str | Path,
) -> dict[str, Path]:
    study_path = Path(study_path)
    study = _load_json(study_path)
    expected = {condition["id"]: condition for condition in study["conditions"]}
    matches: dict[str, list[Path]] = defaultdict(list)
    for config_path in Path(runs_root).glob("*/config.json"):
        config = _load_json(config_path)
        metadata = config.get("metadata", {})
        condition_id = metadata.get("study_condition")
        condition = expected.get(condition_id)
        if condition is None or metadata.get("exclude_from_study_analysis"):
            continue
        if not _same_path(metadata.get("study_file"), study_path):
            continue
        if not _same_path(
            metadata.get("exact_evidence_source_run"),
            condition["source_run"],
        ):
            continue
        if metadata.get("probe_author_model") != condition["probe_author_model"]:
            continue
        if metadata.get("cross_judge_model") != condition["evaluator_model"]:
            continue
        summary_path = config_path.parent / "run_summary.json"
        if (
            summary_path.exists()
            and _load_json(summary_path).get("status") == "completed"
        ):
            matches[condition_id].append(config_path.parent)
    duplicates = {
        condition_id: paths
        for condition_id, paths in matches.items()
        if len(paths) > 1
    }
    if duplicates:
        details = ", ".join(
            f"{condition_id} ({len(paths)})"
            for condition_id, paths in sorted(duplicates.items())
        )
        raise ValueError(
            "multiple completed crossed runs match the same condition; "
            f"exclude superseded runs explicitly: {details}"
        )
    return {
        condition_id: paths[0]
        for condition_id, paths in matches.items()
        if paths
    }


def summarize_crossed_cells(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_author: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_evaluator: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_author[cell["probe_author_model"]].append(cell)
        by_evaluator[cell["evaluator_model"]].append(cell)

    authors = [
        {
            "model": model,
            "short_name": rows[0]["probe_author_short_name"],
            "evaluator_count": len(rows),
            "source_opening_pair_accuracy": float(
                rows[0]["source_opening_pair_accuracy"]
            ),
            "source_final_pair_accuracy": float(
                rows[0]["source_final_pair_accuracy"]
            ),
            "opening_pair_accuracy_mean": _metric_mean(
                rows, "opening_pair_accuracy"
            ),
            "final_pair_accuracy_mean": _metric_mean(
                rows, "final_pair_accuracy"
            ),
        }
        for model, rows in by_author.items()
    ]
    evaluators = [
        {
            "model": model,
            "short_name": rows[0]["evaluator_short_name"],
            "battery_count": len(rows),
            "opening_pair_accuracy_mean": _metric_mean(
                rows, "opening_pair_accuracy"
            ),
            "final_pair_accuracy_mean": _metric_mean(
                rows, "final_pair_accuracy"
            ),
        }
        for model, rows in by_evaluator.items()
    ]
    return {
        "authors": sorted(
            authors,
            key=lambda row: (
                -row["opening_pair_accuracy_mean"],
                row["short_name"],
            ),
        ),
        "evaluators": sorted(
            evaluators,
            key=lambda row: (
                -row["opening_pair_accuracy_mean"],
                row["short_name"],
            ),
        ),
        "opening_pair_accuracy_mean": _metric_mean(
            cells, "opening_pair_accuracy"
        ),
        "final_pair_accuracy_mean": _metric_mean(
            cells, "final_pair_accuracy"
        ),
        "reported_cost_usd": sum(
            float(cell["reported_cost_usd"]) for cell in cells
        ),
        "model_calls": sum(int(cell["model_calls"]) for cell in cells),
    }


def render_crossed_probe_report(summary: Mapping[str, Any]) -> str:
    evaluator_models = [
        row["model"] for row in summary["evaluators"]
    ]
    evaluator_names = {
        row["model"]: row["short_name"] for row in summary["evaluators"]
    }
    cells_by_author: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    author_names = {}
    for cell in summary["cells"]:
        author = cell["probe_author_model"]
        cells_by_author[author][cell["evaluator_model"]] = cell
        author_names[author] = cell["probe_author_short_name"]

    header = "".join(
        f"<th>{escape(evaluator_names[model])}</th>"
        for model in evaluator_models
    )
    rows = []
    for author in sorted(
        cells_by_author,
        key=lambda model: author_names[model],
    ):
        values = []
        for evaluator in evaluator_models:
            cell = cells_by_author[author].get(evaluator)
            if cell is None:
                values.append("<td class='missing'>not run</td>")
                continue
            values.append(
                "<td>"
                f"<strong>{_pct(cell['opening_pair_accuracy'])}</strong>"
                f"<span>{_pct(cell['final_pair_accuracy'])} after follow-up</span>"
                "</td>"
            )
        rows.append(
            f"<tr><th>{escape(author_names[author])}</th>{''.join(values)}</tr>"
        )

    author_summary = "".join(
        "<tr>"
        f"<th>{escape(row['short_name'])}</th>"
        f"<td>{_pct(row['source_opening_pair_accuracy'])}</td>"
        f"<td>{_pct(row['source_final_pair_accuracy'])}</td>"
        f"<td>{_pct(row['opening_pair_accuracy_mean'])}</td>"
        f"<td>{_pct(row['final_pair_accuracy_mean'])}</td>"
        "</tr>"
        for row in summary["authors"]
    )
    evaluator_summary = "".join(
        "<tr>"
        f"<th>{escape(row['short_name'])}</th>"
        f"<td>{_pct(row['opening_pair_accuracy_mean'])}</td>"
        f"<td>{_pct(row['final_pair_accuracy_mean'])}</td>"
        "</tr>"
        for row in summary["evaluators"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crossed probe study</title>
  <style>
    :root {{ color-scheme: light; --ink:#17211d; --muted:#5d6964; --line:#ccd4cf;
      --paper:#f8faf8; --accent:#0b6b53; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink);
      font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    main {{ width:min(1040px,calc(100% - 32px)); margin:42px auto 72px; }}
    h1 {{ margin:0 0 8px; font:700 36px/1.1 Georgia,serif; }}
    h2 {{ margin:34px 0 10px; font:700 22px/1.2 Georgia,serif; }}
    p {{ max-width:760px; color:var(--muted); }}
    .metric {{ display:inline-block; margin:16px 20px 8px 0; }}
    .metric strong {{ display:block; color:var(--accent); font-size:26px; }}
    figure {{ margin:0; overflow-x:auto; }}
    svg {{ display:block; width:100%; min-width:680px; height:auto; background:white; }}
    .table-wrap {{ overflow-x:auto; border-top:2px solid var(--ink); }}
    table {{ width:100%; border-collapse:collapse; background:white; }}
    th,td {{ padding:11px 13px; border-bottom:1px solid var(--line); text-align:left; }}
    thead th {{ color:var(--muted); font-size:12px; text-transform:uppercase; }}
    tbody td strong,tbody td span {{ display:block; }}
    tbody td span {{ color:var(--muted); font-size:12px; margin-top:3px; }}
    .missing {{ color:var(--muted); }}
    .note {{ font-size:13px; }}
  </style>
</head>
<body><main>
  <h1>Who writes the test, and who reads the evidence?</h1>
  <p>{escape(summary['research_question'])}</p>
  <div class="metric"><strong>{_pct(summary['opening_pair_accuracy_mean'])}</strong>
    opening mean</div>
  <div class="metric"><strong>{_pct(summary['final_pair_accuracy_mean'])}</strong>
    after follow-up</div>
  <div class="metric"><strong>{len(summary['cells'])}</strong>crossed cells</div>

  <h2>Five-probe comparison</h2>
  <figure>{_opening_accuracy_svg(summary)}</figure>
  <p class="note">Dots use a fixed 0–100% axis. The gray square is the original
    probe author's judgment under the source presentation order; colored dots
    are the strict same-order crossed evaluations.</p>

  <h2>Exact-evidence matrix</h2>
  <p>Rows identify the model that wrote the probes. Columns identify the model
    that interpreted the same archived answers. Answer order is held fixed within
    each row.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Probe author</th>{header}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>

  <h2>Probe-author effect</h2>
  <p>The source-author columns reproduce the original judgment and therefore
    use the source run's presentation order. The crossed-evaluator mean uses the
    matched order shown above.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Author</th><th>Source author, five</th>
      <th>Source author, final</th><th>Crossed mean, five</th>
      <th>Crossed mean, final</th></tr></thead>
    <tbody>{author_summary}</tbody>
  </table></div>

  <h2>Evaluator effect</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>Evaluator</th><th>Five probes</th><th>After follow-up</th></tr></thead>
    <tbody>{evaluator_summary}</tbody>
  </table></div>
  <p class="note">The external catalog is a reference ordering, not ground truth.
    This control changes only the evaluator; probes, candidate answers, anonymous
    identities, and within-probe presentation order are reused exactly.</p>
</main></body></html>
"""


def _opening_accuracy_svg(summary: Mapping[str, Any]) -> str:
    evaluators = [row["model"] for row in summary["evaluators"]]
    evaluator_names = {
        row["model"]: row["short_name"] for row in summary["evaluators"]
    }
    colors = ("#0b6b53", "#b0442e", "#356ea3", "#8a5f20")
    color_by_evaluator = {
        model: colors[index % len(colors)]
        for index, model in enumerate(evaluators)
    }
    by_author: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in summary["cells"]:
        by_author[cell["probe_author_model"]].append(cell)
    authors = sorted(
        by_author,
        key=lambda model: by_author[model][0]["probe_author_short_name"],
    )

    width = 860
    left, right, top, row_height = 190, 28, 70, 68
    plot_width = width - left - right
    height = top + len(authors) * row_height + 36

    def x(value: float) -> float:
        return left + float(value) * plot_width

    parts = []
    for tick in (0, 0.25, 0.5, 0.75, 1):
        parts.append(
            f"<line x1='{x(tick):.1f}' y1='{top-18}' x2='{x(tick):.1f}' "
            f"y2='{height-26}' stroke='#e0e5e2'/>"
            f"<text x='{x(tick):.1f}' y='{top-29}' text-anchor='middle' "
            f"font-size='11' fill='#5d6964'>{tick:.0%}</text>"
        )
    for row_index, author in enumerate(authors):
        rows = by_author[author]
        y = top + row_index * row_height + 18
        values = [float(row["opening_pair_accuracy"]) for row in rows]
        parts.append(
            f"<text x='{left-12}' y='{y+4}' text-anchor='end' "
            f"font-size='12' fill='#17211d'>"
            f"{escape(rows[0]['probe_author_short_name'])}</text>"
            f"<line x1='{x(min(values)):.1f}' y1='{y}' "
            f"x2='{x(max(values)):.1f}' y2='{y}' stroke='#aab5af' "
            "stroke-width='2'/>"
        )
        source_value = float(rows[0]["source_opening_pair_accuracy"])
        parts.append(
            f"<rect x='{x(source_value)-4:.1f}' y='{y-4:.1f}' width='8' "
            "height='8' fill='#707b76'/>"
        )
        for row in rows:
            evaluator = row["evaluator_model"]
            parts.append(
                f"<circle cx='{x(row['opening_pair_accuracy']):.1f}' cy='{y}' "
                f"r='6' fill='{color_by_evaluator[evaluator]}'/>"
            )
    legend_x = left
    parts.append(
        f"<rect x='{legend_x}' y='17' width='8' height='8' fill='#707b76'/>"
        f"<text x='{legend_x+13}' y='25' font-size='11' fill='#5d6964'>"
        "source author</text>"
    )
    legend_x += 105
    for evaluator in evaluators:
        parts.append(
            f"<circle cx='{legend_x+5}' cy='21' r='5' "
            f"fill='{color_by_evaluator[evaluator]}'/>"
            f"<text x='{legend_x+15}' y='25' font-size='11' fill='#5d6964'>"
            f"{escape(evaluator_names[evaluator])}</text>"
        )
        legend_x += 128
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        "aria-label='Five-probe pairwise accuracy by probe author and evaluator'>"
        f"{''.join(parts)}</svg>"
    )


def _condition_cell(
    condition: Mapping[str, Any],
    run_dir: Path,
    catalog_path: str | Path,
    display_names: Mapping[str, Any],
) -> dict[str, Any]:
    run_summary = _load_json(run_dir / "run_summary.json")
    judgments = _judgments_for_run(run_dir, catalog_path)
    source_judgments = _judgments_for_run(
        Path(condition["source_run"]),
        catalog_path,
    )
    opening = judgments[0]
    final = judgments[-1]
    return {
        "id": condition["id"],
        "run_dir": str(run_dir),
        "source_run": condition["source_run"],
        "probe_author_model": condition["probe_author_model"],
        "probe_author_short_name": _display_name(
            condition["probe_author_model"],
            display_names,
        ),
        "evaluator_model": condition["evaluator_model"],
        "evaluator_short_name": _display_name(
            condition["evaluator_model"],
            display_names,
        ),
        "comparison_seed": int(condition["comparison_seed"]),
        "candidate_count": len(opening["ranking"]),
        "opening_pair_accuracy": float(opening["pairwise_accuracy"]),
        "opening_kendall_tau": float(opening["kendall_tau"]),
        "opening_spearman_rho": float(opening["spearman_rho"]),
        "final_pair_accuracy": float(final["pairwise_accuracy"]),
        "final_kendall_tau": float(final["kendall_tau"]),
        "final_spearman_rho": float(final["spearman_rho"]),
        "adaptive_delta_pair_accuracy": (
            float(final["pairwise_accuracy"])
            - float(opening["pairwise_accuracy"])
        ),
        "source_opening_pair_accuracy": float(
            source_judgments[0]["pairwise_accuracy"]
        ),
        "source_final_pair_accuracy": float(
            source_judgments[-1]["pairwise_accuracy"]
        ),
        "reported_cost_usd": float(run_summary.get("reported_cost_usd", 0)),
        "model_calls": int(run_summary.get("model_calls", 0)),
    }


def _judgments_for_run(
    run_dir: Path,
    catalog_path: str | Path,
) -> list[dict[str, Any]]:
    analysis_path = run_dir / "analysis_summary.json"
    if not analysis_path.exists():
        analyze_run(run_dir, prior_ranking_file=catalog_path)
    analysis = _load_json(analysis_path)
    judgments = sorted(
        analysis["prior_agreement"]["judgments"],
        key=lambda row: (int(row.get("round_index", 0)), row["phase"]),
    )
    if not judgments:
        raise ValueError(f"run has no judgments: {run_dir}")
    return judgments


def _metric_mean(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> float:
    if not rows:
        raise ValueError("cannot summarize an empty crossed study")
    return mean(float(row[field]) for row in rows)


def _display_name(
    model_id: str,
    display_names: Mapping[str, Any],
) -> str:
    value = display_names.get(model_id)
    if isinstance(value, str) and value:
        return value.split(": ", 1)[-1]
    return model_id.split("/", 1)[-1].replace("-", " ").title()


def _pct(value: float) -> str:
    return f"{100 * float(value):.1f}%"


def _same_path(left: Any, right: str | Path) -> bool:
    return isinstance(left, str) and Path(left).resolve() == Path(right).resolve()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
