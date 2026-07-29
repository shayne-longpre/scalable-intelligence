from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from ai_council.rankings import kendall_tau_between


METRICS = (
    "kendall_tau",
    "spearman_rho",
    "pairwise_accuracy",
    "rank_score_r_squared",
)


def build_catalog_probe_extension_report(
    *,
    pairs: Sequence[tuple[str, str | Path, str | Path]],
    output_dir: str | Path,
    published_json_path: str | Path | None = None,
) -> dict[str, Any]:
    conditions = [
        compare_catalog_runs(label, source_run, extension_run)
        for label, source_run, extension_run in pairs
    ]
    summary = {
        "schema_version": "catalog-probe-extension-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "conditions": conditions,
        "aggregate": {
            "opening_pairwise_accuracy_at_5": mean(
                row["source_opening"]["pairwise_accuracy"] for row in conditions
            ),
            "opening_pairwise_accuracy_at_10": mean(
                row["extended_opening"]["pairwise_accuracy"] for row in conditions
            ),
            "opening_pairwise_accuracy_delta": mean(
                row["opening_delta"]["pairwise_accuracy"] for row in conditions
            ),
            "final_pairwise_accuracy": mean(
                row["extended_final"]["pairwise_accuracy"] for row in conditions
            ),
            "opening_kendall_tau_at_5": mean(
                row["source_opening"]["kendall_tau"] for row in conditions
            ),
            "opening_kendall_tau_at_10": mean(
                row["extended_opening"]["kendall_tau"] for row in conditions
            ),
            "opening_kendall_tau_delta": mean(
                row["opening_delta"]["kendall_tau"] for row in conditions
            ),
            "incremental_reported_cost_usd": sum(
                row["lineage"]["reported_cost_usd"] for row in conditions
            ),
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report_card.html").write_text(
        render_report(summary),
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


def compare_catalog_runs(
    label: str,
    source_run: str | Path,
    extension_run: str | Path,
) -> dict[str, Any]:
    source_run = Path(source_run)
    extension_run = Path(extension_run)
    _validate_roster(source_run, extension_run)
    source = _load_json(source_run / "analysis_summary.json")
    extension = _load_json(extension_run / "analysis_summary.json")
    source_checkpoints = _checkpoints(source)
    extension_checkpoints = _checkpoints(extension)
    source_opening = source_checkpoints[0]
    extended_opening = extension_checkpoints[0]
    extended_final = extension_checkpoints[-1]
    return {
        "label": label,
        "source_run": str(source_run),
        "extension_run": str(extension_run),
        "source_checkpoints": source_checkpoints,
        "extension_checkpoints": extension_checkpoints,
        "source_opening": source_opening,
        "extended_opening": extended_opening,
        "extended_final": extended_final,
        "opening_delta": {
            metric: extended_opening[metric] - source_opening[metric]
            for metric in METRICS
        },
        "source_to_extended_opening": _ranking_change(
            source_opening["ranking"],
            extended_opening["ranking"],
        ),
        "adaptive_steps": [
            {
                "from_probe_count": left["probe_count"],
                "to_probe_count": right["probe_count"],
                **_ranking_change(left["ranking"], right["ranking"]),
                "pairwise_accuracy_delta": (
                    right["pairwise_accuracy"] - left["pairwise_accuracy"]
                ),
                "kendall_tau_delta": right["kendall_tau"] - left["kendall_tau"],
            }
            for left, right in zip(
                extension_checkpoints,
                extension_checkpoints[1:],
            )
        ],
        "adaptive_targets": _adaptive_targets(extension_run),
        "unavailable_answer_count": _unavailable_answer_count(extension_run),
        "lineage": _repair_lineage(extension_run),
    }


def _checkpoints(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for judgment in summary["prior_agreement"]["judgments"]:
        scored = judgment.get("reported_score_subset") or judgment
        rows.append(
            {
                "probe_count": int(judgment["judgment_probe_count"]),
                "ranking": list(scored["ranking"]),
                **{metric: float(scored[metric]) for metric in METRICS},
            }
        )
    if not rows:
        raise ValueError("analysis has no prior-agreement judgments")
    return sorted(rows, key=lambda row: row["probe_count"])


def _ranking_change(left: Sequence[str], right: Sequence[str]) -> dict[str, Any]:
    right_set = set(right)
    shared = [participant for participant in left if participant in right_set]
    if len(shared) != len(left) or len(shared) != len(right):
        raise ValueError("rankings must contain the same participants")
    left_positions = {participant: index for index, participant in enumerate(left)}
    right_positions = {participant: index for index, participant in enumerate(right)}
    displacements = [
        abs(left_positions[participant] - right_positions[participant])
        for participant in shared
    ]
    top_count = min(5, len(shared))
    return {
        "kendall_tau": kendall_tau_between(list(left), list(right)),
        "top5_overlap": (
            len(set(left[:top_count]) & set(right[:top_count])) / top_count
            if top_count
            else 0.0
        ),
        "mean_absolute_rank_change": mean(displacements) if displacements else 0.0,
        "max_absolute_rank_change": max(displacements, default=0),
        "moved_candidate_count": sum(value > 0 for value in displacements),
    }


def _adaptive_targets(run_dir: Path) -> list[dict[str, Any]]:
    participant_models = _participant_models(run_dir)
    rows = []
    for entry in _jsonl(run_dir / "transcript.jsonl"):
        metadata = entry.get("metadata") or {}
        if (
            metadata.get("interaction_role") != "question"
            or int(entry.get("round_index") or 0) <= 1
        ):
            continue
        respondents = list(metadata.get("respondents") or [])
        rows.append(
            {
                "round": int(entry["round_index"]),
                "probe_count_after_round": int(metadata["probe_sequence_number"]),
                "participant_ids": respondents,
                "models": [participant_models[value] for value in respondents],
            }
        )
    return rows


def _repair_lineage(run_dir: Path) -> dict[str, Any]:
    seen: set[Path] = set()
    current = run_dir
    rows = []
    runtime_sensitive = False
    extension_source: str | None = None
    while current not in seen:
        seen.add(current)
        config = _load_json(current / "config.json")
        accounting = _run_accounting(current)
        metadata = config.get("metadata") or {}
        extension_source = extension_source or metadata.get(
            "ceiling_extension_source_run"
        )
        overrides = metadata.get("repair_parameter_overrides") or {}
        runtime_sensitive = runtime_sensitive or bool(overrides)
        rows.append(
            {
                "run_dir": str(current),
                "model_calls": accounting["model_calls"],
                "reported_cost_usd": accounting["reported_cost_usd"],
                "parameter_overrides": overrides,
            }
        )
        source = metadata.get("resume_source_run") or metadata.get(
            "repair_source_run"
        )
        if not source:
            break
        source_path = Path(source)
        source_metadata = (
            _load_json(source_path / "config.json").get("metadata") or {}
        )
        if (
            extension_source
            and source_metadata.get("ceiling_extension_source_run")
            != extension_source
        ):
            break
        current = source_path
    return {
        "runs": rows,
        "model_calls": sum(row["model_calls"] for row in rows),
        "reported_cost_usd": sum(row["reported_cost_usd"] for row in rows),
        "runtime_sensitive": runtime_sensitive,
    }


def _run_accounting(run_dir: Path) -> dict[str, int | float]:
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        summary = _load_json(summary_path)
        return {
            "model_calls": int(summary.get("model_calls") or 0),
            "reported_cost_usd": float(
                summary.get("reported_cost_usd") or 0
            ),
        }

    direct_entries = []
    for entry in _jsonl(run_dir / "transcript.jsonl"):
        metadata = entry.get("metadata") or {}
        if (
            not metadata.get("provider")
            or metadata.get("provider") == "preauthored"
            or metadata.get("finish_reason") == "replayed"
        ):
            continue
        direct_entries.append(metadata)
    return {
        "model_calls": len(direct_entries),
        "reported_cost_usd": sum(
            float((metadata.get("usage") or {}).get("cost") or 0)
            for metadata in direct_entries
        ),
    }


def _unavailable_answer_count(run_dir: Path) -> int:
    return sum(
        (entry.get("metadata") or {}).get("answer_unavailable") is True
        for entry in _jsonl(run_dir / "transcript.jsonl")
    )


def _validate_roster(source_run: Path, extension_run: Path) -> None:
    if _participant_models(source_run) != _participant_models(extension_run):
        raise ValueError("source and extension candidate rosters differ")


def _participant_models(run_dir: Path) -> dict[str, str]:
    config = _load_json(run_dir / "config.json")
    models = config["models"]
    model_rows = models if isinstance(models, dict) else {
        row["name"]: row for row in models
    }
    return {
        participant["id"]: model_rows[participant["model"]]["model"]
        for participant in config["participants"]
    }


def render_report(summary: Mapping[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{escape(row['label'])}</td>"
        f"<td>{row['source_opening']['pairwise_accuracy']:.1%}</td>"
        f"<td>{row['extended_opening']['pairwise_accuracy']:.1%}</td>"
        f"<td>{row['opening_delta']['pairwise_accuracy']:+.1%}</td>"
        f"<td>{row['extended_final']['pairwise_accuracy']:.1%}</td>"
        f"<td>{row['source_to_extended_opening']['kendall_tau']:.2f}</td>"
        f"<td>${row['lineage']['reported_cost_usd']:.2f}</td>"
        "</tr>"
        for row in summary["conditions"]
    )
    aggregate = summary["aggregate"]
    trajectory = _trajectory_svg(summary["conditions"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ten-probe catalog extension</title><style>
:root{{--ink:#172126;--muted:#66747b;--line:#d9e0e3;--blue:#176c8c;--red:#bf513b}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif}}
main{{max-width:1080px;margin:auto;padding:46px 26px 70px}}h1{{font:700 42px/1.08 Georgia,serif;margin:0 0 12px}}
h2{{font:700 26px/1.2 Georgia,serif;margin-top:42px}}.lede,figcaption{{color:var(--muted)}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line);margin:28px 0}}
.metric{{padding:18px;border-right:1px solid var(--line)}}.metric:last-child{{border:0}}
.metric strong{{display:block;font-size:28px}}table{{width:100%;border-collapse:collapse;border-top:2px solid var(--ink)}}
th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{font-size:12px;color:var(--muted)}}figure{{margin:30px 0}}svg{{width:100%;height:auto}}
@media(max-width:700px){{main{{padding:28px 15px}}.metrics{{grid-template-columns:1fr}}.metric{{border-right:0;border-bottom:1px solid var(--line)}}}}</style>
</head><body><main><h1>What changes when five opening probes become ten?</h1>
<p class="lede">The original five-probe evidence is held fixed. Each judge adds five
complementary probes, ranks the same anonymous 50-model ladder, then receives two
targeted adaptive probes.</p><section class="metrics">
<div class="metric"><strong>{aggregate['opening_pairwise_accuracy_at_10']:.1%}</strong><span>mean accuracy at 10 probes</span></div>
<div class="metric"><strong>{aggregate['opening_pairwise_accuracy_delta']:+.1%}</strong><span>change from 5 probes</span></div>
<div class="metric"><strong>${aggregate['incremental_reported_cost_usd']:.2f}</strong><span>extension and repair spend</span></div>
</section><h2>Gold agreement</h2><table><thead><tr><th>Judge</th><th>5 probes</th>
<th>10 probes</th><th>Change</th><th>Final</th><th>5→10 rank τ</th><th>Spend</th>
</tr></thead><tbody>{rows}</tbody></table><figure>{trajectory}
<figcaption>Pairwise agreement with the reported-score ordering. Probe counts 6–7
are the earlier adaptive run; counts 11–12 are new adaptive rounds after the
ten-probe opening.</figcaption></figure></main></body></html>"""


def _trajectory_svg(conditions: Sequence[Mapping[str, Any]]) -> str:
    width, height = 900, 360
    left, right, top, bottom = 72, 24, 25, 52
    plot_width, plot_height = width - left - right, height - top - bottom
    probe_counts = sorted(
        {
            row["probe_count"]
            for condition in conditions
            for key in ("source_checkpoints", "extension_checkpoints")
            for row in condition[key]
        }
    )
    x_positions = {
        value: left + index * plot_width / max(1, len(probe_counts) - 1)
        for index, value in enumerate(probe_counts)
    }
    y_min, y_max = 0.75, 0.9

    def y(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    colors = ("#176c8c", "#bf513b")
    lines = []
    for tick in (0.75, 0.8, 0.85, 0.9):
        position = y(tick)
        lines.append(
            f"<line x1='{left}' y1='{position:.1f}' x2='{width-right}' "
            f"y2='{position:.1f}' stroke='#d9e0e3'/><text x='{left-10}' "
            f"y='{position+4:.1f}' text-anchor='end' fill='#66747b'>{tick:.0%}</text>"
        )
    for index, condition in enumerate(conditions):
        color = colors[index % len(colors)]
        for key, dash in (("source_checkpoints", "5 5"), ("extension_checkpoints", "")):
            points = " ".join(
                f"{x_positions[row['probe_count']]:.1f},{y(row['pairwise_accuracy']):.1f}"
                for row in condition[key]
            )
            lines.append(
                f"<polyline points='{points}' fill='none' stroke='{color}' "
                f"stroke-width='3' stroke-dasharray='{dash}'/>"
            )
        lines.append(
            f"<text x='{width-right}' y='{top + index*20 + 12}' "
            f"text-anchor='end' fill='{color}' font-weight='700'>"
            f"{escape(condition['label'])}</text>"
        )
    for value, position in x_positions.items():
        lines.append(
            f"<text x='{position:.1f}' y='{height-20}' text-anchor='middle' "
            f"fill='#66747b'>{value}</text>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        "aria-label='Pairwise accuracy by probe count'>"
        + "".join(lines)
        + "</svg>"
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
