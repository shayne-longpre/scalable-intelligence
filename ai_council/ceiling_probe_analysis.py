from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from ai_council.analysis import analyze_run
from ai_council.probe_study import analyze_probe_run
from ai_council.taxonomy import load_taxonomy


SCHEMA_VERSION = "ceiling-probe-extension-analysis-v1"


def build_ceiling_probe_report(
    *,
    study_path: str | Path,
    runs_root: str | Path,
    catalog_path: str | Path,
    output_dir: str | Path,
    published_json_path: str | Path | None = None,
) -> dict[str, Any]:
    study_path = Path(study_path)
    study = _load_json(study_path)
    catalog_path = Path(catalog_path)
    catalog = _load_json(catalog_path)
    catalog_rows = {
        row["provider_model_id"]: row for row in catalog.get("models", [])
    }
    taxonomy = load_taxonomy()
    run_dirs = discover_ceiling_extension_runs(study_path, runs_root)
    missing = [
        condition["id"]
        for condition in study["conditions"]
        if condition["id"] not in run_dirs
    ]
    if missing:
        raise ValueError(f"completed ceiling extension runs are missing: {missing}")
    conditions = []
    for condition in study["conditions"]:
        source_run = Path(condition["source_run"])
        extension_run = run_dirs[condition["id"]]
        if not (extension_run / "analysis_summary.json").exists():
            analyze_run(extension_run, prior_ranking_file=catalog_path)
        source_analysis = _load_json(source_run / "analysis_summary.json")
        extension_analysis = _load_json(
            extension_run / "analysis_summary.json"
        )
        extension_probe_record = analyze_probe_run(
            extension_run,
            cohort="ceiling_extension",
            catalog_rows=catalog_rows,
            taxonomy=taxonomy,
        )
        conditions.append(
            summarize_ceiling_condition(
                condition_id=condition["id"],
                source_run=source_run,
                extension_run=extension_run,
                source_analysis=source_analysis,
                extension_analysis=extension_analysis,
                extension_probe_record=extension_probe_record,
            )
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study": study["name"],
        "research_question": study["research_question"],
        "study_file": str(study_path),
        "catalog_file": str(catalog_path),
        "condition_count": len(conditions),
        "conditions": conditions,
        **_aggregate_conditions(conditions),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report_card.html").write_text(
        render_ceiling_probe_report(summary),
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


def discover_ceiling_extension_runs(
    study_path: str | Path,
    runs_root: str | Path,
) -> dict[str, Path]:
    study_path = Path(study_path)
    study = _load_json(study_path)
    expected = {condition["id"] for condition in study["conditions"]}
    matches: dict[str, list[Path]] = {}
    for config_path in Path(runs_root).glob("*/config.json"):
        config = _load_json(config_path)
        metadata = config.get("metadata", {})
        condition_id = metadata.get("study_condition")
        if condition_id not in expected:
            continue
        if not _same_path(metadata.get("study_file"), study_path):
            continue
        summary_path = config_path.parent / "run_summary.json"
        if (
            summary_path.exists()
            and _load_json(summary_path).get("status") == "completed"
        ):
            matches.setdefault(str(condition_id), []).append(config_path.parent)
    duplicates = {
        key: paths for key, paths in matches.items() if len(paths) > 1
    }
    if duplicates:
        raise ValueError(
            "multiple completed runs match ceiling conditions: "
            + ", ".join(
                f"{key} ({len(paths)})"
                for key, paths in sorted(duplicates.items())
            )
        )
    return {key: paths[0] for key, paths in matches.items()}


def summarize_ceiling_condition(
    *,
    condition_id: str,
    source_run: Path,
    extension_run: Path,
    source_analysis: Mapping[str, Any],
    extension_analysis: Mapping[str, Any],
    extension_probe_record: Mapping[str, Any],
) -> dict[str, Any]:
    source_judgments = source_analysis["prior_agreement"]["judgments"]
    extension_judgments = extension_analysis["prior_agreement"]["judgments"]
    if not source_judgments or not extension_judgments:
        raise ValueError(f"{condition_id} has no scored judgments")
    opening = source_judgments[0]
    archived_final = source_judgments[-1]
    extended = extension_judgments[-1]
    config = _load_json(extension_run / "config.json")
    opening_count = int(
        config["metadata"]["archived_opening_probe_count"]
    )
    new_probes = [
        probe
        for probe in extension_probe_record["probes"]
        if int(probe.get("sequence") or 0) > opening_count
    ]
    run_summary = _load_json(extension_run / "run_summary.json")
    question_types = Counter(
        label for probe in new_probes for label in probe["question_types"]
    )
    strategies = Counter(
        label for probe in new_probes for label in probe["strategy_tags"]
    )
    validity = Counter(
        probe["validity"] for probe in new_probes if probe.get("validity")
    )
    return {
        "id": condition_id,
        "judge_model": extension_probe_record["judge_model"],
        "judge_name": extension_probe_record["judge_name"],
        "judge_score": extension_probe_record["judge_score"],
        "candidate_count": extension_probe_record["candidate_count"],
        "source_run": str(source_run),
        "extension_run": str(extension_run),
        "opening_probe_count": opening_count,
        "new_probe_count": len(new_probes),
        "total_probe_count": int(extended.get("judgment_probe_count") or 0),
        "opening": _judgment_metrics(opening),
        "archived_final": _judgment_metrics(archived_final),
        "extended": _judgment_metrics(extended),
        "accuracy_delta_vs_opening": (
            float(extended["pairwise_accuracy"])
            - float(opening["pairwise_accuracy"])
        ),
        "accuracy_delta_vs_archived_final": (
            float(extended["pairwise_accuracy"])
            - float(archived_final["pairwise_accuracy"])
        ),
        "rank_churn_vs_opening": rank_churn(
            opening["ranking"], extended["ranking"]
        ),
        "new_probe_pair_accuracy": pooled_probe_accuracy(new_probes),
        "new_probe_pair_correct": sum(
            float(probe.get("pair_correct") or 0) for probe in new_probes
        ),
        "new_probe_pair_count": sum(
            int(probe.get("pair_count") or 0) for probe in new_probes
        ),
        "new_probe_decided_pair_accuracy": pooled_decided_probe_accuracy(
            new_probes
        ),
        "new_probe_validity_counts": dict(validity),
        "new_probe_informative_rate": (
            validity["informative"] / sum(validity.values())
            if validity
            else None
        ),
        "new_question_type_counts": dict(question_types),
        "new_strategy_counts": dict(strategies),
        "new_probes": [
            {
                key: probe.get(key)
                for key in (
                    "probe_id",
                    "sequence",
                    "title",
                    "question_types",
                    "strategy_tags",
                    "validity",
                    "pair_accuracy",
                    "decided_pair_accuracy",
                    "tie_pair_rate",
                )
            }
            for probe in new_probes
        ],
        "model_calls": int(run_summary["model_calls"]),
        "reported_cost_usd": float(run_summary["reported_cost_usd"]),
    }


def pooled_probe_accuracy(
    probes: Sequence[Mapping[str, Any]],
) -> float | None:
    count = sum(int(probe.get("pair_count") or 0) for probe in probes)
    correct = sum(float(probe.get("pair_correct") or 0) for probe in probes)
    return correct / count if count else None


def pooled_decided_probe_accuracy(
    probes: Sequence[Mapping[str, Any]],
) -> float | None:
    count = sum(
        int(probe.get("decided_pair_count") or 0) for probe in probes
    )
    correct = sum(
        float(probe.get("decided_pair_correct") or 0) for probe in probes
    )
    return correct / count if count else None


def rank_churn(left: Sequence[str], right: Sequence[str]) -> float:
    if (
        set(left) != set(right)
        or len(left) != len(right)
        or len(set(left)) != len(left)
        or len(set(right)) != len(right)
    ):
        raise ValueError("rank churn requires the same unique candidates")
    if not left:
        return 0.0
    positions = {candidate: index for index, candidate in enumerate(left)}
    return mean(
        abs(positions[candidate] - index)
        for index, candidate in enumerate(right)
    )


def render_ceiling_probe_report(summary: Mapping[str, Any]) -> str:
    conditions = summary["conditions"]
    opening_count = _common_count(conditions, "opening_probe_count")
    total_count = _common_count(conditions, "total_probe_count")
    new_count = _common_count(conditions, "new_probe_count")
    condition_rows = "".join(
        "<tr>"
        f"<th>{escape(row['judge_name'])}<span>{row['judge_score']:.1f}</span></th>"
        f"<td>{_pct(row['opening']['pairwise_accuracy'])}</td>"
        f"<td>{_pct(row['archived_final']['pairwise_accuracy'])}</td>"
        f"<td><strong>{_pct(row['extended']['pairwise_accuracy'])}</strong></td>"
        f"<td class='{_delta_class(row['accuracy_delta_vs_opening'])}'>"
        f"{_signed_points(row['accuracy_delta_vs_opening'])}</td>"
        f"<td>{_pct(row['new_probe_pair_accuracy'])}</td>"
        f"<td>{_pct(row['new_probe_informative_rate'])}</td>"
        f"<td>${row['reported_cost_usd']:.2f}</td>"
        "</tr>"
        for row in conditions
    )
    probe_rows = "".join(
        "<tr>"
        f"<td>{escape(row['judge_name'])}</td>"
        f"<td>{probe['sequence']}</td>"
        f"<td>{escape(probe['title'])}</td>"
        f"<td>{escape(', '.join(probe['question_types']) or 'unclassified')}</td>"
        f"<td>{escape(str(probe['validity'] or 'unknown'))}</td>"
        f"<td>{_pct(probe['pair_accuracy'])}</td>"
        "</tr>"
        for row in conditions
        for probe in row["new_probes"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ceiling-aware probe extension</title>
<style>
:root{{--ink:#18201d;--muted:#5d6863;--line:#cbd3cf;--paper:#f7f9f7;
--accent:#0b6b53;--negative:#a33b2f}} *{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,
-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{width:min(1180px,calc(100% - 32px));margin:42px auto 72px}}
h1{{margin:0 0 8px;font:700 36px/1.1 Georgia,serif}} h2{{margin:34px 0 10px;
font:700 22px/1.2 Georgia,serif}} p{{max-width:800px;color:var(--muted)}}
.metrics{{display:flex;gap:30px;flex-wrap:wrap;margin:20px 0}} .metric strong{{
display:block;color:var(--accent);font-size:28px}} .metric span{{color:var(--muted)}}
.table-wrap{{overflow-x:auto;border-top:2px solid var(--ink)}} table{{width:100%;
border-collapse:collapse;background:white}} th,td{{padding:10px 12px;border-bottom:1px
solid var(--line);text-align:left;vertical-align:top}} thead th{{font-size:12px;
text-transform:uppercase;color:var(--muted)}} tbody th span{{display:block;
font-size:12px;font-weight:400;color:var(--muted)}} td strong,.positive{{color:var(--accent)}}
.negative{{color:var(--negative)}} .note{{font-size:13px}} .charts{{display:grid;
grid-template-columns:1fr 1fr;gap:28px;margin:28px 0}} figure{{margin:0}}
svg{{width:100%;height:auto;background:white;border-top:2px solid var(--ink)}}
figcaption{{margin-top:8px;color:var(--muted);font-size:13px}}
@media(max-width:760px){{.charts{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>Do ceiling-aware probes help?</h1>
<p>{escape(summary['research_question'])}</p>
<div class="metrics">
<div class="metric"><strong>{summary['condition_count']}</strong><span>judges</span></div>
<div class="metric"><strong>{_signed_points(summary['mean_accuracy_delta_vs_opening'])}</strong>
<span>mean extended-vs-opening change</span></div>
<div class="metric"><strong>{_pct(summary['new_probe_pair_accuracy'])}</strong>
<span>new-probe pair accuracy</span></div>
<div class="metric"><strong>${summary['reported_cost_usd']:.2f}</strong>
<span>incremental spend</span></div></div>
<div class="charts"><figure>{_accuracy_slope_svg(conditions)}
<figcaption>External-order pair accuracy before and after the new probes.
Rightward movement is improvement.</figcaption></figure>
<figure>{_probe_accuracy_heatmap(conditions)}
<figcaption>Tie-aware external-order accuracy for each newly authored probe.
Chance is 50%; red cells are anticorrelated with the reference ordering.</figcaption>
</figure></div>
<h2>Ranking results</h2><div class="table-wrap"><table><thead><tr>
<th>Judge · external score</th><th>Opening ({opening_count})</th>
<th>Archived final</th><th>Extended ({total_count})</th>
<th>Change</th><th>New-probe pairs</th><th>New informative</th><th>Spend</th>
</tr></thead><tbody>{condition_rows}</tbody></table></div>
<p class="note">The extended condition is an archived unguided opening battery
plus {new_count} new ceiling-aware probes. It is not a homogeneous opening battery.
Candidate identities remain anonymous. The external catalog is a reference
ordering rather than literal ground truth.</p>
<h2>What the judges added</h2><div class="table-wrap"><table><thead><tr>
<th>Judge</th><th>Probe</th><th>Short description</th><th>Question types</th>
<th>Validity</th><th>Pair accuracy</th></tr></thead><tbody>{probe_rows}</tbody>
</table></div>
</main></body></html>"""


def _accuracy_slope_svg(
    conditions: Sequence[Mapping[str, Any]],
) -> str:
    width = 560
    left = 130
    right = 28
    top = 38
    row_height = 42
    bottom = 34
    height = top + row_height * len(conditions) + bottom
    span = width - left - right

    def x(value: float) -> float:
        return left + span * float(value)

    grid = "".join(
        f"<line x1='{x(tick):.1f}' y1='{top - 12}' x2='{x(tick):.1f}' "
        f"y2='{height - bottom + 4}' stroke='#e2e7e4'/>"
        f"<text x='{x(tick):.1f}' y='{height - 8}' text-anchor='middle' "
        f"font-size='11' fill='#5d6863'>{int(tick * 100)}%</text>"
        for tick in (0.25, 0.5, 0.75, 1.0)
    )
    rows = []
    for index, condition in enumerate(
        sorted(conditions, key=lambda row: float(row["judge_score"]))
    ):
        y = top + index * row_height
        opening = float(condition["opening"]["pairwise_accuracy"])
        extended = float(condition["extended"]["pairwise_accuracy"])
        color = "#0b6b53" if extended >= opening else "#a33b2f"
        rows.append(
            f"<text x='{left - 10}' y='{y + 4}' text-anchor='end' "
            f"font-size='12' fill='#18201d'>{escape(condition['judge_name'])}</text>"
            f"<line x1='{x(opening):.1f}' y1='{y}' x2='{x(extended):.1f}' "
            f"y2='{y}' stroke='{color}' stroke-width='3'/>"
            f"<circle cx='{x(opening):.1f}' cy='{y}' r='5' fill='white' "
            f"stroke='#18201d' stroke-width='2'/>"
            f"<circle cx='{x(extended):.1f}' cy='{y}' r='6' fill='{color}'/>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        "aria-label='Opening and extended ranking accuracy by judge'>"
        f"{grid}{''.join(rows)}</svg>"
    )


def _probe_accuracy_heatmap(
    conditions: Sequence[Mapping[str, Any]],
) -> str:
    probe_numbers = sorted(
        {
            int(probe["sequence"])
            for condition in conditions
            for probe in condition["new_probes"]
        }
    )
    left = 130
    top = 42
    cell_width = 74
    row_height = 42
    bottom = 16
    width = max(560, left + cell_width * len(probe_numbers) + 18)
    height = top + row_height * len(conditions) + bottom
    headers = "".join(
        f"<text x='{left + index * cell_width + cell_width / 2:.1f}' y='22' "
        f"text-anchor='middle' font-size='11' fill='#5d6863'>Probe {number}</text>"
        for index, number in enumerate(probe_numbers)
    )
    rows = []
    for row_index, condition in enumerate(
        sorted(conditions, key=lambda row: float(row["judge_score"]))
    ):
        y = top + row_index * row_height
        rows.append(
            f"<text x='{left - 10}' y='{y + 19}' text-anchor='end' "
            f"font-size='12' fill='#18201d'>{escape(condition['judge_name'])}</text>"
        )
        probes = {
            int(probe["sequence"]): probe
            for probe in condition["new_probes"]
        }
        for column, number in enumerate(probe_numbers):
            value = probes.get(number, {}).get("pair_accuracy")
            fill = _accuracy_color(value)
            label = "—" if value is None else f"{100 * float(value):.0f}%"
            x = left + column * cell_width
            rows.append(
                f"<rect x='{x + 2}' y='{y}' width='{cell_width - 4}' "
                f"height='30' fill='{fill}'/>"
                f"<text x='{x + cell_width / 2:.1f}' y='{y + 20}' "
                f"text-anchor='middle' font-size='11' fill='#18201d'>{label}</text>"
            )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' "
        "aria-label='New probe pair accuracy heatmap'>"
        f"{headers}{''.join(rows)}</svg>"
    )


def _accuracy_color(value: Any) -> str:
    if value is None:
        return "#edf0ee"
    parsed = max(0.0, min(1.0, float(value)))
    if parsed < 0.5:
        strength = (0.5 - parsed) / 0.5
        return _mix_hex("#f4f5f3", "#e8a199", strength)
    strength = (parsed - 0.5) / 0.5
    return _mix_hex("#f4f5f3", "#83c6ad", strength)


def _mix_hex(left: str, right: str, weight: float) -> str:
    weight = max(0.0, min(1.0, weight))
    values = []
    for offset in (1, 3, 5):
        start = int(left[offset : offset + 2], 16)
        end = int(right[offset : offset + 2], 16)
        values.append(round(start + (end - start) * weight))
    return "#" + "".join(f"{value:02x}" for value in values)


def _aggregate_conditions(
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    all_new_probes = [
        probe for condition in conditions for probe in condition["new_probes"]
    ]
    return {
        "mean_opening_pair_accuracy": mean(
            row["opening"]["pairwise_accuracy"] for row in conditions
        ),
        "mean_archived_final_pair_accuracy": mean(
            row["archived_final"]["pairwise_accuracy"] for row in conditions
        ),
        "mean_extended_pair_accuracy": mean(
            row["extended"]["pairwise_accuracy"] for row in conditions
        ),
        "mean_accuracy_delta_vs_opening": mean(
            row["accuracy_delta_vs_opening"] for row in conditions
        ),
        "improved_count": sum(
            row["accuracy_delta_vs_opening"] > 0 for row in conditions
        ),
        "unchanged_count": sum(
            row["accuracy_delta_vs_opening"] == 0 for row in conditions
        ),
        "worsened_count": sum(
            row["accuracy_delta_vs_opening"] < 0 for row in conditions
        ),
        "new_probe_pair_accuracy": _weighted_condition_accuracy(
            conditions
        ),
        "reported_cost_usd": sum(
            row["reported_cost_usd"] for row in conditions
        ),
        "model_calls": sum(row["model_calls"] for row in conditions),
        "new_probe_count": len(all_new_probes),
        "new_question_type_counts": dict(
            Counter(
                label
                for probe in all_new_probes
                for label in probe["question_types"]
            )
        ),
        "new_validity_counts": dict(
            Counter(
                probe["validity"]
                for probe in all_new_probes
                if probe.get("validity")
            )
        ),
    }


def _common_count(
    conditions: Sequence[Mapping[str, Any]], key: str
) -> str:
    values = {int(row[key]) for row in conditions}
    if not values:
        return "n/a"
    if len(values) == 1:
        return str(next(iter(values)))
    return "varies"


def _weighted_condition_accuracy(
    conditions: Sequence[Mapping[str, Any]],
) -> float | None:
    pair_count = sum(int(row["new_probe_pair_count"]) for row in conditions)
    pair_correct = sum(
        float(row["new_probe_pair_correct"]) for row in conditions
    )
    return pair_correct / pair_count if pair_count else None


def _judgment_metrics(judgment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: judgment.get(key)
        for key in (
            "ranking",
            "confidence",
            "kendall_tau",
            "spearman_rho",
            "pairwise_accuracy",
            "top1_matches_prior",
            "judgment_probe_count",
        )
    }


def _same_path(left: Any, right: Path) -> bool:
    if not isinstance(left, str):
        return False
    return Path(left).resolve() == right.resolve()


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _signed_points(value: float) -> str:
    return f"{100 * value:+.1f} pp"


def _delta_class(value: float) -> str:
    return "positive" if value > 0 else "negative" if value < 0 else ""
