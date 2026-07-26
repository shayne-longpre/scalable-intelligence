from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from ai_council.analysis import analyze_run
from ai_council.oversight_analysis import discover_study_runs, judgment_metrics
from ai_council.rankings import kendall_tau_between


def compare_rankings(
    source: Sequence[str],
    replay: Sequence[str],
) -> dict[str, Any]:
    source_positions = {participant_id: index for index, participant_id in enumerate(source)}
    replay_positions = {participant_id: index for index, participant_id in enumerate(replay)}
    common = [participant_id for participant_id in source if participant_id in replay_positions]
    return {
        "kendall_tau": kendall_tau_between(list(source), list(replay)),
        "top_rank_stable": bool(source and replay and source[0] == replay[0]),
        "top_three_overlap": len(set(source[:3]) & set(replay[:3])),
        "mean_absolute_displacement": (
            mean(
                abs(source_positions[participant_id] - replay_positions[participant_id])
                for participant_id in common
            )
            if common
            else None
        ),
    }


def exact_evidence_match(
    source_transcript: Sequence[Mapping[str, Any]],
    replay_transcript: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source = _evidence_by_stream(source_transcript)
    replay = _evidence_by_stream(replay_transcript)
    shared = source.keys() & replay.keys()
    mismatched = sorted(
        stream_id for stream_id in shared if source[stream_id] != replay[stream_id]
    )
    return {
        "exact": source.keys() == replay.keys() and not mismatched,
        "source_item_count": len(source),
        "replay_item_count": len(replay),
        "missing_stream_ids": sorted(source.keys() - replay.keys()),
        "extra_stream_ids": sorted(replay.keys() - source.keys()),
        "mismatched_stream_ids": mismatched,
    }


def build_order_replay_report(
    *,
    study_path: str | Path,
    runs_root: str | Path,
    output_dir: str | Path,
    published_json_path: str | Path | None = None,
) -> dict[str, Any]:
    study_path = Path(study_path)
    study = _load_json(study_path)
    source_study = _load_json(study["source_study"])
    catalog_path = Path(study["catalog"])
    catalog = _load_json(catalog_path)
    catalog_scores = {
        row["provider_model_id"]: float(row["intelligence_score"])
        for row in catalog["models"]
        if row.get("intelligence_score") is not None
    }
    source_conditions = {
        condition["id"]: condition for condition in source_study["conditions"]
    }
    replay_runs = discover_study_runs(study, runs_root, study_path=study_path)
    missing = [
        condition["id"]
        for condition in study["conditions"]
        if condition["id"] not in replay_runs
    ]
    if missing:
        raise ValueError(f"completed order replays are missing: {missing}")

    conditions = [
        _analyze_replay(
            condition=condition,
            source_condition=source_conditions[condition["source_condition_id"]],
            replay_run=replay_runs[condition["id"]],
            catalog_path=catalog_path,
            catalog_scores=catalog_scores,
        )
        for condition in study["conditions"]
    ]
    taus = [
        condition["ranking_stability"]["kendall_tau"]
        for condition in conditions
        if condition["ranking_stability"]["kendall_tau"] is not None
    ]
    summary = {
        "schema_version": "order-replay-report-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study": study["name"],
        "research_question": study["research_question"],
        "study_file": str(study_path),
        "conditions": conditions,
        "aggregate": {
            "condition_count": len(conditions),
            "mean_kendall_tau": mean(taus) if taus else None,
            "top_rank_stable_count": sum(
                condition["ranking_stability"]["top_rank_stable"]
                for condition in conditions
            ),
            "all_evidence_exact": all(
                condition["evidence_identity"]["exact"] for condition in conditions
            ),
            "fresh_candidate_call_count": sum(
                condition["fresh_candidate_call_count"] for condition in conditions
            ),
            "changed_answer_order_count": sum(
                condition["changed_answer_order_count"] for condition in conditions
            ),
            "comparison_count": sum(
                condition["comparison_count"] for condition in conditions
            ),
            "reported_cost_usd": sum(
                condition["reported_cost_usd"] for condition in conditions
            ),
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report_card.html").write_text(
        render_order_replay_report(summary), encoding="utf-8"
    )
    if published_json_path:
        published_path = Path(published_json_path)
        published_path.parent.mkdir(parents=True, exist_ok=True)
        published_path.write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def render_order_replay_report(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    rows = "".join(
        "<tr>"
        f"<th>{escape(condition['judge_model'])}</th>"
        f"<td>{condition['ranking_stability']['kendall_tau']:.2f}</td>"
        f"<td>{'yes' if condition['ranking_stability']['top_rank_stable'] else 'no'}</td>"
        f"<td>{condition['ranking_stability']['top_three_overlap']}/3</td>"
        f"<td>{condition['source_pair_accuracy']:.1%}</td>"
        f"<td>{condition['replay_pair_accuracy']:.1%}</td>"
        f"<td>{condition['pair_accuracy_delta']:+.1%}</td>"
        f"<td>{condition['changed_answer_order_count']}/{condition['comparison_count']}</td>"
        f"<td>{'exact' if condition['evidence_identity']['exact'] else 'mismatch'}</td>"
        f"<td>${condition['reported_cost_usd']:.2f}</td>"
        "</tr>"
        for condition in summary["conditions"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Answer-order replay</title>
  <style>
    body {{ margin:0; color:#172026; font:16px/1.5 system-ui,sans-serif; }}
    main {{ max-width:1100px; margin:0 auto; padding:48px 24px 72px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(2rem,5vw,3.5rem); }}
    p {{ max-width:780px; color:#56636b; }}
    .facts {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr));
      border-block:1px solid #d9dfe2; margin:32px 0; }}
    .fact {{ padding:18px 12px; }}
    .fact strong {{ display:block; font-size:1.6rem; }}
    .fact span {{ color:#637078; font-size:.85rem; }}
    .table-wrap {{ overflow:auto; border:1px solid #d9dfe2; }}
    table {{ width:100%; border-collapse:collapse; min-width:900px; }}
    th,td {{ padding:10px 12px; border-bottom:1px solid #e6eaec; text-align:left; }}
    thead th {{ background:#f4f6f6; font-size:.78rem; text-transform:uppercase; }}
    tbody th {{ max-width:230px; font-size:.9rem; }}
    .note {{ margin-top:18px; font-size:.9rem; }}
    @media (max-width:700px) {{ .facts {{ grid-template-columns:repeat(2,1fr); }} }}
  </style>
</head>
<body><main>
  <h1>Does answer order change the verdict?</h1>
  <p>{escape(summary['research_question'])} Each judge received the exact same
  probes and candidate answers under a new seeded order. Only comparative
  judgments were regenerated.</p>
  <div class="facts">
    <div class="fact"><strong>{aggregate['mean_kendall_tau']:.2f}</strong><span>mean Kendall rank agreement</span></div>
    <div class="fact"><strong>{aggregate['top_rank_stable_count']}/{aggregate['condition_count']}</strong><span>same top-ranked candidate</span></div>
    <div class="fact"><strong>{aggregate['changed_answer_order_count']}/{aggregate['comparison_count']}</strong><span>probe comparisons reordered</span></div>
    <div class="fact"><strong>{aggregate['fresh_candidate_call_count']}</strong><span>fresh candidate calls</span></div>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Judge</th><th>Kendall τ</th><th>Same top</th><th>Top-3 overlap</th>
    <th>Original accuracy</th><th>Replay accuracy</th><th>Change</th>
    <th>Orders changed</th><th>Evidence</th><th>Cost</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  <p class="note">External capability scores are reference measurements, not
  ground truth. This audit measures presentation sensitivity with candidate
  content held fixed.</p>
</main></body></html>
"""


def _analyze_replay(
    *,
    condition: Mapping[str, Any],
    source_condition: Mapping[str, Any],
    replay_run: Path,
    catalog_path: Path,
    catalog_scores: Mapping[str, float],
) -> dict[str, Any]:
    source_run = Path(condition["source_run"])
    source_analysis = _analysis(source_run, catalog_path)
    replay_analysis = _analysis(replay_run, catalog_path)
    source_judgment = max(
        source_analysis["prior_agreement"]["judgments"],
        key=lambda row: (row.get("round_index", 0), row.get("phase", "")),
    )
    replay_judgment = max(
        replay_analysis["prior_agreement"]["judgments"],
        key=lambda row: (row.get("round_index", 0), row.get("phase", "")),
    )
    source_prior = source_analysis["prior_agreement"]
    participant_models = source_prior["participant_model_ids"]
    participant_scores = {
        participant_id: float(score)
        for participant_id, score in source_prior["participant_prior_scores"].items()
    }
    judge_model = source_condition["judge"]["model"]
    judge_score = catalog_scores[judge_model]
    self_participant = next(
        participant_id
        for participant_id, model_id in participant_models.items()
        if model_id == judge_model
    )
    source_metrics = judgment_metrics(
        source_judgment["ranking"],
        participant_scores,
        judge_score,
        self_participant,
    )
    replay_metrics = judgment_metrics(
        replay_judgment["ranking"],
        participant_scores,
        judge_score,
        self_participant,
    )
    source_transcript = _load_jsonl(source_run / "transcript.jsonl")
    replay_transcript = _load_jsonl(replay_run / "transcript.jsonl")
    replay_config = _load_json(replay_run / "config.json")
    source_orders = _comparison_orders(source_transcript)
    replay_orders = _comparison_orders(replay_transcript)
    common_probes = source_orders.keys() & replay_orders.keys()
    run_summary = _load_json(replay_run / "run_summary.json")
    return {
        "id": condition["id"],
        "source_condition_id": condition["source_condition_id"],
        "judge_model": judge_model,
        "source_run": str(source_run),
        "replay_run": str(replay_run),
        "ranking_stability": compare_rankings(
            source_judgment["ranking"], replay_judgment["ranking"]
        ),
        "source_pair_accuracy": source_metrics["pairs"]["overall"]["accuracy"],
        "replay_pair_accuracy": replay_metrics["pairs"]["overall"]["accuracy"],
        "pair_accuracy_delta": (
            replay_metrics["pairs"]["overall"]["accuracy"]
            - source_metrics["pairs"]["overall"]["accuracy"]
        ),
        "source_self_relative_correct": source_metrics["self_relative_correct"],
        "replay_self_relative_correct": replay_metrics["self_relative_correct"],
        "evidence_identity": exact_evidence_match(
            source_transcript, replay_transcript
        ),
        "comparison_count": len(common_probes),
        "changed_answer_order_count": sum(
            source_orders[probe_id] != replay_orders[probe_id]
            for probe_id in common_probes
        ),
        "fresh_candidate_call_count": candidate_call_count(
            run_summary, replay_config
        ),
        "reported_cost_usd": float(run_summary.get("reported_cost_usd", 0)),
        "model_calls": int(run_summary.get("model_calls", 0)),
    }


def _analysis(run_dir: Path, catalog_path: Path) -> dict[str, Any]:
    analysis_path = run_dir / "analysis_summary.json"
    if not analysis_path.exists():
        analyze_run(run_dir, prior_ranking_file=catalog_path)
    return _load_json(analysis_path)


def _evidence_by_stream(
    transcript: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    evidence = {}
    for row in transcript:
        metadata = row.get("metadata", {})
        if metadata.get("interaction_role") not in {"probe", "answer"}:
            continue
        stream_id = metadata.get("stream_id")
        if isinstance(stream_id, str):
            evidence[stream_id] = str(row.get("content", ""))
    return evidence


def _comparison_orders(
    transcript: Sequence[Mapping[str, Any]],
) -> dict[str, list[str]]:
    return {
        row["metadata"]["probe_id"]: row["metadata"]["answer_presentation_order"]
        for row in transcript
        if row.get("metadata", {}).get("interaction_role") == "probe_comparison"
        and isinstance(row.get("metadata", {}).get("probe_id"), str)
        and isinstance(row.get("metadata", {}).get("answer_presentation_order"), list)
    }


def candidate_call_count(
    run_summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> int:
    participant_models = {
        row.get("model")
        for row in config.get("participants", [])
        if isinstance(row, Mapping) and isinstance(row.get("model"), str)
    }
    spend = run_summary.get("model_spend", {})
    return sum(
        int(spend.get(model_ref, {}).get("model_calls", 0))
        for model_ref in participant_models
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
