from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from html import escape as html_escape
import json
from pathlib import Path
from typing import Any

from ai_council.analysis import analyze_run
from ai_council.rankings import kendall_tau_between
from ai_council.report_summary import generate_report_summary
from ai_council.taxonomy import load_taxonomy


def build_report_card(
    run_dirs: list[str | Path],
    *,
    prior_ranking_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    llm_summary_config: str | Path | None = None,
) -> dict[str, Any]:
    cards = [
        _card_for_run(Path(run_dir), prior_ranking_file=prior_ranking_file)
        for run_dir in run_dirs
    ]
    output_path = _output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prior_ranking_file": str(prior_ranking_file) if prior_ranking_file else None,
        "runs": cards,
    }
    judge_conditions = _judge_condition_summary(cards)
    if judge_conditions:
        payload["judge_condition_summary"] = judge_conditions
    paired_comparison = _paired_comparison(cards)
    if paired_comparison:
        payload["paired_comparison"] = paired_comparison
    if llm_summary_config:
        payload["llm_summary"] = generate_report_summary(
            payload,
            config_path=llm_summary_config,
        )
    (output_path / "report_card_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_path / "report_card.md").write_text(
        render_report_card(payload),
        encoding="utf-8",
    )
    (output_path / "report_card.html").write_text(
        render_report_card_html(payload),
        encoding="utf-8",
    )
    payload["output_dir"] = str(output_path)
    return payload


def render_report_card(payload: dict[str, Any]) -> str:
    runs = payload.get("runs", [])
    lines = [
        "# AI Council Run Report Card",
        "",
        "## At A Glance",
        "",
        "| Run | Mode | Models | Rounds | Transcript turns | Events / Q&A | Final agreement | Prior agreement |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for card in runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(card.get("name", "")),
                    _escape_md(card.get("mode", "")),
                    _escape_md(_models_short(card)),
                    str(card.get("structure", {}).get("rounds", "n/a")),
                    str(card.get("turn_count", 0)),
                    _escape_md(_probe_summary(card)),
                    _escape_md(_agreement_summary(card)),
                    _escape_md(_prior_summary(card)),
                ]
            )
            + " |"
        )
    lines.extend(_judge_condition_markdown(payload.get("judge_condition_summary")))
    llm_summary = payload.get("llm_summary")
    if isinstance(llm_summary, dict) and llm_summary.get("content"):
        lines.extend(["", "## LLM Highlights", "", str(llm_summary["content"]).strip(), ""])
    lines.extend(["", "## Dynamics", ""])
    lines.extend(_taxonomy_table(runs))
    lines.extend(["", "## Rankings", ""])
    for card in runs:
        lines.append(f"### {card.get('name')}")
        lines.append("")
        lines.append(f"- Mode: {card.get('mode')}.")
        lines.append(f"- Participants: {_participants_text(card)}.")
        if card.get("judges"):
            lines.append(f"- Independent judges: {_agents_text(card.get('judges', []))}.")
        lines.append(f"- Final rankings: {_final_rankings_text(card)}.")
        lines.append(f"- Ranking churn: {_churn_text(card)}.")
        lines.append(f"- Evolution: {_evolution_text(card)}.")
        lines.append(f"- Reported spend: {_spend_text(card)}.")
        if card.get("prior_expected_order"):
            lines.append(f"- Prior expected order: {' > '.join(card['prior_expected_order'])}.")
        lines.extend(_probe_budget_markdown(card))
        lines.append("")
    lines.extend(_mode_notes(runs))
    return "\n".join(lines).rstrip() + "\n"


def render_report_card_html(payload: dict[str, Any]) -> str:
    runs = payload.get("runs", [])
    title = "AI Council Run Report Card"
    body = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>{_h(title)}</title>",
        "<style>",
        _html_styles(),
        "</style>",
        "</head>",
        "<body>",
        "<main>",
        f"<header class=\"hero\"><p class=\"eyebrow\">Machine Society Evaluation</p><h1>{_h(title)}</h1>"
        f"<p class=\"subtitle\">{len(runs)} run comparison generated {_h(str(payload.get('created_at', '')))}</p></header>",
        _html_at_a_glance(runs),
        _html_model_priors(runs),
        _html_judge_conditions(payload.get("judge_condition_summary")),
        _html_paired_comparison(payload.get("paired_comparison")),
        _html_llm_summary(payload.get("llm_summary")),
        _html_run_sections(runs),
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(body)


def _html_at_a_glance(runs: list[dict[str, Any]]) -> str:
    cards = []
    for card in runs:
        structure = card.get("structure", {})
        cards.append(
            "<article class=\"run-card\">"
            f"<h2>{_h(card.get('mode', 'unknown'))}</h2>"
            f"<p class=\"run-name\">{_h(card.get('name', ''))}</p>"
            + _html_metric_grid(card)
            + f"<p class=\"formula\">{_h(structure.get('formula', ''))}</p>"
            + f"<p class=\"trace-note\">Trace: {_h(card.get('turn_count'))} transcript turns; "
            + f"{_h(card.get('model_calls', 'n/a'))} model calls; "
            + f"{_h(_format_cost(card.get('reported_cost_usd')))} reported cost; "
            + f"{_h(_format_duration(card.get('elapsed_seconds')))} elapsed.</p>"
            + "</article>"
        )
    return "<section><h2>At A Glance</h2><div class=\"run-grid\">" + "".join(cards) + "</div></section>"


def _html_metric_grid(card: dict[str, Any]) -> str:
    agreement = card.get("final_agreement", {})
    prior = card.get("final_prior_agreement", {})
    structure = card.get("structure", {})
    if structure.get("kind") in {"structured round-robin", "independent judge ranking"}:
        metrics = [
            ("Rounds", structure.get("rounds")),
            ("Probes", structure.get("probe_count")),
            ("Expected Q/A", structure.get("routed_qa_expected")),
            ("Observed Q/A", card.get("qa_pair_count")),
            ("Agreement tau", _format_number(agreement.get("mean_pairwise_tau"))),
            ("Prior tau", _format_number(prior.get("mean_tau"))),
        ]
    else:
        metrics = [
            ("Rounds", structure.get("rounds")),
            ("Public Turns", card.get("public_turn_count")),
            ("Events", card.get("probe_event_count")),
            ("Candidate Qs", card.get("candidate_question_turn_count")),
            ("Agreement tau", _format_number(agreement.get("mean_pairwise_tau"))),
            ("Prior tau", _format_number(prior.get("mean_tau"))),
        ]
    return "<div class=\"metrics\">" + "".join(_metric(label, value) for label, value in metrics) + "</div>"


def _html_model_priors(runs: list[dict[str, Any]]) -> str:
    if not runs:
        return ""
    models: dict[str, dict[str, Any]] = {}
    for card in runs:
        prior_ranks = card.get("prior_participant_ranks", {})
        prior_scores = card.get("prior_participant_scores", {})
        reported_participants = set(card.get("prior_reported_score_participants", []))
        prior_ranks = prior_ranks if isinstance(prior_ranks, dict) else {}
        prior_scores = prior_scores if isinstance(prior_scores, dict) else {}
        for participant in card.get("participants", []):
            if not isinstance(participant, dict):
                continue
            provider_model_id = participant.get("provider_model_id")
            if not isinstance(provider_model_id, str):
                continue
            participant_id = participant.get("id")
            models.setdefault(
                provider_model_id,
                {
                    "model_ref": participant.get("model_ref"),
                    "provider_model_id": provider_model_id,
                    "rank": prior_ranks.get(participant_id),
                    "score": prior_scores.get(participant_id),
                    "basis": (
                        "reported"
                        if participant_id in reported_participants
                        else "estimated" if reported_participants else "unspecified"
                    ),
                },
            )
    rows = []
    for model in sorted(models.values(), key=lambda item: item.get("rank") or 10**9):
        rank = model.get("rank")
        score = model.get("score")
        rows.append(
            "<tr>"
            f"<td class=\"id-cell\">{_h(model.get('model_ref'))}</td>"
            f"<td><code>{_h(model.get('provider_model_id'))}</code></td>"
            f"<td>{_h(_format_rank(rank))}</td>"
            f"<td>{_h(_format_number(score))}</td>"
            f"<td>{_h(model.get('basis'))}</td>"
            f"<td>{_h(_rank_bucket(rank))}</td>"
            "</tr>"
        )
    order_text = " > ".join(str(item.get("model_ref")) for item in sorted(
        models.values(), key=lambda item: item.get("rank") or 10**9
    )) or "unavailable"
    judges = _agents_text(runs[0].get("judges", []))
    judge_note = f" Independent judges: <strong>{_h(judges)}</strong>." if judges else ""
    return (
        "<section><h2>Model Priors</h2>"
        f"<p class=\"note\">Lower prior rank means stronger expected general capability. Expected order: <strong>{_h(order_text)}</strong>.{judge_note}</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Model</th><th>Provider model ID</th><th>Prior rank</th><th>Prior score</th><th>Basis</th><th>Prior band</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def _html_judge_conditions(summary: Any) -> str:
    if not isinstance(summary, dict):
        return ""
    accuracy_rows = []
    for item in summary.get("accuracy_by_probe_count", []):
        if not isinstance(item, dict):
            continue
        accuracy_rows.append(
            "<tr>"
            f"<td class=\"id-cell\">{_h(item.get('model_ref'))}</td>"
            f"<td>{_h(item.get('probe_count'))}</td>"
            f"<td>{_h(item.get('n'))}</td>"
            f"<td>{_h(_format_number(item.get('mean_pairwise_accuracy')))}</td>"
            f"<td>{_h(_format_interval(item.get('min_pairwise_accuracy'), item.get('max_pairwise_accuracy')))}</td>"
            f"<td>{_h(_format_number(item.get('mean_kendall_tau')))}</td>"
            f"<td>{_h(_format_number(item.get('top1_rate')))}</td>"
            f"<td>{_h(_format_number(item.get('mean_confidence')))}</td>"
            "</tr>"
        )
    behavior_rows = []
    for item in summary.get("judge_behavior", []):
        if not isinstance(item, dict):
            continue
        validity = item.get("probe_validity", {})
        validity = validity if isinstance(validity, dict) else {}
        behavior_rows.append(
            "<tr>"
            f"<td class=\"id-cell\">{_h(item.get('model_ref'))}</td>"
            f"<td>{_h(validity.get('informative', 0))}</td>"
            f"<td>{_h(validity.get('limited', 0))}</td>"
            f"<td>{_h(validity.get('invalid', 0))}</td>"
            f"<td>{_h(item.get('adaptive_decisions'))}</td>"
            f"<td>{_h(_format_number(item.get('mean_target_size')))}</td>"
            f"<td>{_h(item.get('rank_changes'))}</td>"
            f"<td>{_h(item.get('direct_model_calls'))}</td>"
            f"<td>{_h(_format_cost(item.get('direct_reported_cost_usd')))}</td>"
            "</tr>"
        )
    return (
        "<section><h2>Judge Conditions Across Runs</h2>"
        f"<p class=\"note\">Accepted runs: {_h(summary.get('run_count'))}. "
        f"Mean final inter-judge tau: <strong>{_h(_format_number(summary.get('mean_final_interjudge_tau')))}</strong>. "
        "Accuracy is measured against the external prior; ranges show run-to-run variation.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Judge model</th><th>Probes</th><th>Runs</th><th>Mean pairwise</th><th>Range</th><th>Mean Kendall</th><th>Top-1 rate</th><th>Mean conf.</th>"
        "</tr></thead><tbody>" + "".join(accuracy_rows) + "</tbody></table></div>"
        "<div class=\"table-wrap full\"><table><thead><tr>"
        "<th>Judge model</th><th>Informative</th><th>Limited</th><th>Invalid</th><th>Adaptive decisions</th><th>Mean targets</th><th>Rank changes</th><th>Direct calls</th><th>Direct cost</th>"
        "</tr></thead><tbody>" + "".join(behavior_rows) + "</tbody></table></div>"
        "<p class=\"note\">Direct cost covers judge-model calls only; candidate-answer costs remain in each run's spend table.</p>"
        "</section>"
    )


def _html_paired_comparison(comparison: Any) -> str:
    if not isinstance(comparison, dict) or not comparison.get("participants"):
        return ""
    rows = []
    for item in comparison.get("participants", []):
        if not isinstance(item, dict):
            continue
        by_mode = item.get("by_mode", {})
        by_mode = by_mode if isinstance(by_mode, dict) else {}
        free = by_mode.get("free discussion", {})
        structured = by_mode.get("structured round-robin", {})
        free = free if isinstance(free, dict) else {}
        structured = structured if isinstance(structured, dict) else {}
        rows.append(
            "<tr>"
            f"<td class=\"id-cell\">{_h(item.get('participant_id'))}</td>"
            f"<td>{_h(item.get('model_ref'))}</td>"
            f"<td>{_h(_format_rank(item.get('prior_rank')))}</td>"
            f"<td>{_h(_format_number(free.get('mean_received_rank')))}</td>"
            f"<td>{_h(free.get('top1_votes', 'n/a'))}</td>"
            f"<td>{_h(_format_number(structured.get('mean_received_rank')))}</td>"
            f"<td>{_h(structured.get('top1_votes', 'n/a'))}</td>"
            "</tr>"
        )
    return (
        "<section><h2>Paired Mode Comparison</h2>"
        "<p class=\"note\">Lower received rank is better. Top-1 votes count how many final judges ranked that participant first in each mode.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Participant</th><th>Model</th><th>Prior rank</th>"
        "<th>Free avg rank</th><th>Free top-1</th>"
        "<th>Structured avg rank</th><th>Structured top-1</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def _html_llm_summary(summary: Any) -> str:
    if not isinstance(summary, dict) or not summary.get("content"):
        return ""
    return (
        "<section><h2>LLM Highlights</h2>"
        "<article class=\"panel full summary-panel\">"
        f"{_html_text_block(summary.get('content'))}"
        f"<p class=\"note\">Summary model: {_h(summary.get('model', 'unknown'))}</p>"
        "</article></section>"
    )


def _html_run_sections(runs: list[dict[str, Any]]) -> str:
    sections = []
    for card in runs:
        taxonomy = card.get("taxonomy_counts", {})
        qtypes = taxonomy.get("question_type_frequency", {}) if isinstance(taxonomy, dict) else {}
        strategies = taxonomy.get("signal_frequency", {}) if isinstance(taxonomy, dict) else {}
        sections.append(
            "<section class=\"run-detail\">"
            f"<h2>{_h(card.get('mode', 'unknown'))}: {_h(card.get('name', ''))}</h2>"
            "<div class=\"two-col\">"
            + _html_ranking_block(card)
            + _html_probe_budget_block(card)
            + _html_gap_accuracy_block(card)
            + _html_probe_comparisons_block(card)
            + _html_adaptive_trace_block(card)
            + _html_quality_gate_block(card)
            + _html_highlights_block(card)
            + _html_spend_block(card)
            + "</div>"
            "<div class=\"three-col\">"
            + _html_bar_block("Question Types", qtypes)
            + _html_bar_block("Evaluation Strategies", strategies)
            + _html_bar_block("Turn Dynamics", card.get("transition_counts", {}))
            + "</div>"
            + _html_round_table(card)
            + _html_timeline(card)
            + "</section>"
        )
    return "".join(sections)


def _html_ranking_block(card: dict[str, Any]) -> str:
    agreement = card.get("final_agreement", {})
    prior = card.get("final_prior_agreement", {})
    agreement_label = (
        "Judge agreement"
        if card.get("mode") == "independent judge ranking"
        else "Participant agreement"
    )
    rows = []
    for item in card.get("final_rankings", []):
        ranking = item.get("ranking") if isinstance(item, dict) else None
        if not isinstance(ranking, list):
            continue
        rows.append(
            "<tr>"
            f"<td class=\"id-cell\">{_h(item.get('speaker'))}</td>"
            f"<td>{_h(' > '.join(str(value) for value in ranking))}</td>"
            f"<td>{_h(_format_number(item.get('confidence')))}</td>"
            "</tr>"
        )
    return (
        "<article class=\"panel\"><h3>Final Rankings</h3>"
        f"<p class=\"note\">{agreement_label} tau: <strong>{_h(_format_number(agreement.get('mean_pairwise_tau')))}</strong>; "
        f"prior agreement tau: <strong>{_h(_format_number(prior.get('mean_tau')))}</strong> "
        f"({_h(prior.get('basis', 'all scored candidates'))}).</p>"
        "<table><thead><tr><th>Judge</th><th>Ranking</th><th>Conf.</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        f"<p class=\"note\">Churn: {_h(_churn_text(card))}</p>"
        "</article>"
    )


def _html_probe_budget_block(card: dict[str, Any]) -> str:
    results = card.get("probe_budget_results", [])
    if not isinstance(results, list) or not results:
        return ""
    rows = []
    adaptive = bool(card.get("structure", {}).get("probe_schedule"))
    for item in results:
        if not isinstance(item, dict):
            continue
        round_cell = f"<td>{_h(item.get('round_index'))}</td>" if adaptive else ""
        rows.append(
            "<tr>"
            f"<td class=\"id-cell\">{_h(item.get('speaker'))}</td>"
            f"{round_cell}"
            f"<td>{_h(item.get('probe_count'))}</td>"
            f"<td>{_h(_format_number(item.get('kendall_tau')))}</td>"
            f"<td>{_h(_format_number(item.get('spearman_rho')))}</td>"
            f"<td>{_h(_format_number(item.get('pairwise_accuracy')))}</td>"
            f"<td>{_h(_format_number(item.get('confidence')))}</td>"
            f"<td>{_h(' > '.join(str(value) for value in item.get('ranking', [])))}</td>"
            f"<td>{_h(_format_number(item.get('rank_score_r_squared')))}</td>"
            "</tr>"
        )
    title = "Round-by-Round Ranking" if adaptive else "Probe Budget Ablation"
    note = (
        "Each row is the cumulative ranking checkpoint after that many probes. "
        "Later adaptive probes may cover only the judge-selected comparison set."
        if adaptive
        else "Each row is a fresh judgment using only the first N probes; candidate answers are shared across branches."
    )
    return (
        f"<article class=\"panel full\"><h3>{_h(title)}</h3>"
        f"<p class=\"note\">{_h(note)}</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Judge</th>"
        + ("<th>Round</th>" if adaptive else "")
        + "<th>Cumulative probes</th><th>Kendall</th><th>Spearman</th><th>Pairwise</th><th>Conf.</th><th>Ranking</th><th>Rank-score R2</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )


def _html_gap_accuracy_block(card: dict[str, Any]) -> str:
    results = [
        item
        for item in card.get("probe_budget_results", [])
        if isinstance(item, dict)
        and isinstance(item.get("pairwise_accuracy_by_score_gap"), list)
    ]
    if not results:
        return ""
    labels = [
        str(item.get("label"))
        for item in results[0]["pairwise_accuracy_by_score_gap"]
        if isinstance(item, dict)
    ]
    rows = []
    for result in results:
        by_label = {
            str(item.get("label")): item
            for item in result["pairwise_accuracy_by_score_gap"]
            if isinstance(item, dict)
        }
        cells = []
        for label in labels:
            item = by_label.get(label, {})
            cells.append(
                f"<td>{_h(_format_number(item.get('accuracy')))} "
                f"<span class=\"note\">(n={_h(item.get('pair_count', 0))})</span></td>"
            )
        rows.append(
            "<tr>"
            f"<td>{_h(result.get('round_index'))}</td>"
            f"<td>{_h(result.get('probe_count'))}</td>"
            + "".join(cells)
            + "</tr>"
        )
    return (
        "<article class=\"panel full\"><h3>Discrimination By Capability Gap</h3>"
        "<p class=\"note\">Pairwise accuracy on the reported-score subset. "
        "Larger score gaps should be easier to order.</p>"
        "<div class=\"table-wrap\"><table><thead><tr><th>Round</th>"
        "<th>Probes</th>"
        + "".join(f"<th>{_h(label)} points</th>" for label in labels)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )
def _html_probe_comparisons_block(card: dict[str, Any]) -> str:
    comparisons = card.get("probe_comparisons", [])
    if not isinstance(comparisons, list) or not comparisons:
        return ""
    rows = []
    for record in comparisons:
        if not isinstance(record, dict):
            continue
        parsed = record.get("parsed")
        parsed = parsed if isinstance(parsed, dict) else {}
        ordering = parsed.get("ordering")
        ordering_text = " > ".join(str(value) for value in ordering) if isinstance(ordering, list) else ""
        rows.append(
            "<tr>"
            f"<td>{_h(record.get('round_index'))}</td>"
            f"<td>{_h(record.get('probe_sequence_number'))}</td>"
            f"<td>{_h(parsed.get('probe_validity'))}</td>"
            f"<td>{_h(_format_number(parsed.get('confidence')))}</td>"
            f"<td>{_h(ordering_text)}</td>"
            f"<td class=\"trace-note\">turn {_h(record.get('turn_id'))}</td>"
            "</tr>"
        )
    return (
        "<article class=\"panel full\"><h3>Per-Probe Comparisons</h3>"
        "<p class=\"note\">These are direct within-probe orderings before evidence is merged across probes.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Round</th><th>Probe</th><th>Validity</th><th>Conf.</th><th>Ordering</th><th>Source</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )


def _html_adaptive_trace_block(card: dict[str, Any]) -> str:
    adaptive = card.get("adaptive_metrics", {})
    trace = adaptive.get("decision_trace", []) if isinstance(adaptive, dict) else []
    if not isinstance(trace, list) or not trace:
        return ""
    rows = []
    for decision in trace:
        if not isinstance(decision, dict):
            continue
        strategy = decision.get("planned_strategy", [])
        strategy_text = "; ".join(str(value) for value in strategy[:2]) if isinstance(strategy, list) else ""
        question_types = decision.get("question_types", [])
        question_type_text = (
            ", ".join(_labelize(value) for value in question_types)
            if isinstance(question_types, list)
            else ""
        )
        rows.append(
            "<tr>"
            f"<td>{_h(decision.get('round_index'))}</td>"
            f"<td>{_h(decision.get('probe_sequence_number'))}</td>"
            f"<td>{_h(', '.join(str(value) for value in decision.get('actual_candidates', [])))}</td>"
            f"<td>{_h(_labelize(decision.get('target_change')))}</td>"
            f"<td>{_h(question_type_text)}</td>"
            f"<td>{_h(decision.get('probe_validity'))}</td>"
            f"<td>{_h(_labelize(decision.get('evidence_outcome')))}</td>"
            f"<td>{_h('yes' if decision.get('ranking_changed') else 'no')}</td>"
            f"<td>{_h(_format_number(decision.get('confidence_delta')))}</td>"
            f"<td class=\"trace-note\">{_h(strategy_text)}</td>"
            "</tr>"
        )
    return (
        "<article class=\"panel full\"><h3>Adaptive Decision Trace</h3>"
        "<p class=\"note\">Targeting and outcomes are linked to the prior judgment and current probe. Validity is judge-reported; rank change is deterministic.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Round</th><th>Probe</th><th>Targets</th><th>Target change</th>"
        "<th>Question type</th><th>Validity</th><th>Outcome</th><th>Rank changed</th>"
        "<th>Conf. delta</th><th>Planned strategy</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div></article>"
    )


def _html_highlights_block(card: dict[str, Any]) -> str:
    items = []
    for highlight in card.get("highlights", []):
        qtypes = ", ".join(_labelize(item) for item in highlight.get("question_types", []))
        qtype_text = f"<span class=\"tag\">{_h(qtypes)}</span>" if qtypes else ""
        label = _time_label(card, highlight)
        trace = _trace_label(highlight)
        items.append(
            "<li>"
            f"<div><strong>{_h(label)}</strong>{trace} "
            f"{_h(highlight.get('speaker'))} to {_h(highlight.get('target'))} {qtype_text}</div>"
            f"<p>{_h(highlight.get('text', ''))}</p>"
            "</li>"
        )
    if not items:
        items.append("<li>No highlight candidates extracted.</li>")
    return (
        "<article class=\"panel\"><h3>Question Highlights</h3>"
        "<p class=\"note\">Representative probe excerpts selected deterministically.</p>"
        "<ol class=\"highlights\">"
        + "".join(items)
        + "</ol></article>"
    )


def _html_quality_gate_block(card: dict[str, Any]) -> str:
    quality = card.get("quality_gates", {})
    summary = quality.get("summary", {}) if isinstance(quality, dict) else {}
    codes = summary.get("codes", {}) if isinstance(summary, dict) else {}
    severities = summary.get("severities", {}) if isinstance(summary, dict) else {}
    rows = []
    if isinstance(codes, dict):
        for code, count in Counter({str(key): int(value) for key, value in codes.items()}).most_common(6):
            rows.append(
                "<tr>"
                f"<td>{_h(_labelize(code))}</td>"
                f"<td>{_h(count)}</td>"
                "</tr>"
            )
    if not rows:
        rows.append("<tr><td>No audit flags</td><td>0</td></tr>")
    findings = card.get("audit_findings", [])
    detail_items = []
    for finding in findings[:5] if isinstance(findings, list) else []:
        if not isinstance(finding, dict):
            continue
        detail_items.append(
            "<li>"
            f"<strong>{_h(_labelize(finding.get('code')))}</strong> "
            f"<span class=\"tag\">{_h(finding.get('severity'))}</span>"
            f"<p>{_h(finding.get('message'))}</p>"
            f"<p class=\"trace-note\">{_h(finding.get('evidence', ''))}</p>"
            "</li>"
        )
    details = "<ol class=\"audit-list\">" + "".join(detail_items) + "</ol>" if detail_items else ""
    return (
        "<article class=\"panel\"><h3>Quality Gates</h3>"
        f"<p class=\"note\">Flags: <strong>{_h(summary.get('finding_count', 0))}</strong>; "
        f"errors: <strong>{_h(severities.get('error', 0) if isinstance(severities, dict) else 0)}</strong>; "
        f"warnings: <strong>{_h(severities.get('warning', 0) if isinstance(severities, dict) else 0)}</strong>.</p>"
        "<table><thead><tr><th>Flag</th><th>Count</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        + details
        + "</article>"
    )


def _html_spend_block(card: dict[str, Any]) -> str:
    model_spend = card.get("model_spend", {})
    rows = []
    if isinstance(model_spend, dict):
        for model_ref, item in sorted(model_spend.items()):
            if not isinstance(item, dict):
                continue
            rows.append(
                "<tr>"
                f"<td class=\"id-cell\">{_h(model_ref)}</td>"
                f"<td><code>{_h(item.get('provider_model_id'))}</code></td>"
                f"<td>{_h(item.get('model_calls', 0))}</td>"
                f"<td>{_h(_format_cost(item.get('reported_cost_usd')))}</td>"
                "</tr>"
            )
    if not rows:
        rows.append("<tr><td colspan=\"4\">No per-model cost data.</td></tr>")
    run_count = card.get("spend_run_count", 1)
    lineage_note = (
        f" Cumulative across {run_count} replay checkpoints."
        if isinstance(run_count, int) and run_count > 1
        else ""
    )
    if card.get("spend_lineage_complete") is False:
        lineage_note += " This is a minimum recorded total; at least one source summary is unavailable."
    return (
        "<article class=\"panel\"><h3>Reported Spend</h3>"
        "<p class=\"note\">Includes only costs returned by the provider."
        f"{_h(lineage_note)}</p>"
        "<table><thead><tr><th>Model</th><th>Provider ID</th><th>Calls</th><th>Cost</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></article>"
    )


def _html_bar_block(title: str, counts: Any, limit: int = 8) -> str:
    rows = _bar_rows(counts, limit=limit)
    if not rows:
        rows = "<p class=\"note\">None detected.</p>"
    return f"<article class=\"panel\"><h3>{_h(title)}</h3>{rows}</article>"


def _html_round_table(card: dict[str, Any]) -> str:
    round_counts = card.get("question_types_by_round", {})
    if not isinstance(round_counts, dict) or not round_counts:
        return ""
    rows = []
    for round_key, counts in sorted(round_counts.items(), key=lambda item: _round_sort_key(item[0])):
        rows.append(
            "<tr>"
            f"<td>{_h(_round_display_label(round_key))}</td>"
            f"<td>{_h(_top_counts(counts, limit=5))}</td>"
            f"<td>{_h(_top_counts(card.get('strategies_by_round', {}).get(round_key, {}), limit=5))}</td>"
            "</tr>"
        )
    return (
        "<article class=\"panel full\"><h3>Round Movement</h3>"
        "<table><thead><tr><th>Round</th><th>Top question types</th><th>Top strategies</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></article>"
    )


def _html_timeline(card: dict[str, Any]) -> str:
    events = card.get("event_timeline", [])
    if not isinstance(events, list) or not events:
        return ""
    items = []
    for event in events:
        label = _time_label(card, event)
        trace = _trace_label(event)
        items.append(
            "<li>"
            f"<div class=\"timeline-meta\">{_h(label)}{trace} · {_h(event.get('speaker'))} · "
            f"{_h(event.get('phase'))} · <span>{_h(_labelize(event.get('transition_label')))}</span>"
            f" · {_h(_labelize(event.get('dependency_semantics')))}</div>"
            f"<div class=\"timeline-tags\">{_tag_html(event.get('primary_question_type'))}{_tag_html(event.get('primary_strategy'))}</div>"
            f"<p>{_h(event.get('excerpt', ''))}</p>"
            "</li>"
        )
    return (
        "<article class=\"panel full\"><h3>Conversation Flow</h3>"
        "<ol class=\"timeline\">"
        + "".join(items)
        + "</ol></article>"
    )


def _time_label(card: dict[str, Any], item: dict[str, Any]) -> str:
    round_index = item.get("round_index")
    if card.get("mode") in {"structured round-robin", "independent judge ranking"} and round_index is not None:
        return f"Round {round_index}"
    phase_label = _phase_round_label(item.get("phase"))
    if phase_label:
        return phase_label
    if round_index is not None:
        return f"Round {round_index}"
    return f"Turn {item.get('turn_id', 'n/a')}"


def _trace_label(item: dict[str, Any]) -> str:
    turn_id = item.get("turn_id")
    if turn_id is None:
        return ""
    return f" <span class=\"trace\">turn {_h(turn_id)}</span>"


def _bar_rows(counts: Any, limit: int) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    items = Counter({str(key): int(value) for key, value in counts.items()}).most_common(limit)
    max_value = max(value for _, value in items) if items else 1
    rows = []
    for key, value in items:
        width = 100 * value / max_value if max_value else 0
        rows.append(
            "<div class=\"bar-row\">"
            f"<span class=\"bar-label\">{_h(_labelize(key))}</span>"
            "<span class=\"bar-track\">"
            f"<span class=\"bar-fill\" style=\"width:{width:.1f}%\"></span>"
            "</span>"
            f"<span class=\"bar-value\">{value}</span>"
            "</div>"
        )
    return "".join(rows)


def _metric(label: str, value: Any) -> str:
    return (
        "<div class=\"metric\">"
        f"<span>{_h(label)}</span>"
        f"<strong>{_h(value)}</strong>"
        "</div>"
    )


def _tag_html(value: Any) -> str:
    if not value:
        return ""
    return f"<span class=\"tag\">{_h(_labelize(value))}</span>"


def _html_text_block(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    items = []
    paragraphs = []
    for line in lines:
        if line.startswith(("- ", "* ")):
            items.append(f"<li>{_h(line[2:].strip())}</li>")
        else:
            paragraphs.append(f"<p>{_h(line)}</p>")
    html = "".join(paragraphs)
    if items:
        html += "<ul>" + "".join(items) + "</ul>"
    return html


def _html_styles() -> str:
    return """
:root { color-scheme: light; --ink:#17201b; --muted:#607066; --line:#d9e1da; --soft:#f5f7f3; --panel:#ffffff; --accent:#2f6f5e; --accent2:#8a5f1f; }
* { box-sizing: border-box; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#eef2ed; }
main { max-width:1180px; margin:0 auto; padding:32px 24px 56px; }
.hero { padding:28px 0 18px; }
.eyebrow { margin:0 0 8px; text-transform:uppercase; letter-spacing:.08em; color:var(--accent); font-size:12px; font-weight:700; }
h1 { margin:0; font-size:38px; line-height:1.05; letter-spacing:0; }
h2 { margin:0 0 16px; font-size:24px; letter-spacing:0; }
h3 { margin:0 0 12px; font-size:16px; letter-spacing:0; }
.subtitle, .note, .formula, .run-name { color:var(--muted); }
.trace-note, .trace { color:var(--muted); font-size:12px; font-weight:500; }
.trace { margin-left:4px; }
section { margin-top:28px; }
.run-grid, .two-col, .three-col { display:grid; gap:16px; }
.run-grid { grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }
.two-col { grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); }
.three-col { grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); margin-top:16px; }
.run-card, .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.full { margin-top:16px; }
.metrics { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:14px 0; }
.metric { background:var(--soft); border:1px solid var(--line); border-radius:6px; padding:10px; min-height:64px; }
.metric span { display:block; color:var(--muted); font-size:12px; margin-bottom:6px; }
.metric strong { font-size:20px; }
.table-wrap { overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:8px; }
table { width:100%; border-collapse:collapse; font-size:14px; }
th, td { text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
th { background:var(--soft); font-size:12px; color:var(--muted); text-transform:uppercase; }
code { font-size:12px; white-space:nowrap; }
.id-cell { font-weight:700; color:var(--accent); }
.bar-row { display:grid; grid-template-columns:minmax(120px,1fr) 3fr 36px; gap:10px; align-items:center; margin:8px 0; font-size:13px; }
.bar-track { height:10px; background:#e7ece7; border-radius:999px; overflow:hidden; }
.bar-fill { display:block; height:100%; background:linear-gradient(90deg,var(--accent),#68a391); }
.bar-value { color:var(--muted); text-align:right; }
.tag { display:inline-block; border:1px solid var(--line); background:var(--soft); border-radius:999px; padding:2px 7px; margin:2px 4px 2px 0; font-size:12px; color:var(--muted); }
.highlights { padding-left:20px; }
.highlights li { margin:0 0 14px; }
.audit-list { padding-left:20px; margin:12px 0 0; }
.audit-list li { margin:0 0 10px; }
.audit-list p { margin:4px 0 0; color:var(--muted); }
.summary-panel ul { margin:0; padding-left:20px; }
.summary-panel li { margin:0 0 8px; }
.summary-panel p { margin:0 0 10px; }
.highlights p, .timeline p { margin:6px 0 0; color:var(--muted); }
.timeline { list-style:none; margin:0; padding:0; border-left:3px solid var(--line); }
.timeline li { position:relative; margin:0 0 14px 18px; padding:12px 14px; background:var(--soft); border-radius:6px; }
.timeline li:before { content:""; position:absolute; left:-27px; top:16px; width:12px; height:12px; border-radius:50%; background:var(--accent2); }
.timeline-meta { font-weight:700; font-size:13px; }
.timeline-meta span { color:var(--accent2); }
.timeline-tags { margin-top:6px; }
@media (max-width:720px) { main { padding:22px 14px 40px; } h1 { font-size:30px; } .metrics { grid-template-columns:repeat(2,1fr); } }
"""


def _card_for_run(run_dir: Path, *, prior_ranking_file: str | Path | None) -> dict[str, Any]:
    summary = analyze_run(run_dir, prior_ranking_file=prior_ranking_file)
    config = _load_json(run_dir / "config.json")
    run_summary = _load_json(run_dir / "run_summary.json")
    run_metrics = _load_json(run_dir / "run_metrics.json")
    extraction = _load_json(run_dir / "posthoc_extraction.json")
    behavior_audit = _load_json(run_dir / "behavior_audit.json")
    metrics = summary.get("metrics", {})
    prior = summary.get("prior_agreement", {})
    transcript_taxonomy = _compact_taxonomy_counts(summary.get("taxonomy", {}))
    probe_taxonomy = _probe_taxonomy_counts(
        summary.get("posthoc_extraction", {}),
        transcript_taxonomy,
    )
    experiment_spend = summary.get("experiment_spend", {})
    cumulative_spend = (
        experiment_spend
        if isinstance(experiment_spend, dict)
        and experiment_spend.get("run_count", 0) > 1
        else {}
    )
    return {
        "run_dir": str(run_dir),
        "name": config.get("name", run_dir.name),
        "mode": _mode(config),
        "structure": _structure(config),
        "participants": _participants(config),
        "judges": _agent_rows(config, "judges"),
        "turn_count": summary.get("turn_count", 0),
        "public_turn_count": summary.get("public_turn_count", 0),
        "private_turn_count": summary.get("private_turn_count", 0),
        "model_calls": cumulative_spend.get(
            "model_calls",
            run_summary.get("model_calls"),
        ),
        "incremental_model_calls": run_summary.get("model_calls"),
        "elapsed_seconds": run_summary.get("elapsed_seconds"),
        "reported_cost_usd": cumulative_spend.get(
            "reported_cost_usd",
            run_summary.get("reported_cost_usd"),
        ),
        "incremental_reported_cost_usd": run_summary.get("reported_cost_usd"),
        "spend_run_count": cumulative_spend.get("run_count", 1),
        "spend_lineage_complete": cumulative_spend.get("complete", True),
        "repair": _repair_metadata(config),
        "model_spend": cumulative_spend.get(
            "model_spend",
            summary.get("model_spend", {}),
        ),
        "qa_pair_count": summary.get("posthoc_extraction", {}).get("qa_pair_count", 0),
        "candidate_question_turn_count": summary.get("posthoc_extraction", {}).get(
            "candidate_question_turn_count", 0
        ),
        "probe_event_count": metrics.get("evolution", {}).get("probe_event_count", 0),
        "question_types_by_round": metrics.get("evolution", {}).get("question_types_by_round", {}),
        "strategies_by_round": metrics.get("evolution", {}).get("strategies_by_round", {}),
        "transition_counts": metrics.get("evolution", {}).get("transition_counts", {}),
        "topical_transition_counts": metrics.get("evolution", {}).get(
            "topical_transition_counts",
            {},
        ),
        "dependency_counts": metrics.get("evolution", {}).get("dependency_counts", {}),
        "adaptive_metrics": run_metrics.get("evolution", {}).get("adaptive", {}),
        "final_rankings": metrics.get("rankings", {}).get("final_rankings", []),
        "final_agreement": metrics.get("rankings", {}).get("final_agreement", {}),
        "churn_by_speaker": metrics.get("rankings", {}).get("churn_by_speaker", {}),
        "prior_participant_ranks": prior.get("participant_prior_ranks", {}) if isinstance(prior, dict) else {},
        "prior_participant_scores": prior.get("participant_prior_scores", {}) if isinstance(prior, dict) else {},
        "prior_reported_score_participants": prior.get(
            "reported_score_participants",
            [],
        ) if isinstance(prior, dict) else [],
        "prior_expected_order": prior.get("expected_order", []) if isinstance(prior, dict) else [],
        "final_prior_agreement": _final_prior_agreement(prior),
        "probe_budget_results": _probe_budget_results(prior, extraction),
        "probe_comparisons": extraction.get("probe_comparisons", []),
        # Report-card bars count unique probes, not every routed answer that
        # repeats a probe's taxonomy labels.
        "taxonomy_counts": probe_taxonomy,
        "transcript_taxonomy_counts": transcript_taxonomy,
        "event_timeline": _event_timeline(run_metrics.get("evolution", {})),
        "highlights": _highlight_candidates(extraction, run_metrics),
        "quality_gates": {"summary": behavior_audit.get("summary", {})},
        "audit_findings": behavior_audit.get("findings", []),
    }


def _repair_metadata(config: dict[str, Any]) -> dict[str, Any]:
    metadata = config.get("metadata", {})
    overrides = metadata.get("repair_parameter_overrides", {})
    if not isinstance(overrides, dict):
        overrides = {}
    is_repair = bool(metadata.get("repair_source_run"))
    uses_recovery_params = bool(metadata.get("repair_uses_recovery_params"))
    return {
        "is_repair": is_repair,
        "uses_recovery_params": uses_recovery_params,
        "parameter_overrides": overrides,
        "runtime_sensitive": bool(
            is_repair and (uses_recovery_params or overrides)
        ),
    }


def _structure(config: dict[str, Any]) -> dict[str, Any]:
    participants = config.get("participants", [])
    participant_count = len(participants) if isinstance(participants, list) else 0
    phases = config.get("protocol", {}).get("phases", [])
    if not isinstance(phases, list):
        phases = []
    round_robin_phases = [
        phase for phase in phases if isinstance(phase, dict) and phase.get("kind") == "round_robin_probes"
    ]
    if round_robin_phases:
        rounds = sum(int(phase.get("rounds", 1)) for phase in round_robin_phases)
        include_self = any(bool(phase.get("include_self")) for phase in round_robin_phases)
        respondents_per_probe = participant_count if include_self else max(participant_count - 1, 0)
        probe_count = rounds * participant_count
        routed_qa_count = probe_count * respondents_per_probe
        return {
            "kind": "structured round-robin",
            "rounds": rounds,
            "participant_count": participant_count,
            "probe_count": probe_count,
            "routed_qa_expected": routed_qa_count,
            "formula": (
                f"rounds({rounds}) x models({participant_count}) x "
                f"respondents_per_probe({respondents_per_probe})"
            ),
        }
    judge_phases = [
        phase
        for phase in phases
        if isinstance(phase, dict) and phase.get("kind") == "independent_judge_ranking"
    ]
    if judge_phases:
        judge_count = len(config.get("judges", [])) if isinstance(config.get("judges"), list) else 0
        schedules = [
            [int(value) for value in phase.get("probe_schedule", [])]
            for phase in judge_phases
            if isinstance(phase.get("probe_schedule"), list)
            and phase.get("probe_schedule")
        ]
        if schedules:
            rounds = max(len(schedule) for schedule in schedules)
            probes_per_judge = sum(sum(schedule) for schedule in schedules)
            probe_count = judge_count * probes_per_judge
            baseline_qa = judge_count * sum(schedule[0] for schedule in schedules) * participant_count
            max_qa = probe_count * participant_count
            routed_qa_count: int | str = (
                max_qa
                if all(
                    phase.get("adaptive_targeting", "judge_selected") == "all"
                    for phase in judge_phases
                )
                else f"{baseline_qa}-{max_qa}"
            )
            schedule_text = " + ".join(
                "[" + ", ".join(str(value) for value in schedule) + "]"
                for schedule in schedules
            )
            formula = (
                f"judges({judge_count}) x probe_schedule({schedule_text}); "
                "later rounds may target a subset"
            )
        else:
            rounds = max(int(phase.get("rounds", 1)) for phase in judge_phases)
            baseline_probes = sum(int(phase.get("probes_per_round", 1)) for phase in judge_phases)
            probe_count = judge_count * baseline_probes
            max_qa = probe_count * participant_count
            routed_qa_count = max_qa if rounds == 1 else "adaptive"
            formula = (
                f"judges({judge_count}) x baseline_probes({baseline_probes}) x "
                f"candidates({participant_count}); adaptive rounds are selective"
            )
        return {
            "kind": "independent judge ranking",
            "rounds": rounds,
            "participant_count": participant_count,
            "judge_count": judge_count,
            "probe_count": probe_count,
            "routed_qa_expected": routed_qa_count,
            "formula": formula,
            "probe_schedule": schedules[0] if len(schedules) == 1 else schedules,
        }
    discussion_rounds = sum(
        int(phase.get("rounds", 1))
        for phase in phases
        if isinstance(phase, dict) and phase.get("kind") == "interactive_discussion"
    )
    return {
        "kind": "free discussion",
        "rounds": discussion_rounds,
        "participant_count": participant_count,
        "public_turns_expected": discussion_rounds * participant_count,
        "formula": f"discussion_rounds({discussion_rounds}) x models({participant_count})",
    }


def _event_timeline(evolution_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    events = evolution_metrics.get("probe_events", []) if isinstance(evolution_metrics, dict) else []
    timeline = []
    for event in events if isinstance(events, list) else []:
        if not isinstance(event, dict):
            continue
        timeline.append(
            {
                "turn_id": event.get("turn_id"),
                "phase": event.get("phase"),
                "round_index": event.get("round_index"),
                "speaker": event.get("speaker"),
                "event_type": event.get("event_type"),
                "transition_label": event.get("transition_label"),
                "topical_transition": event.get("topical_transition"),
                "dependency_semantics": event.get("dependency_semantics"),
                "primary_question_type": event.get("primary_question_type"),
                "primary_strategy": event.get("primary_strategy"),
                "excerpt": _excerpt(event.get("content", ""), 180),
            }
        )
    return timeline


def _highlight_candidates(
    extraction: dict[str, Any],
    run_metrics: dict[str, Any],
    limit: int = 4,
) -> list[dict[str, Any]]:
    pairs = extraction.get("qa_pairs", []) if isinstance(extraction, dict) else []
    pair_candidates = []
    seen = set()
    for pair in pairs if isinstance(pairs, list) else []:
        if not isinstance(pair, dict):
            continue
        question_turn_id = pair.get("question_turn_id")
        if question_turn_id in seen:
            continue
        seen.add(question_turn_id)
        pair_candidates.append(
            {
                "source": "routed Q/A",
                "speaker": pair.get("interviewer_id"),
                "target": pair.get("respondent_id"),
                "turn_id": question_turn_id,
                "round_index": pair.get("round_index"),
                "question_types": _tag_names(pair.get("question_type_tags", [])),
                "text": _excerpt(pair.get("question_text", ""), 260),
            }
        )
    highlights = _balanced_take(pair_candidates, key="speaker", limit=limit)
    if len(highlights) < limit:
        events = (
            run_metrics.get("evolution", {}).get("probe_events", [])
            if isinstance(run_metrics, dict)
            else []
        )
        for event in events if isinstance(events, list) else []:
            if not isinstance(event, dict) or event.get("turn_id") in seen:
                continue
            if event.get("event_type") not in {"candidate_question", "discussion_move", "routed_question"}:
                continue
            seen.add(event.get("turn_id"))
            highlights.append(
                {
                    "source": str(event.get("event_type")),
                    "speaker": event.get("speaker"),
                    "target": "group",
                    "turn_id": event.get("turn_id"),
                    "round_index": event.get("round_index"),
                    "question_types": _tag_names(event.get("question_type_tags", [])),
                    "text": _excerpt(event.get("content", ""), 260),
                }
            )
            if len(highlights) >= limit:
                break
    return highlights[:limit]


def _balanced_take(items: list[dict[str, Any]], *, key: str, limit: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item.get(key, "")), []).append(item)
    selected = []
    index = 0
    while len(selected) < limit:
        added = False
        for group in groups.values():
            if index < len(group):
                selected.append(group[index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        index += 1
    return selected


def _tag_names(tags: Any) -> list[str]:
    if not isinstance(tags, list):
        return []
    names = []
    for tag in tags:
        if isinstance(tag, dict) and isinstance(tag.get("tag"), str):
            names.append(tag["tag"])
    return names


def _final_prior_agreement(prior: Any) -> dict[str, Any]:
    if not isinstance(prior, dict):
        return {"judgment_count": 0, "mean_tau": None, "top1_matches": 0}
    judgments = [
        judgment
        for judgment in prior.get("judgments", [])
        if isinstance(judgment, dict) and judgment.get("phase") == "final_judgment"
    ]
    if not judgments:
        all_judgments = [
            judgment
            for judgment in prior.get("judgments", [])
            if isinstance(judgment, dict)
        ]
        primary_judgments = [
            judgment
            for judgment in all_judgments
            if judgment.get("is_primary_judgment") is True
        ]
        if primary_judgments:
            all_judgments = primary_judgments
        latest_by_speaker = {}
        for judgment in all_judgments:
            latest_by_speaker[judgment.get("speaker")] = judgment
        judgments = list(latest_by_speaker.values())
    primary = [
        judgment.get("reported_score_subset")
        if isinstance(judgment.get("reported_score_subset"), dict)
        and judgment["reported_score_subset"].get("candidate_count", 0) >= 2
        else judgment
        for judgment in judgments
    ]
    taus = [
        judgment.get("kendall_tau")
        for judgment in primary
        if isinstance(judgment.get("kendall_tau"), (int, float))
    ]
    all_candidate_taus = [
        judgment.get("kendall_tau")
        for judgment in judgments
        if isinstance(judgment.get("kendall_tau"), (int, float))
    ]
    top1 = [
        judgment.get("top1_matches_prior")
        for judgment in primary
        if judgment.get("top1_matches_prior") is not None
    ]
    uses_reported_subset = any(
        primary_item is not judgment
        for primary_item, judgment in zip(primary, judgments, strict=True)
    )
    return {
        "judgment_count": len(judgments),
        "mean_tau": _mean(taus),
        "all_candidate_mean_tau": _mean(all_candidate_taus),
        "basis": "reported-score subset" if uses_reported_subset else "all scored candidates",
        "top1_matches": sum(1 for value in top1 if value),
        "top1_total": len(top1),
    }


def _probe_budget_results(prior: Any, extraction: Any = None) -> list[dict[str, Any]]:
    results_by_key: dict[tuple[str, Any, int], dict[str, Any]] = {}
    if isinstance(extraction, dict):
        for judgment in extraction.get("wave_judgments", []):
            if not isinstance(judgment, dict):
                continue
            probe_count = judgment.get("judgment_probe_count")
            parsed = judgment.get("parsed")
            if not isinstance(probe_count, int) or not isinstance(parsed, dict):
                continue
            item = {
                "speaker": judgment.get("speaker"),
                "round_index": judgment.get("round_index"),
                "probe_count": probe_count,
                "kendall_tau": None,
                "spearman_rho": None,
                "pairwise_accuracy": None,
                "confidence": parsed.get("confidence"),
                "rank_score_r_squared": None,
                "top1_matches_prior": None,
                "ranking": parsed.get("ranking"),
            }
            key = (str(item["speaker"]), item["round_index"], probe_count)
            results_by_key[key] = item
    judgments = prior.get("judgments", []) if isinstance(prior, dict) else []
    for judgment in judgments:
        if not isinstance(judgment, dict):
            continue
        probe_count = judgment.get("judgment_probe_count")
        if not isinstance(probe_count, int):
            continue
        reported = judgment.get("reported_score_subset")
        primary = (
            reported
            if isinstance(reported, dict) and reported.get("candidate_count", 0) >= 2
            else judgment
        )
        item = {
            "speaker": judgment.get("speaker"),
            "round_index": judgment.get("round_index"),
            "probe_count": probe_count,
            "kendall_tau": primary.get("kendall_tau"),
            "spearman_rho": primary.get("spearman_rho"),
            "pairwise_accuracy": primary.get("pairwise_accuracy"),
            "confidence": judgment.get("confidence"),
            "rank_score_r_squared": primary.get("rank_score_r_squared"),
            "top1_matches_prior": primary.get("top1_matches_prior"),
            "pairwise_accuracy_by_score_gap": primary.get(
                "pairwise_accuracy_by_score_gap"
            ),
            "ranking": judgment.get("ranking"),
            "all_candidate_kendall_tau": judgment.get("kendall_tau"),
            "all_candidate_spearman_rho": judgment.get("spearman_rho"),
            "all_candidate_pairwise_accuracy": judgment.get("pairwise_accuracy"),
            "all_candidate_rank_score_r_squared": judgment.get("rank_score_r_squared"),
            "metric_basis": (
                "reported-score subset" if primary is not judgment else "all scored candidates"
            ),
        }
        key = (str(item["speaker"]), item["round_index"], probe_count)
        results_by_key[key] = item
    return sorted(
        results_by_key.values(),
        key=lambda item: (
            str(item.get("speaker")),
            item.get("round_index") if isinstance(item.get("round_index"), int) else 0,
            item["probe_count"],
        ),
    )


def _probe_budget_markdown(card: dict[str, Any]) -> list[str]:
    results = card.get("probe_budget_results", [])
    if not isinstance(results, list) or not results:
        return []
    adaptive = bool(card.get("structure", {}).get("probe_schedule"))
    lines = [""]
    if adaptive:
        lines.extend(
            [
                "| Judge | Round | Cumulative probes | Kendall | Spearman | Pairwise | Conf. | Ranking | Rank-score R2 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| Judge | Probes | Kendall | Spearman | Pairwise | Conf. | Ranking | Rank-score R2 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
            ]
        )
    for item in results:
        if not isinstance(item, dict):
            continue
        prefix = f"| {item.get('speaker')} | "
        if adaptive:
            prefix += f"{item.get('round_index')} | "
        lines.append(
            prefix
            + f"{item.get('probe_count')} | "
            f"{_format_number(item.get('kendall_tau'))} | "
            f"{_format_number(item.get('spearman_rho'))} | "
            f"{_format_number(item.get('pairwise_accuracy'))} | "
            f"{_format_number(item.get('confidence'))} | "
            f"{' > '.join(str(value) for value in item.get('ranking', []))} | "
            f"{_format_number(item.get('rank_score_r_squared'))} |"
        )
    return lines


def _taxonomy_table(runs: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Run | Top question types | Evolution labels | Top strategies |",
        "| --- | --- | --- | --- |",
    ]
    for card in runs:
        taxonomy = card.get("taxonomy_counts", {})
        question_types = taxonomy.get("question_type_frequency", {}) if isinstance(taxonomy, dict) else {}
        strategies = taxonomy.get("signal_frequency", {}) if isinstance(taxonomy, dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_md(str(card.get("name", ""))),
                    _escape_md(_top_counts(question_types)),
                    _escape_md(_top_counts(card.get("transition_counts", {}))),
                    _escape_md(_top_counts(strategies)),
                ]
            )
            + " |"
        )
    return lines


def _judge_condition_summary(cards: list[dict[str, Any]]) -> dict[str, Any]:
    if len(cards) < 2:
        return {}
    accuracy: dict[tuple[str, int], dict[str, Any]] = {}
    behavior: dict[str, dict[str, Any]] = {}
    final_interjudge_taus = []
    final_rankings_across_runs: list[dict[str, Any]] = []
    for card in cards:
        judges = {
            str(judge.get("id")): judge
            for judge in card.get("judges", [])
            if isinstance(judge, dict) and judge.get("id") is not None
        }
        if not judges:
            continue
        final_tau = card.get("final_agreement", {}).get("mean_pairwise_tau")
        if isinstance(final_tau, (int, float)):
            final_interjudge_taus.append(float(final_tau))
        for final_ranking in card.get("final_rankings", []):
            if not isinstance(final_ranking, dict):
                continue
            judge = judges.get(str(final_ranking.get("speaker")))
            ranking = final_ranking.get("ranking")
            if judge is None or not isinstance(ranking, list):
                continue
            final_rankings_across_runs.append(
                {
                    "provider_model_id": str(
                        judge.get("provider_model_id") or judge.get("model_ref")
                    ),
                    "ranking": [str(value) for value in ranking],
                    "run": card.get("name"),
                    "roster_signature": _roster_signature(card),
                }
            )
        for judge_id, judge in judges.items():
            provider_model_id = str(judge.get("provider_model_id") or judge.get("model_ref"))
            item = behavior.setdefault(
                provider_model_id,
                {
                    "model_ref": judge.get("model_ref"),
                    "provider_model_id": provider_model_id,
                    "probe_validity": Counter(),
                    "adaptive_decisions": 0,
                    "target_sizes": [],
                    "rank_changes": 0,
                    "direct_model_calls": 0,
                    "direct_reported_cost_usd": 0.0,
                },
            )
            spend = card.get("model_spend", {}).get(judge.get("model_ref"), {})
            if isinstance(spend, dict):
                item["direct_model_calls"] += int(spend.get("model_calls") or 0)
                item["direct_reported_cost_usd"] += float(
                    spend.get("reported_cost_usd") or 0.0
                )
        for result in card.get("probe_budget_results", []):
            if not isinstance(result, dict) or not isinstance(result.get("probe_count"), int):
                continue
            judge = judges.get(str(result.get("speaker")))
            if judge is None:
                continue
            provider_model_id = str(judge.get("provider_model_id") or judge.get("model_ref"))
            key = (provider_model_id, result["probe_count"])
            group = accuracy.setdefault(
                key,
                {
                    "model_ref": judge.get("model_ref"),
                    "provider_model_id": provider_model_id,
                    "probe_count": result["probe_count"],
                    "pairwise": [],
                    "kendall": [],
                    "confidence": [],
                    "top1": [],
                },
            )
            for source_key, target_key in (
                ("pairwise_accuracy", "pairwise"),
                ("kendall_tau", "kendall"),
                ("confidence", "confidence"),
            ):
                value = result.get(source_key)
                if isinstance(value, (int, float)):
                    group[target_key].append(float(value))
            if result.get("top1_matches_prior") is not None:
                group["top1"].append(bool(result.get("top1_matches_prior")))
        for comparison in card.get("probe_comparisons", []):
            if not isinstance(comparison, dict):
                continue
            judge = judges.get(str(comparison.get("speaker")))
            parsed = comparison.get("parsed")
            if judge is None or not isinstance(parsed, dict):
                continue
            validity = parsed.get("probe_validity")
            if validity not in {"informative", "limited", "invalid"}:
                continue
            provider_model_id = str(judge.get("provider_model_id") or judge.get("model_ref"))
            behavior[provider_model_id]["probe_validity"][str(validity)] += 1
        trace = card.get("adaptive_metrics", {}).get("decision_trace", [])
        for decision in trace if isinstance(trace, list) else []:
            if not isinstance(decision, dict):
                continue
            judge = judges.get(str(decision.get("judge_id")))
            if judge is None:
                continue
            provider_model_id = str(judge.get("provider_model_id") or judge.get("model_ref"))
            item = behavior[provider_model_id]
            item["adaptive_decisions"] += 1
            targets = decision.get("actual_candidates")
            if isinstance(targets, list):
                item["target_sizes"].append(len(targets))
            if decision.get("ranking_changed") is True:
                item["rank_changes"] += 1

    cross_run_pairs = []
    for index, left in enumerate(final_rankings_across_runs):
        for right in final_rankings_across_runs[index + 1 :]:
            if left["provider_model_id"] == right["provider_model_id"]:
                continue
            if (
                left["roster_signature"] is None
                or left["roster_signature"] != right["roster_signature"]
            ):
                continue
            tau = kendall_tau_between(left["ranking"], right["ranking"])
            if tau is None:
                continue
            final_interjudge_taus.append(tau)
            cross_run_pairs.append(
                {
                    "left_judge": left["provider_model_id"],
                    "right_judge": right["provider_model_id"],
                    "left_run": left["run"],
                    "right_run": right["run"],
                    "kendall_tau": tau,
                    "same_top1": bool(
                        left["ranking"]
                        and right["ranking"]
                        and left["ranking"][0] == right["ranking"][0]
                    ),
                }
            )

    accuracy_rows = []
    for group in accuracy.values():
        pairwise = group.pop("pairwise")
        kendall = group.pop("kendall")
        confidence = group.pop("confidence")
        top1 = group.pop("top1")
        accuracy_rows.append(
            {
                **group,
                "n": len(pairwise),
                "mean_pairwise_accuracy": _mean(pairwise),
                "min_pairwise_accuracy": min(pairwise) if pairwise else None,
                "max_pairwise_accuracy": max(pairwise) if pairwise else None,
                "mean_kendall_tau": _mean(kendall),
                "top1_rate": _mean([1.0 if value else 0.0 for value in top1]),
                "mean_confidence": _mean(confidence),
            }
        )
    behavior_rows = []
    for item in behavior.values():
        target_sizes = item.pop("target_sizes")
        validity = item.get("probe_validity")
        behavior_rows.append(
            {
                **item,
                "probe_validity": dict(validity) if isinstance(validity, Counter) else {},
                "mean_target_size": _mean(target_sizes),
            }
        )
    return {
        "run_count": len(cards),
        "mean_final_interjudge_tau": _mean(final_interjudge_taus),
        "final_interjudge_pairs": cross_run_pairs,
        "accuracy_by_probe_count": sorted(
            accuracy_rows,
            key=lambda item: (str(item.get("model_ref")), int(item.get("probe_count") or 0)),
        ),
        "judge_behavior": sorted(behavior_rows, key=lambda item: str(item.get("model_ref"))),
    }


def _roster_signature(card: dict[str, Any]) -> tuple[tuple[str, str], ...] | None:
    participants = card.get("participants")
    if not isinstance(participants, list) or not participants:
        return None
    rows = []
    for participant in participants:
        if not isinstance(participant, dict):
            return None
        participant_id = participant.get("id")
        model_id = participant.get("provider_model_id") or participant.get("model_ref")
        if not isinstance(participant_id, str) or not isinstance(model_id, str):
            return None
        rows.append((participant_id, model_id))
    return tuple(sorted(rows))


def _judge_condition_markdown(summary: Any) -> list[str]:
    if not isinstance(summary, dict):
        return []
    lines = [
        "",
        "## Judge Conditions Across Runs",
        "",
        f"Accepted runs: {summary.get('run_count')}; mean final inter-judge tau: "
        f"{_format_number(summary.get('mean_final_interjudge_tau'))}.",
        "",
        "| Judge model | Probes | Runs | Mean pairwise | Range | Mean Kendall | Top-1 rate | Mean confidence |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for item in summary.get("accuracy_by_probe_count", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            f"| {item.get('model_ref')} | {item.get('probe_count')} | {item.get('n')} | "
            f"{_format_number(item.get('mean_pairwise_accuracy'))} | "
            f"{_format_interval(item.get('min_pairwise_accuracy'), item.get('max_pairwise_accuracy'))} | "
            f"{_format_number(item.get('mean_kendall_tau'))} | "
            f"{_format_number(item.get('top1_rate'))} | "
            f"{_format_number(item.get('mean_confidence'))} |"
        )
    lines.extend(
        [
            "",
            "| Judge model | Informative | Limited | Invalid | Adaptive decisions | Mean targets | Rank changes | Direct calls | Direct cost |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary.get("judge_behavior", []):
        if not isinstance(item, dict):
            continue
        validity = item.get("probe_validity", {})
        validity = validity if isinstance(validity, dict) else {}
        lines.append(
            f"| {item.get('model_ref')} | {validity.get('informative', 0)} | "
            f"{validity.get('limited', 0)} | {validity.get('invalid', 0)} | "
            f"{item.get('adaptive_decisions')} | {_format_number(item.get('mean_target_size'))} | "
            f"{item.get('rank_changes')} | {item.get('direct_model_calls')} | "
            f"{_format_cost(item.get('direct_reported_cost_usd'))} |"
        )
    return lines


def _mode_notes(runs: list[dict[str, Any]]) -> list[str]:
    if len(runs) < 2:
        return []
    by_mode = {}
    for card in runs:
        by_mode.setdefault(card.get("mode"), []).append(card)
    lines = ["## Mode Notes", ""]
    for mode, cards in sorted(by_mode.items()):
        qa_total = sum(int(card.get("qa_pair_count") or 0) for card in cards)
        candidate_total = sum(int(card.get("candidate_question_turn_count") or 0) for card in cards)
        agreement_values = [
            card.get("final_agreement", {}).get("mean_pairwise_tau")
            for card in cards
            if isinstance(card.get("final_agreement", {}).get("mean_pairwise_tau"), (int, float))
        ]
        lines.append(
            f"- {mode}: {len(cards)} run(s), {qa_total} routed Q/A pairs, "
            f"{candidate_total} candidate free-form question turns, "
            f"mean final agreement tau {_format_number(_mean(agreement_values))}."
        )
    lines.append("")
    return lines


def _paired_comparison(cards: list[dict[str, Any]]) -> dict[str, Any]:
    modes = {card.get("mode") for card in cards}
    if len(cards) < 2 or not {"free discussion", "structured round-robin"}.issubset(modes):
        return {}
    participants: dict[str, dict[str, Any]] = {}
    for card in cards:
        prior_ranks = card.get("prior_participant_ranks", {})
        prior_ranks = prior_ranks if isinstance(prior_ranks, dict) else {}
        for participant in card.get("participants", []):
            if not isinstance(participant, dict):
                continue
            participant_id = participant.get("id")
            if not participant_id:
                continue
            participants.setdefault(
                str(participant_id),
                {
                    "participant_id": str(participant_id),
                    "model_ref": participant.get("model_ref"),
                    "provider_model_id": participant.get("provider_model_id"),
                    "prior_rank": prior_ranks.get(participant_id),
                    "by_mode": {},
                },
            )
    for card in cards:
        mode = str(card.get("mode"))
        received = _received_rank_stats(card.get("final_rankings", []))
        for participant_id, stats in received.items():
            participants.setdefault(
                participant_id,
                {"participant_id": participant_id, "by_mode": {}},
            )["by_mode"][mode] = stats
    return {
        "participants": sorted(
            participants.values(),
            key=lambda item: (
                item.get("prior_rank")
                if isinstance(item.get("prior_rank"), (int, float))
                else 10**9,
                item.get("participant_id", ""),
            ),
        )
    }


def _received_rank_stats(final_rankings: Any) -> dict[str, dict[str, Any]]:
    ranks: dict[str, list[int]] = {}
    top1 = Counter()
    if not isinstance(final_rankings, list):
        return {}
    for item in final_rankings:
        if not isinstance(item, dict) or not isinstance(item.get("ranking"), list):
            continue
        ranking = [str(value) for value in item["ranking"]]
        if ranking:
            top1[ranking[0]] += 1
        for index, participant_id in enumerate(ranking, start=1):
            ranks.setdefault(participant_id, []).append(index)
    return {
        participant_id: {
            "mean_received_rank": _mean([float(value) for value in values]),
            "top1_votes": int(top1.get(participant_id, 0)),
            "judgment_count": len(values),
        }
        for participant_id, values in ranks.items()
    }


def _mode(config: dict[str, Any]) -> str:
    phases = config.get("protocol", {}).get("phases", [])
    kinds = [
        phase.get("kind")
        for phase in phases
        if isinstance(phase, dict) and isinstance(phase.get("kind"), str)
    ]
    if "round_robin_probes" in kinds:
        return "structured round-robin"
    if "independent_judge_ranking" in kinds:
        return "independent judge ranking"
    if "interactive_discussion" in kinds:
        return "free discussion"
    return ", ".join(dict.fromkeys(kinds)) or "unknown"


def _participants(config: dict[str, Any]) -> list[dict[str, str]]:
    return _agent_rows(config, "participants")


def _agent_rows(config: dict[str, Any], field: str) -> list[dict[str, str]]:
    models = _named_map(config.get("models", []))
    participants = []
    for participant in config.get(field, []):
        if not isinstance(participant, dict):
            continue
        model_ref = participant.get("model")
        model = models.get(model_ref, {})
        participants.append(
            {
                "id": str(participant.get("id")),
                "model_ref": str(model_ref),
                "provider_model_id": str(model.get("model", "")),
            }
        )
    return participants


def _compact_taxonomy_counts(taxonomy: Any) -> dict[str, Any]:
    if not isinstance(taxonomy, dict):
        return {}
    return {
        "name": taxonomy.get("name"),
        "version": taxonomy.get("version"),
        "signal_frequency": taxonomy.get("signal_frequency", {}),
        "question_type_frequency": taxonomy.get("question_type_frequency", {}),
    }


def _probe_taxonomy_counts(
    extraction_summary: Any,
    transcript_taxonomy: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(extraction_summary, dict):
        extraction_summary = {}
    return {
        "name": transcript_taxonomy.get("name"),
        "version": transcript_taxonomy.get("version"),
        "signal_frequency": extraction_summary.get("strategy_frequency", {}),
        "question_type_frequency": extraction_summary.get("question_type_frequency", {}),
    }


def _named_map(items: Any) -> dict[str, dict[str, Any]]:
    if isinstance(items, dict):
        return items
    if isinstance(items, list):
        return {
            item["name"]: item
            for item in items
            if isinstance(item, dict) and "name" in item
        }
    return {}


def _models_short(card: dict[str, Any]) -> str:
    return ", ".join(participant["id"] for participant in card.get("participants", []))


def _participants_text(card: dict[str, Any]) -> str:
    return _agents_text(card.get("participants", []))


def _agents_text(agents: Any) -> str:
    if not isinstance(agents, list):
        return ""
    return "; ".join(
        f"{participant['id']}={participant['model_ref']}"
        for participant in agents
    )


def _probe_summary(card: dict[str, Any]) -> str:
    structure = card.get("structure", {})
    if isinstance(structure, dict) and structure.get("kind") in {
        "structured round-robin",
        "independent judge ranking",
    }:
        expected = structure.get("routed_qa_expected")
        observed = str(card.get("qa_pair_count", 0))
        qa_text = f"{observed}/{expected}" if expected is not None else observed
        return (
            f"{structure.get('rounds', 0)} rounds; "
            f"{structure.get('probe_count', 0)} probes; "
            f"{qa_text} routed Q/A"
        )
    return (
        f"{card.get('probe_event_count', 0)} events; "
        f"{card.get('qa_pair_count', 0)} routed Q/A; "
        f"{card.get('candidate_question_turn_count', 0)} candidate questions"
    )


def _agreement_summary(card: dict[str, Any]) -> str:
    agreement = card.get("final_agreement", {})
    if not isinstance(agreement, dict):
        return "n/a"
    return (
        f"tau {_format_number(agreement.get('mean_pairwise_tau'))}; "
        f"same top-1 {agreement.get('same_top1_pairs', 0)}/{agreement.get('pair_count', 0)}"
    )


def _prior_summary(card: dict[str, Any]) -> str:
    prior = card.get("final_prior_agreement", {})
    if not isinstance(prior, dict) or not prior.get("judgment_count"):
        return "n/a"
    return (
        f"tau {_format_number(prior.get('mean_tau'))}; "
        f"top-1 {prior.get('top1_matches', 0)}/{prior.get('top1_total', 0)}"
    )


def _final_rankings_text(card: dict[str, Any]) -> str:
    rankings = []
    for item in card.get("final_rankings", []):
        ranking = item.get("ranking") if isinstance(item, dict) else None
        if isinstance(ranking, list):
            rankings.append(f"{item.get('speaker')}: {' > '.join(str(value) for value in ranking)}")
    return "; ".join(rankings) if rankings else "none parsed"


def _churn_text(card: dict[str, Any]) -> str:
    parts = []
    for speaker, item in sorted(card.get("churn_by_speaker", {}).items()):
        if not isinstance(item, dict):
            continue
        parts.append(
            f"{speaker}: {item.get('top1_changes', 0)} top-1 changes, "
            f"adjacent tau {_format_number(item.get('mean_adjacent_tau'))}"
        )
    return "; ".join(parts) if parts else "no repeated rankings"


def _evolution_text(card: dict[str, Any]) -> str:
    transitions = _top_counts(card.get("transition_counts", {}))
    question_rounds = _round_counts(card.get("question_types_by_round", {}))
    if transitions and question_rounds:
        return f"{transitions}; by round: {question_rounds}"
    return transitions or question_rounds or "none detected"


def _spend_text(card: dict[str, Any]) -> str:
    total = _format_cost(card.get("reported_cost_usd"))
    model_spend = card.get("model_spend", {})
    if not isinstance(model_spend, dict) or not model_spend:
        return total
    breakdown = []
    for model_ref, item in sorted(model_spend.items()):
        if isinstance(item, dict):
            breakdown.append(
                f"{model_ref}: {_format_cost(item.get('reported_cost_usd'))} "
                f"across {item.get('model_calls', 0)} calls"
            )
    result = f"{total} total ({'; '.join(breakdown)})" if breakdown else total
    if card.get("spend_lineage_complete") is False:
        result += "; minimum recorded lineage total"
    return result


def _top_counts(counts: Any, limit: int = 4) -> str:
    if not isinstance(counts, dict) or not counts:
        return ""
    counter = Counter({str(key): int(value) for key, value in counts.items()})
    return ", ".join(f"{key}={value}" for key, value in counter.most_common(limit))


def _round_counts(round_counts: Any) -> str:
    if not isinstance(round_counts, dict) or not round_counts:
        return ""
    parts = []
    for round_key, counts in sorted(round_counts.items(), key=lambda item: _round_sort_key(item[0])):
        top = _top_counts(counts, limit=3)
        if top:
            parts.append(f"{_round_display_label(round_key)}: {top}")
    return "; ".join(parts)


def _round_display_label(value: Any) -> str:
    phase_label = _phase_round_label(value)
    if phase_label:
        return phase_label
    text = str(value)
    if text.startswith("round_"):
        return "Round " + text.rsplit("_", 1)[-1]
    return text


def _phase_round_label(value: Any) -> str:
    text = str(value or "")
    if text.startswith("discussion_round_"):
        return "Discussion Round " + text.rsplit("_", 1)[-1]
    if text.startswith("round_"):
        return "Round " + text.rsplit("_", 1)[-1]
    return ""


def _round_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    number = ""
    for char in reversed(text):
        if char.isdigit():
            number = char + number
        elif number:
            break
    return (int(number) if number else 10**9, text)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _format_number(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.2f}"


def _format_interval(low: Any, high: Any) -> str:
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return "n/a"
    if abs(float(low) - float(high)) < 1e-12:
        return _format_number(low)
    return f"{_format_number(low)}-{_format_number(high)}"


def _format_cost(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"${value:.6f}"


def _format_duration(value: Any) -> str:
    if not isinstance(value, (int, float)) or value < 0:
        return "n/a"
    if value < 60:
        return f"{value:.1f}s"
    minutes, seconds = divmod(int(round(value)), 60)
    return f"{minutes}m {seconds:02d}s"


def _format_rank(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _rank_bucket(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    if value <= 10:
        return "frontier / top 10"
    if value <= 30:
        return "strong / top 30"
    if value <= 50:
        return "mid-strong / top 50"
    if value <= 100:
        return "weaker pilot tier / top 100"
    return "long-tail"


def _labelize(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    curated = _taxonomy_label_map().get(text)
    if curated:
        return curated
    return text.replace("_", " ").replace("-", " ").title()


@lru_cache(maxsize=1)
def _taxonomy_label_map() -> dict[str, str]:
    taxonomy = load_taxonomy()
    labels = {}
    for collection_name in ("tags", "question_types"):
        for item in taxonomy.get(collection_name, []):
            if not isinstance(item, dict):
                continue
            tag_id = item.get("id")
            label = item.get("label")
            if isinstance(tag_id, str) and isinstance(label, str):
                labels[tag_id] = label
    return labels


def _excerpt(value: Any, limit: int = 220) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    base = Path("runs") / "report_cards"
    stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = base / stem
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base / f"{stem}_{suffix}"
    return candidate


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _h(value: Any) -> str:
    return html_escape(str(value if value is not None else ""))
