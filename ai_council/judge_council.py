from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from ai_council.rankings import kendall_tau_between, ranking_ids
from ai_council.spend import compute_spend_lineage


SCHEMA_VERSION = "judge-council-study-v1"


def build_judge_council_report(
    *,
    study_path: str | Path,
    output_dir: str | Path,
    published_json_path: str | Path | None = None,
) -> dict[str, Any]:
    study_path = Path(study_path)
    study = _load_json(study_path)
    catalog = _load_json(study["catalog"])
    catalog_scores = {
        row["provider_model_id"]: float(row["intelligence_score"])
        for row in catalog.get("models", [])
        if isinstance(row.get("intelligence_score"), (int, float))
    }
    panels = [
        analyze_panel(panel, catalog_scores=catalog_scores)
        for panel in study["panels"]
    ]
    evaluator_cost_by_model = _evaluator_cost_by_model(panels)
    evidence_cost_by_model = _evidence_cost_by_model(panels)
    accepted_evidence_cost = sum(
        panel["author"]["evidence_lineage_cost_usd"]
        for panel in panels
    )
    evaluator_cost = sum(
        panel["reported_evaluator_cost_usd"] for panel in panels
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study": study["name"],
        "research_questions": study["research_questions"],
        "study_file": str(study_path),
        "catalog_file": study["catalog"],
        "panels": panels,
        "conditions": summarize_conditions(panels),
        "matched_effects": summarize_matched_effects(panels),
        "reported_evaluator_cost_usd": evaluator_cost,
        "evaluator_cost_by_model": evaluator_cost_by_model,
        "accepted_evidence_lineage_cost_usd": accepted_evidence_cost,
        "accepted_evidence_cost_by_model": evidence_cost_by_model,
        "accepted_artifact_cost_usd": accepted_evidence_cost + evaluator_cost,
        "evaluator_model_calls": sum(
            panel["evaluator_model_calls"] for panel in panels
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "matched-results.svg").write_text(
        render_matched_results_svg(summary),
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


def analyze_panel(
    panel: Mapping[str, Any],
    *,
    catalog_scores: Mapping[str, float],
) -> dict[str, Any]:
    member_runs = {
        model: load_ranking_run(path, catalog_scores=catalog_scores)
        for model, path in panel["member_runs"].items()
    }
    anchor_model = panel["anchor_model"]
    if anchor_model not in member_runs:
        raise ValueError(
            f"panel {panel['id']} has no anchor member {anchor_model}"
        )
    participant_maps = {
        tuple(sorted(row["participant_models"].items()))
        for row in member_runs.values()
    }
    if len(participant_maps) != 1:
        raise ValueError(
            f"panel {panel['id']} member runs do not share a participant map"
        )
    participant_models = next(iter(member_runs.values()))[
        "participant_models"
    ]
    participant_scores = {
        participant_id: catalog_scores[model_id]
        for participant_id, model_id in participant_models.items()
    }
    rankings = {
        model: row["ranking"] for model, row in member_runs.items()
    }
    expected_source = Path(panel["author_run"]).resolve()
    for declared_model, row in member_runs.items():
        if row["evaluator_model"] != declared_model:
            raise ValueError(
                f"panel {panel['id']} declares {declared_model} but "
                f"{row['run_dir']} uses {row['evaluator_model']}"
            )
        evidence_source = row["exact_evidence_source_run"] or row["run_dir"]
        if Path(evidence_source).resolve() != expected_source:
            raise ValueError(
                f"panel {panel['id']} member {declared_model} judged "
                "different evidence"
            )
    council = aggregate_rankings(rankings, participant_scores)
    anchor = score_ranking(rankings[anchor_model], participant_scores)
    author_run = load_ranking_run(
        panel["author_run"],
        catalog_scores=catalog_scores,
    )
    if author_run["participant_models"] != participant_models:
        raise ValueError(
            f"panel {panel['id']} author and council evidence rosters differ"
        )
    author_model = panel["author_model"]
    author_participant = next(
        (
            participant_id
            for participant_id, model_id in participant_models.items()
            if model_id == author_model
        ),
        None,
    )
    if author_participant is None:
        raise ValueError(
            f"panel {panel['id']} does not contain anonymous author response"
        )
    member_agreements = []
    member_models = sorted(rankings)
    for index, left in enumerate(member_models):
        for right in member_models[index + 1 :]:
            member_agreements.append(
                kendall_tau_between(rankings[left], rankings[right])
            )
    author_metrics = score_ranking(
        author_run["ranking"],
        participant_scores,
    )
    author_spend = compute_spend_lineage(panel["author_run"])
    return {
        "id": panel["id"],
        "matched_panel": panel["matched_panel"],
        "battery": panel["battery"],
        "author_model": author_model,
        "author_participant": author_participant,
        "candidate_count": len(participant_scores),
        "participant_models": participant_models,
        "participant_scores": participant_scores,
        "member_runs": {
            model: row["run_dir"] for model, row in member_runs.items()
        },
        "member_rankings": rankings,
        "member_pairwise_accuracy": {
            model: score_ranking(ranking, participant_scores)[
                "pairwise_accuracy"
            ]
            for model, ranking in rankings.items()
        },
        "member_reported_cost_usd": {
            model: row["reported_cost_usd"]
            for model, row in member_runs.items()
        },
        "member_model_calls": {
            model: row["model_calls"]
            for model, row in member_runs.items()
        },
        "mean_member_kendall": mean(
            value for value in member_agreements if value is not None
        ),
        "anchor_model": anchor_model,
        "anchor": {
            **anchor,
            **superior_recognition_for_ranking(
                rankings[anchor_model],
                participant_scores,
                author_participant,
            ),
            "probe_validity_counts": member_runs[anchor_model][
                "probe_validity_counts"
            ],
        },
        "council": {
            **council,
            **superior_recognition_for_votes(
                rankings,
                participant_scores,
                author_participant,
            ),
        },
        "author": {
            **author_metrics,
            **superior_recognition_for_ranking(
                author_run["ranking"],
                participant_scores,
                author_participant,
            ),
            "run_dir": author_run["run_dir"],
            "probe_validity_counts": author_run["probe_validity_counts"],
            "evidence_lineage_cost_usd": author_spend[
                "reported_cost_usd"
            ],
            "evidence_lineage_model_calls": author_spend["model_calls"],
            "evidence_lineage_model_spend": author_spend["model_spend"],
        },
        "reported_evaluator_cost_usd": sum(
            row["reported_cost_usd"] for row in member_runs.values()
        ),
        "evaluator_model_calls": sum(
            row["model_calls"] for row in member_runs.values()
        ),
    }


def load_ranking_run(
    run_dir: str | Path,
    *,
    catalog_scores: Mapping[str, float],
    probe_count: int = 5,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config = _load_json(run_dir / "config.json")
    summary = _load_json(run_dir / "run_summary.json")
    if summary.get("status") != "completed":
        raise ValueError(f"ranking run is not completed: {run_dir}")
    models = _named_rows(config.get("models", []))
    judge_rows = config.get("judges", [])
    judges = (
        list(judge_rows.values())
        if isinstance(judge_rows, Mapping)
        else list(judge_rows)
    )
    if len(judges) != 1 or not isinstance(judges[0], Mapping):
        raise ValueError(f"expected one evaluator in ranking run: {run_dir}")
    judge = judges[0]
    judge_model_ref = judge.get("model")
    if judge_model_ref not in models:
        raise ValueError(
            f"ranking run evaluator has unknown model ref: {run_dir}"
        )
    participant_models = {
        participant["id"]: models[participant["model"]]["model"]
        for participant in config.get("participants", [])
    }
    missing_scores = sorted(
        {
            model_id
            for model_id in participant_models.values()
            if model_id not in catalog_scores
        }
    )
    if missing_scores:
        raise ValueError(
            f"ranking run has candidates without catalog scores: {missing_scores}"
        )
    judgments = []
    validity = Counter()
    for entry in _load_jsonl(run_dir / "transcript.jsonl"):
        metadata = entry.get("metadata") or {}
        parsed = entry.get("parsed")
        if (
            metadata.get("interaction_role") == "probe_comparison"
            and int(metadata.get("probe_sequence_number") or 0) <= probe_count
            and isinstance(parsed, Mapping)
            and parsed.get("probe_validity")
            in {"informative", "limited", "invalid"}
        ):
            validity[str(parsed["probe_validity"])] += 1
        if (
            metadata.get("interaction_role") != "wave_judgment"
            or metadata.get("judgment_probe_count") != probe_count
            or not isinstance(parsed, Mapping)
        ):
            continue
        ranking = ranking_ids(parsed.get("ranking"), accept_id_objects=True)
        if set(ranking) == set(participant_models) and len(ranking) == len(
            participant_models
        ):
            judgments.append(ranking)
    if len(judgments) != 1:
        raise ValueError(
            f"expected one {probe_count}-probe ranking in {run_dir}, "
            f"found {len(judgments)}"
        )
    return {
        "run_dir": str(run_dir),
        "evaluator_model": models[judge_model_ref]["model"],
        "exact_evidence_source_run": (
            config.get("metadata", {}).get("exact_evidence_source_run")
        ),
        "participant_models": participant_models,
        "ranking": judgments[0],
        "probe_validity_counts": {
            label: validity.get(label, 0)
            for label in ("informative", "limited", "invalid")
        },
        "reported_cost_usd": float(summary.get("reported_cost_usd", 0)),
        "model_calls": int(summary.get("model_calls", 0)),
    }


def aggregate_rankings(
    rankings: Mapping[str, Sequence[str]],
    scores: Mapping[str, float],
) -> dict[str, Any]:
    if len(rankings) < 3 or len(rankings) % 2 == 0:
        raise ValueError("a council requires an odd number of at least three members")
    rosters = {tuple(sorted(ranking)) for ranking in rankings.values()}
    if len(rosters) != 1:
        raise ValueError("council members must rank the same roster")
    roster = list(next(iter(rosters)))
    if set(roster) != set(scores):
        raise ValueError("council roster and score roster must match")
    positions = {
        member: {
            participant_id: index
            for index, participant_id in enumerate(ranking)
        }
        for member, ranking in rankings.items()
    }
    pair_rows = []
    correct = 0
    comparable = 0
    majority_wins = defaultdict(int)
    for index, left in enumerate(sorted(roster)):
        for right in sorted(roster)[index + 1 :]:
            left_votes = sum(
                member_positions[left] < member_positions[right]
                for member_positions in positions.values()
            )
            winner = left if left_votes > len(rankings) / 2 else right
            loser = right if winner == left else left
            majority_wins[winner] += 1
            is_correct = (
                scores[winner] > scores[loser]
                if scores[left] != scores[right]
                else None
            )
            if is_correct is not None:
                comparable += 1
                correct += int(is_correct)
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "left_votes": left_votes,
                    "right_votes": len(rankings) - left_votes,
                    "winner": winner,
                    "correct": is_correct,
                }
            )
    mean_ranks = {
        participant_id: mean(
            member_positions[participant_id] + 1
            for member_positions in positions.values()
        )
        for participant_id in roster
    }
    full_ranking = sorted(
        roster,
        key=lambda participant_id: (
            mean_ranks[participant_id],
            -majority_wins[participant_id],
            participant_id,
        ),
    )
    return {
        "member_count": len(rankings),
        "pairwise_correct": correct,
        "pairwise_count": comparable,
        "pairwise_accuracy": correct / comparable if comparable else None,
        "ranking": full_ranking,
        "mean_ranks": mean_ranks,
        "pairwise_votes": pair_rows,
    }


def score_ranking(
    ranking: Sequence[str],
    scores: Mapping[str, float],
) -> dict[str, Any]:
    correct = 0
    count = 0
    for index, left in enumerate(ranking):
        for right in ranking[index + 1 :]:
            if scores[left] == scores[right]:
                continue
            count += 1
            correct += int(scores[left] > scores[right])
    return {
        "ranking": list(ranking),
        "pairwise_correct": correct,
        "pairwise_count": count,
        "pairwise_accuracy": correct / count if count else None,
    }


def superior_recognition_for_ranking(
    ranking: Sequence[str],
    scores: Mapping[str, float],
    author_participant: str,
) -> dict[str, Any]:
    positions = {
        participant_id: index
        for index, participant_id in enumerate(ranking)
    }
    superior = [
        participant_id
        for participant_id, score in scores.items()
        if score > scores[author_participant]
    ]
    recognized = sum(
        positions[participant_id] < positions[author_participant]
        for participant_id in superior
    )
    return {
        "superior_recognized": recognized,
        "superior_total": len(superior),
        "superior_recognition_rate": (
            recognized / len(superior) if superior else None
        ),
    }


def superior_recognition_for_votes(
    rankings: Mapping[str, Sequence[str]],
    scores: Mapping[str, float],
    author_participant: str,
) -> dict[str, Any]:
    positions = [
        {
            participant_id: index
            for index, participant_id in enumerate(ranking)
        }
        for ranking in rankings.values()
    ]
    superior = [
        participant_id
        for participant_id, score in scores.items()
        if score > scores[author_participant]
    ]
    recognized = sum(
        sum(
            position[participant_id] < position[author_participant]
            for position in positions
        )
        > len(positions) / 2
        for participant_id in superior
    )
    return {
        "superior_recognized": recognized,
        "superior_total": len(superior),
        "superior_recognition_rate": (
            recognized / len(superior) if superior else None
        ),
    }


def summarize_conditions(
    panels: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for battery in sorted({panel["battery"] for panel in panels}):
        rows = [panel for panel in panels if panel["battery"] == battery]
        output.append(
            {
                "battery": battery,
                "panel_count": len(rows),
                "author_pairwise_accuracy": _pooled_rate(
                    rows, "author", "pairwise_correct", "pairwise_count"
                ),
                "author_superior_recognition_rate": _pooled_rate(
                    rows,
                    "author",
                    "superior_recognized",
                    "superior_total",
                ),
                "author_informative_probe_rate": _validity_rate(
                    rows, "author", "informative"
                ),
                "anchor_pairwise_accuracy": _pooled_rate(
                    rows, "anchor", "pairwise_correct", "pairwise_count"
                ),
                "anchor_superior_recognition_rate": _pooled_rate(
                    rows,
                    "anchor",
                    "superior_recognized",
                    "superior_total",
                ),
                "anchor_informative_probe_rate": _validity_rate(
                    rows, "anchor", "informative"
                ),
                "council_pairwise_accuracy": _pooled_rate(
                    rows, "council", "pairwise_correct", "pairwise_count"
                ),
                "council_superior_recognition_rate": _pooled_rate(
                    rows,
                    "council",
                    "superior_recognized",
                    "superior_total",
                ),
                "mean_member_kendall": mean(
                    row["mean_member_kendall"] for row in rows
                ),
            }
        )
    return output


def summarize_matched_effects(
    panels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_pair = {
        (panel["matched_panel"], panel["battery"]): panel
        for panel in panels
    }
    panel_ids = sorted({panel["matched_panel"] for panel in panels})
    missing = [
        (panel_id, battery)
        for panel_id in panel_ids
        for battery in ("ordinary", "verifier")
        if (panel_id, battery) not in by_pair
    ]
    if missing:
        raise ValueError(f"matched council study is incomplete: {missing}")
    rows = []
    for panel_id in panel_ids:
        ordinary = by_pair[(panel_id, "ordinary")]
        verifier = by_pair[(panel_id, "verifier")]
        row = {"matched_panel": panel_id}
        for role in ("author", "anchor", "council"):
            row[f"{role}_verifier_delta"] = (
                verifier[role]["pairwise_accuracy"]
                - ordinary[role]["pairwise_accuracy"]
            )
        row["ordinary_council_gain"] = (
            ordinary["council"]["pairwise_accuracy"]
            - ordinary["anchor"]["pairwise_accuracy"]
        )
        row["verifier_council_gain"] = (
            verifier["council"]["pairwise_accuracy"]
            - verifier["anchor"]["pairwise_accuracy"]
        )
        row["interaction"] = (
            row["verifier_council_gain"] - row["ordinary_council_gain"]
        )
        row["author_superior_verifier_delta"] = (
            verifier["author"]["superior_recognition_rate"]
            - ordinary["author"]["superior_recognition_rate"]
        )
        rows.append(row)
    condition_rows = {
        row["battery"]: row for row in summarize_conditions(panels)
    }
    return {
        "panels": rows,
        "pooled_author_superior_verifier_delta": (
            condition_rows["verifier"]["author_superior_recognition_rate"]
            - condition_rows["ordinary"]["author_superior_recognition_rate"]
        ),
        "mean_author_superior_verifier_delta": mean(
            row["author_superior_verifier_delta"] for row in rows
        ),
        "mean_author_verifier_delta": mean(
            row["author_verifier_delta"] for row in rows
        ),
        "mean_anchor_verifier_delta": mean(
            row["anchor_verifier_delta"] for row in rows
        ),
        "mean_ordinary_council_gain": mean(
            row["ordinary_council_gain"] for row in rows
        ),
        "mean_verifier_council_gain": mean(
            row["verifier_council_gain"] for row in rows
        ),
        "mean_interaction": mean(row["interaction"] for row in rows),
    }


def render_matched_results_svg(summary: Mapping[str, Any]) -> str:
    conditions = {
        row["battery"]: row for row in summary["conditions"]
    }
    ranking_bars = [
        ("Ordinary · Sol", conditions["ordinary"]["anchor_pairwise_accuracy"]),
        (
            "Ordinary · council",
            conditions["ordinary"]["council_pairwise_accuracy"],
        ),
        ("Verifier · Sol", conditions["verifier"]["anchor_pairwise_accuracy"]),
        (
            "Verifier · council",
            conditions["verifier"]["council_pairwise_accuracy"],
        ),
    ]
    recognition_bars = [
        (
            "Ordinary probes",
            conditions["ordinary"]["author_superior_recognition_rate"],
        ),
        (
            "Verifier probes",
            conditions["verifier"]["author_superior_recognition_rate"],
        ),
    ]
    width, height = 1180, 540
    top, plot_height = 145, 285
    base = 0.45

    def y(value: float) -> float:
        return top + plot_height * (1 - (value - base) / (1 - base))

    def panel(
        bars: Sequence[tuple[str, float]],
        *,
        left: int,
        plot_width: int,
        title: str,
    ) -> str:
        grid = "".join(
            f"<line x1='{left}' y1='{y(value):.1f}' "
            f"x2='{left + plot_width}' y2='{y(value):.1f}' "
            "stroke='#dde2e4'/>"
            f"<text x='{left - 12}' y='{y(value) + 4:.1f}' "
            "text-anchor='end' font-size='11' fill='#5e6a71'>"
            f"{value:.0%}</text>"
            for value in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        bar_width = 90 if len(bars) > 2 else 120
        gap = (plot_width - bar_width * len(bars)) / (len(bars) + 1)
        marks = []
        for index, (label, value) in enumerate(bars):
            x = left + gap + index * (bar_width + gap)
            color = "#247665" if (
                "council" in label or "Verifier" in label
            ) else "#2364aa"
            marks.append(
                f"<rect x='{x:.1f}' y='{y(value):.1f}' width='{bar_width}' "
                f"height='{top + plot_height - y(value):.1f}' "
                f"fill='{color}'/>"
                f"<text x='{x + bar_width / 2:.1f}' "
                f"y='{y(value) - 9:.1f}' text-anchor='middle' "
                "font-size='13' font-weight='700' fill='#182026'>"
                f"{value:.1%}</text>"
                f"<text x='{x + bar_width / 2:.1f}' "
                f"y='{top + plot_height + 25}' text-anchor='middle' "
                f"font-size='11' fill='#182026'>{escape(label)}</text>"
            )
        return (
            f"<text x='{left}' y='{top - 24}' font-size='15' "
            f"font-weight='700' fill='#182026'>{escape(title)}</text>"
            f"{grid}{''.join(marks)}"
        )

    recognition_panel = panel(
        recognition_bars,
        left=80,
        plot_width=390,
        title="A · Stronger candidates placed above self",
    )
    ranking_panel = panel(
        ranking_bars,
        left=625,
        plot_width=500,
        title="B · Overall candidate-pair accuracy",
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg"
viewBox="0 0 {width} {height}" role="img"
aria-label="Matched verifier-oriented and judge-council results">
<rect width="100%" height="100%" fill="#fff"/>
<text x="42" y="44" font-family="Georgia,serif" font-size="29"
fill="#182026">Can probe design and independent councils improve judgment?</text>
<text x="42" y="76" font-family="system-ui,sans-serif" font-size="14"
fill="#5e6a71">Above-self recognition and overall accuracy on four matched nine-candidate panels.</text>
{recognition_panel}{ranking_panel}
</svg>"""


def _pooled_rate(
    rows: Sequence[Mapping[str, Any]],
    role: str,
    numerator: str,
    denominator: str,
) -> float | None:
    total = sum(int(row[role][denominator]) for row in rows)
    correct = sum(int(row[role][numerator]) for row in rows)
    return correct / total if total else None


def _validity_rate(
    rows: Sequence[Mapping[str, Any]],
    role: str,
    label: str,
) -> float | None:
    counts = [
        row[role].get("probe_validity_counts", {})
        for row in rows
    ]
    total = sum(sum(values.values()) for values in counts)
    selected = sum(int(values.get(label, 0)) for values in counts)
    return selected / total if total else None


def _evaluator_cost_by_model(
    panels: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for panel in panels:
        for model, cost in panel["member_reported_cost_usd"].items():
            row = output.setdefault(
                model,
                {"model_calls": 0, "reported_cost_usd": 0.0},
            )
            row["model_calls"] += panel["member_model_calls"][model]
            row["reported_cost_usd"] += cost
    return output


def _evidence_cost_by_model(
    panels: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for panel in panels:
        for spend in panel["author"]["evidence_lineage_model_spend"].values():
            model = spend.get("provider_model_id")
            if not isinstance(model, str):
                continue
            row = output.setdefault(
                model,
                {"model_calls": 0, "reported_cost_usd": 0.0},
            )
            row["model_calls"] += int(spend.get("model_calls", 0))
            row["reported_cost_usd"] += float(
                spend.get("reported_cost_usd", 0)
            )
    return output


def _named_rows(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, Mapping):
        return {
            str(key): dict(row)
            for key, row in value.items()
            if isinstance(row, Mapping)
        }
    return {
        str(row["name"]): dict(row)
        for row in value
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
