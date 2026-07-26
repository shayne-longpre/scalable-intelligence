from __future__ import annotations

import argparse
from datetime import date
from html import escape
from html.parser import HTMLParser
import json
import math
from pathlib import Path
from statistics import mean

from ai_council.oversight_analysis import render_oversight_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT
    / "runs"
    / "report_cards"
    / "catalog_ladder50_crossed_20260721_v1"
    / "report_card_summary.json"
)
DEFAULT_OUTPUT = ROOT / "docs" / "site"
DEFAULT_SCORE_SUMMARY = (
    ROOT
    / "data"
    / "catalog_ladder50_probe_scores.json"
)
DEFAULT_OVERSIGHT_SUMMARY = ROOT / "data" / "oversight_frontier_results.json"
ORDER_REPLAY_RUN = (
    ROOT
    / "runs"
    / "20260725T203344Z_catalog_ladder50_order_audit_fable_on_sol_seed_20260814"
)

PROBE_TITLES = [
    ("Sol", "Pooled tests"),
    ("Sol", "Mechanics"),
    ("Sol", "Concurrency"),
    ("Sol", "Causal ID"),
    ("Fable", "Reachability"),
    ("Fable", "Language"),
    ("Fable", "Causal choice"),
    ("Fable", "Proof audit"),
]

STRATEGY_PROVENANCE = {
    "criterion_negotiation": "Observed in pilots; related to rubric design and construct validity.",
    "direct_task_probe": "Derived from human test items and AI benchmarks.",
    "adaptive_followup": "Observed in pilots; related to diagnostic interviewing.",
    "adversarial_edge_case": "Derived from critical reasoning and robustness testing.",
    "cross_domain_transfer": "Close match to abstraction and analogy tests.",
    "metacognitive_self_assessment": "Derived from metacognition and calibration; observed in pilots.",
    "recursive_evaluator_evaluation": "Observed in pilots and central to the recursive design.",
    "uncertainty_calibration": "Close match to calibration and metacognitive monitoring.",
    "strategic_signaling": "Observed in pilots; related to signaling games.",
    "evasion_or_performativity": "Observed in pilots; related to instruction-following failures.",
    "deception_or_manipulation": "Derived from deception and evaluation-gaming research.",
    "social_convergence": "Observed in pilots; related to group decision-making.",
}

SOURCE_LINKS = [
    ("Wechsler / Pearson", "https://www.pearsonassessments.com/"),
    (
        "GRE",
        "https://www.ets.org/gre/test-takers/general-test/about/content-structure.html",
    ),
    ("PISA", "https://www.oecd.org/en/about/programmes/pisa.html"),
    ("MMLU", "https://arxiv.org/abs/2009.03300"),
    ("BIG-bench", "https://arxiv.org/abs/2206.04615"),
    ("HELM", "https://arxiv.org/abs/2211.09110"),
    ("GSM8K", "https://arxiv.org/abs/2110.14168"),
    ("MATH", "https://arxiv.org/abs/2103.03874"),
    ("GPQA", "https://arxiv.org/abs/2311.12022"),
    ("HumanEval", "https://arxiv.org/abs/2107.03374"),
    ("MBPP", "https://arxiv.org/abs/2108.07732"),
    ("APPS", "https://arxiv.org/abs/2105.09938"),
    ("SWE-bench", "https://arxiv.org/abs/2310.06770"),
    ("LongBench", "https://arxiv.org/abs/2308.14508"),
    ("IFEval", "https://arxiv.org/abs/2311.07911"),
    ("ToolBench", "https://arxiv.org/abs/2307.16789"),
]

PROBE_SELECTIONS = [
    {
        "run": "20260721T085927Z_catalog_ladder50_gpt_5_6_sol",
        "turn": 3,
        "judge": "Sol",
        "stage": "Opening",
        "title": "Concurrent register correctness",
        "summary": (
            "Audit a seqlock-like concurrent register, construct a failing "
            "interleaving, repair it, and reason about liveness and counter wraparound."
        ),
    },
    {
        "run": "20260721T085927Z_catalog_ladder50_gpt_5_6_sol",
        "turn": 210,
        "judge": "Sol",
        "stage": "Adaptive",
        "title": "Hidden-state robot control",
        "summary": (
            "Design and prove an optimal policy for a partially observed robot after "
            "the opening probes left a cluster of candidates difficult to separate."
        ),
    },
    {
        "run": "20260721T102640Z_catalog_ladder50_claude_fable_5",
        "turn": 2,
        "judge": "Fable",
        "stage": "Opening",
        "title": "Invented-language induction",
        "summary": (
            "Infer a compact grammar, translate new sentences, and identify what the "
            "examples leave underdetermined."
        ),
    },
    {
        "run": "20260721T102640Z_catalog_ladder50_claude_fable_5",
        "turn": 4,
        "judge": "Fable",
        "stage": "Opening",
        "title": "Adversarial proof audit",
        "summary": (
            "Classify each step of a flawed number-theory proof, preserve the true "
            "claim, and repair the invalid argument."
        ),
    },
    {
        "run": "20260721T102640Z_catalog_ladder50_claude_fable_5",
        "turn": 223,
        "judge": "Fable",
        "stage": "Adaptive",
        "title": "Combinatorial tours",
        "summary": (
            "Characterize exactly when a modular partial-sum tour exists, construct "
            "one, and diagnose a subtle false proof."
        ),
    },
    {
        "run": "20260720T183846Z_adaptive_medium_judge_broad_p4",
        "turn": 3,
        "judge": "GPT-5.4 Mini",
        "stage": "Opening",
        "title": "Algorithm audit and counterexample",
        "summary": (
            "Inspect a proposed shortest-path solver, find the decisive failure, and "
            "describe a correct replacement."
        ),
    },
    {
        "run": "20260719T034813Z_independent_judges_ladder12_6probe",
        "turn": 120,
        "judge": "GLM-4.5 Air",
        "stage": "Opening",
        "title": "Metacognitive model audit",
        "summary": (
            "Evaluate an overfit market predictor, update after distribution-shift "
            "evidence, and reflect on weaknesses in the reasoning process."
        ),
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def transcript_by_turn(run_dir: Path) -> dict[int, dict]:
    return {
        turn["turn_id"]: turn
        for turn in (
            json.loads(line)
            for line in (run_dir / "transcript.jsonl").read_text().splitlines()
        )
    }


def ranked(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + end + 1) / 2
        for offset in range(index, end):
            result[order[offset]] = average_rank
        index = end
    return result


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    x_values = ranked(left)
    y_values = ranked(right)
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    x_variance = sum((value - x_mean) ** 2 for value in x_values)
    y_variance = sum((value - y_mean) ** 2 for value in y_values)
    denominator = math.sqrt(x_variance * y_variance)
    return numerator / denominator if denominator else 0.0


def partial_spearman(
    left: list[float], right: list[float], control: list[float]
) -> float:
    left_right = spearman(left, right)
    left_control = spearman(left, control)
    right_control = spearman(right, control)
    denominator = math.sqrt(
        max(0.0, (1 - left_control**2) * (1 - right_control**2))
    )
    return (
        (left_right - left_control * right_control) / denominator
        if denominator
        else 0.0
    )


def pairwise_accuracy(ordering: list[str], scores: dict[str, float]) -> float:
    position = {participant_id: rank for rank, participant_id in enumerate(ordering)}
    comparable = [item for item in ordering if item in scores]
    correct = 0
    total = 0
    for left_index, left in enumerate(comparable):
        for right in comparable[left_index + 1 :]:
            if scores[left] == scores[right]:
                continue
            total += 1
            predicted = position[left] < position[right]
            expected = scores[left] > scores[right]
            correct += predicted == expected
    return correct / total if total else 0.0


def score_pairwise_accuracy(
    predictions: dict[str, float], scores: dict[str, float]
) -> float:
    comparable = sorted(predictions.keys() & scores.keys())
    correct = 0.0
    total = 0
    for left_index, left in enumerate(comparable):
        for right in comparable[left_index + 1 :]:
            if scores[left] == scores[right]:
                continue
            total += 1
            if predictions[left] == predictions[right]:
                correct += 0.5
            elif (predictions[left] > predictions[right]) == (
                scores[left] > scores[right]
            ):
                correct += 1
    return correct / total if total else 0.0


def kendall_order(left: list[str], right: list[str]) -> float:
    if set(left) != set(right) or len(left) < 2:
        return 0.0
    left_position = {item: index for index, item in enumerate(left)}
    right_position = {item: index for index, item in enumerate(right)}
    concordant = 0
    discordant = 0
    for left_index, first in enumerate(left):
        for second in left[left_index + 1 :]:
            same_direction = (
                left_position[first] - left_position[second]
            ) * (
                right_position[first] - right_position[second]
            )
            concordant += same_direction > 0
            discordant += same_direction < 0
    return (concordant - discordant) / (concordant + discordant)


def page(title: str, description: str, body: str, active: str) -> str:
    nav_items = [
        ("Results", "index.html", "results"),
        ("Taxonomy", "taxonomy.html", "taxonomy"),
        ("Models", "models.html", "models"),
        ("Audit sample", "audit.html", "audit"),
    ]
    nav = "".join(
        f'<a href="{href}"{" aria-current=\"page\"" if key == active else ""}>'
        f"{label}</a>"
        for label, href, key in nav_items
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{escape(description, quote=True)}">
  <title>{escape(title)}</title>
  <link rel="stylesheet" href="site.css">
</head>
<body>
  <header class="site-header">
    <a class="wordmark" href="index.html">Machine Societies</a>
    <nav aria-label="Primary">{nav}</nav>
  </header>
  <main>{body}</main>
  <footer>
    <p>Exploratory results. External intelligence scores are reference measurements, not ground truth.</p>
    <p>Generated from versioned run artifacts on {date.today().isoformat()}.</p>
  </footer>
</body>
</html>
"""


def badge(text: str, kind: str = "") -> str:
    class_name = f"badge {kind}".strip()
    return f'<span class="{class_name}">{escape(text)}</span>'


def compact(text: str, limit: int = 640) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def evidence_author(run_name: str) -> str:
    normalized = run_name.lower()
    if "sol_evidence" in normalized:
        return "sol"
    if "fable_evidence" in normalized:
        return "fable"
    return "sol" if "sol" in normalized else "fable"


def article(
    report: dict,
    catalog: dict,
    taxonomy: dict,
    score_summary: dict | None,
    oversight_summary: dict | None,
) -> str:
    runs = report["runs"]
    best_run = next(
        run for run in runs if "cross_fable_judges_sol_evidence" in run["name"]
    )
    best_checkpoint = next(
        checkpoint
        for checkpoint in best_run["probe_budget_results"]
        if checkpoint["probe_count"] == 4
    )
    participants = {
        item["id"]: item["provider_model_id"] for item in best_run["participants"]
    }
    catalog_by_id = {item["provider_model_id"]: item for item in catalog["models"]}
    external_order = sorted(
        best_run["prior_participant_scores"],
        key=lambda item: (-best_run["prior_participant_scores"][item], item),
    )
    external_rank = {
        participant_id: rank
        for rank, participant_id in enumerate(external_order, start=1)
    }
    ranking_rows = []
    for judged_rank, participant_id in enumerate(best_checkpoint["ranking"], start=1):
        provider_id = participants[participant_id]
        model = catalog_by_id[provider_id]
        score = best_run["prior_participant_scores"][participant_id]
        estimated = model.get("intelligence_score_is_estimated", True)
        delta = judged_rank - external_rank[participant_id]
        ranking_rows.append(
            "<tr>"
            f"<td class=\"number\">{judged_rank}</td>"
            f"<td><strong>{escape(model['display_name'])}</strong>"
            f"<span class=\"subcell\">{escape(provider_id)}</span></td>"
            f"<td class=\"number\">{score:.1f}{' *' if estimated else ''}</td>"
            f"<td class=\"number\">{external_rank[participant_id]}</td>"
            f"<td class=\"number {'down' if delta > 0 else 'up' if delta < 0 else ''}\">"
            f"{delta:+d}</td>"
            f"<td>{escape(model.get('release_date') or 'Unknown')}</td>"
            "</tr>"
        )

    probe_cards = build_probe_cards(taxonomy)
    heatmap, heatmap_stats = build_heatmap(
        runs, catalog_by_id, score_summary=score_summary
    )
    order_audit = presentation_order_replay(best_checkpoint["ranking"])
    source_pair = report["judge_condition_summary"]["final_interjudge_pairs"]
    same_evidence = [
        item["kendall_tau"]
        for item in source_pair
        if evidence_author(item["left_run"]) == evidence_author(item["right_run"])
    ]
    mean_same_evidence = mean(same_evidence)
    direct_scores = len(best_run["prior_reported_score_participants"])
    strategy_count = len(taxonomy["tags"])
    question_count = len(taxonomy["question_types"])
    heatmap_method = (
        "Each cell is the mean anchored answer-quality score from Sol and Fable"
        if score_summary
        else "Each cell is the mean within-probe percentile from Sol and Fable"
    )
    heatmap_caveat = (
        "Seven probes have both judges; the pooled-test column uses Sol alone "
        "because Fable repeatedly returned an empty provider response."
        if score_summary
        else "These are relative positions, not absolute difficulty scores."
    )
    oversight_section = _oversight_section(oversight_summary).strip()

    return f"""
<article>
  <section class="article-head">
    <p class="eyebrow">A new kind of intelligence test</p>
    <h1>Can AI systems recognize intelligence in one another?</h1>
    <p class="dek">Most AI evaluations begin with human-written questions. This
    experiment turns the process around: capable models invent the questions,
    interrogate anonymous peers, and decide for themselves what counts as
    convincing evidence of intelligence.</p>
    <div class="key-result">
      <span>First catalog result</span>
      <strong>{best_checkpoint['pairwise_accuracy']:.1%}</strong>
      <p>of comparable model pairs were ordered consistently with an external
      intelligence index in the best observed condition.</p>
    </div>
  </section>

  <section id="motivation" class="prose">
    <h2>Why build this?</h2>
    <p>Benchmarks are good at measuring tasks with verifiable answers: factual
    recall, mathematics, coding, exams, and agentic work. They are less direct
    measures of qualities people also associate with intelligence: judgment,
    creativity, strategic thought, adaptability, social reasoning, and the
    ability to recognize competence in someone else.</p>
    <p>Models already evaluate model outputs, but usually against human-written
    rubrics. Here the evaluation method is itself the object of study. What do
    models test when nobody defines intelligence for them? Do they ask better
    questions as evidence accumulates? Can a weaker system recognize a stronger
    one?</p>
  </section>

  <section id="method" class="prose ruled">
    <p class="section-kicker">Method in 30 seconds</p>
    <h2>Anonymous candidates, independent judges</h2>
    <p>Two frontier judges independently wrote four opening probes and sent each
    probe unchanged to 50 anonymous models. The judge compared all answers to a
    probe, then combined the evidence into a ranking. It could ask two later
    follow-ups of the ten candidates it found hardest to separate. Candidate
    names and external scores were hidden throughout.</p>
    <div class="method-grid">
      <div><strong>50</strong><span>candidate models</span></div>
      <div><strong>2</strong><span>independent judges</span></div>
      <div><strong>4 + 2</strong><span>opening + adaptive probes</span></div>
      <div><strong>{direct_scores}</strong><span>direct external scores</span></div>
    </div>
    <p class="note">The external Artificial Analysis Intelligence Index is a
    useful reference, not a literal ground truth. Exact model settings can
    differ, nearby scores are noisy, and three deliberately weak anchors use
    estimates. See the <a href="models.html">model table</a>, the
    <a href="taxonomy.html">{strategy_count}-strategy / {question_count}-question
    taxonomy</a>, and the <a href="audit.html">small human audit sample</a>.</p>
  </section>

  <section id="catalog-results" class="wide ruled">
    <p class="section-kicker">Research question 1</p>
    <h2>Can a strong judge reconstruct a 50-model capability ladder?</h2>
    <p class="section-intro">Broadly, yes. Fine distinctions remain hard. The
    best result came from Fable interpreting answers to Sol-authored probes,
    which separated question design from answer interpretation.</p>

    <figure class="figure-wide">
      <img src="../figures/catalog-ladder50/predicted-vs-external.svg"
           alt="Judged capability percentile plotted against external Intelligence Index for Sol and Fable.">
      <figcaption>Each point is an anonymous candidate. Both independent judges
      recovered the broad external ordering; the largest errors occur among
      nearby models.</figcaption>
    </figure>

    <div class="figure-pair">
      <figure>
        <img src="../figures/catalog-ladder50/discrimination-by-gap.svg"
             alt="Pairwise accuracy increases with the external capability gap.">
        <figcaption>Large capability differences were much easier to recognize
        than small ones.</figcaption>
      </figure>
      <figure>
        <img src="../figures/catalog-ladder50/crossed-judge-accuracy.svg"
             alt="Accuracy for both judges on Sol-authored and Fable-authored evidence.">
        <figcaption>Swapping judges while holding answers fixed revealed a
        strong evidence-design effect.</figcaption>
      </figure>
    </div>

    <figure class="figure-wide compact-figure">
      <img src="../figures/catalog-ladder50/evidence-scaling.svg"
           alt="Ranking accuracy after four, five, and six probes.">
      <figcaption>More evidence did not always improve external agreement. The
      fifth and sixth probes changed rankings, but their value depended on the
      judge and evidence battery.</figcaption>
    </figure>

    <div class="finding-grid">
      <div><strong>86.7%</strong><span>best pairwise accuracy</span></div>
      <div><strong>{best_checkpoint['kendall_tau']:.2f}</strong><span>best Kendall rank correlation</span></div>
      <div><strong>{mean_same_evidence:.2f}</strong><span>mean same-evidence judge agreement</span></div>
      <div><strong>≈ chance</strong><span>for gaps under two index points</span></div>
    </div>

    <aside class="background-note">
      <strong>Stability checks.</strong> The shared-evidence cross-over showed
      substantially greater agreement when judges saw the same answers than when
      one judge saw different probe batteries. In a fresh shuffled-order replay
      of the best four-probe condition, the two rankings agreed at Kendall
      {order_audit['kendall_tau']:.2f}; their top three were unchanged and
      {order_audit['top10_overlap']} of the top ten overlapped. External pairwise
      accuracy moved from {best_checkpoint['pairwise_accuracy']:.1%} to
      {order_audit['pairwise_accuracy']:.1%}. This is bounded but real order
      sensitivity. Global comparison remains primary; panels would add anchoring
      and merge assumptions without clear evidence of a net gain.
    </aside>
  </section>

  <section id="best-ladder" class="wide ruled">
    <p class="section-kicker">Best observed condition</p>
    <h2>The full judged ladder</h2>
    <p class="section-intro">Fable judged the four Sol-authored opening probes.
    “Δ rank” is judged rank minus external rank: positive values mean the judge
    placed a model lower. Asterisks mark three estimated external anchors.</p>
    <div class="table-tools">
      <label>Filter models <input type="search" data-table-filter="ladder-table"
      placeholder="Name or provider ID"></label>
      <a href="models.html">Open the source catalog</a>
    </div>
    <div class="table-frame tall">
      <table id="ladder-table">
        <thead><tr><th>Judged</th><th>Model</th><th>Index</th>
        <th>External</th><th>Δ rank</th><th>Released</th></tr></thead>
        <tbody>{''.join(ranking_rows)}</tbody>
      </table>
    </div>
  </section>

  <section id="probe-repertoire" class="wide ruled">
    <p class="section-kicker">What did the judges ask?</p>
    <h2>Not one IQ test, but a repertoire</h2>
    <p class="section-intro">Frontier judges favored questions with several
    independently checkable obligations: construct something, justify it, find
    edge cases, and know when the evidence is insufficient. The smaller judges
    often chose recognizable templates with fewer interlocking demands.</p>
    <div class="probe-grid">{probe_cards}</div>
  </section>

  <section id="answer-map" class="wide ruled">
    <p class="section-kicker">Answer-quality map</p>
    <h2>Different probes expose different capability profiles</h2>
    <p class="section-intro">Rows follow the external ladder. {heatmap_method}.
    The fixed scale runs from 0 (no usable answer) to 4 (fully correct,
    complete, and rigorous).</p>
    <div class="heat-legend"><span>0 · unusable</span>
      <i class="heat h1"></i><i class="heat h3"></i><i class="heat h5"></i>
      <i class="heat h7"></i><i class="heat h9"></i><span>4 · fully correct</span>
      <i class="heat missing"></i><span>Missing answer</span></div>
    <div class="table-frame heat-frame">{heatmap}</div>
    <p class="note">{heatmap_stats}. {heatmap_caveat}</p>
  </section>

  {oversight_section}
</article>
<script>
for (const input of document.querySelectorAll("[data-table-filter]")) {{
  input.addEventListener("input", () => {{
    const query = input.value.toLowerCase();
    const table = document.getElementById(input.dataset.tableFilter);
    for (const row of table.tBodies[0].rows) {{
      row.hidden = !row.textContent.toLowerCase().includes(query);
    }}
  }});
}}
</script>
"""


def _oversight_section(summary: dict | None) -> str:
    if not summary:
        return """
  <section id="oversight-frontier" class="prose ruled">
    <p class="section-kicker">Next primary experiment</p>
    <h2>The oversight frontier</h2>
    <div class="placeholder">
      <strong>Can a model recognize a system more capable than itself?</strong>
      <p>Judges at several capability levels rank small anonymous panels
      containing models below, near, and above their own external score.</p>
      <span>Experiment design ready · results pending</span>
    </div>
  </section>"""
    aggregate = summary["aggregate"]
    return f"""
  <section id="oversight-frontier" class="prose ruled">
    <p class="section-kicker">Scalable oversight</p>
    <h2>The oversight frontier</h2>
    <p>Six judges spanning the capability ladder each wrote five opening probes,
    ranked seven anonymous candidates, and used one targeted follow-up. Together
    they ordered {aggregate['final_pair_accuracy']:.1%} of candidate pairs in
    line with the external index and placed
    {aggregate['superior_recognized']} of {aggregate['superior_total']} externally
    stronger candidates above their own anonymous selves. Across both stronger
    and weaker candidates, {aggregate['self_relative_correct']} of
    {aggregate['self_relative_total']} self-relative comparisons were correct.</p>
    <p><a class="text-link" href="oversight.html">Explore the full oversight
    scorecard, probe repertoire, and adaptive results →</a></p>
  </section>"""


def build_probe_cards(taxonomy: dict) -> str:
    labels = {
        item["id"]: item["label"] for item in taxonomy["question_types"]
    }
    cards = []
    for selection in PROBE_SELECTIONS:
        run_dir = ROOT / "runs" / selection["run"]
        turn = transcript_by_turn(run_dir)[selection["turn"]]
        extraction = load_json(run_dir / "posthoc_extraction.json")
        event = next(
            item
            for item in extraction["probe_events"]
            if item["turn_id"] == selection["turn"]
        )
        tags = [
            labels.get(item["tag"], item.get("label", item["tag"]))
            for item in event.get("question_type_tags", [])
        ]
        tag_html = "".join(badge(tag) for tag in tags[:4])
        cards.append(
            f"""<article class="probe-card">
  <div class="probe-meta">{badge(selection['judge'], 'judge')} {badge(selection['stage'], 'stage')}</div>
  <h3>{escape(selection['title'])}</h3>
  <p>{escape(selection['summary'])}</p>
  <div class="tag-row">{tag_html}</div>
  <details><summary>Read a source excerpt</summary>
    <blockquote>{escape(compact(turn['content'], 900))}</blockquote>
    <p class="source-ref">Run {escape(selection['run'])}, turn {selection['turn']}</p>
  </details>
</article>"""
        )
    return "".join(cards)


def build_heatmap(
    runs: list[dict],
    catalog_by_id: dict[str, dict],
    score_summary: dict | None = None,
) -> tuple[str, str]:
    if score_summary is not None:
        return build_score_heatmap(runs, catalog_by_id, score_summary)

    run_by_name = {run["name"]: run for run in runs}
    batteries = [
        (
            "Sol",
            run_by_name["catalog_ladder50_gpt_5_6_sol"],
            run_by_name[
                "catalog_ladder50_cross_fable_judges_sol_evidence_complete"
            ],
            ["Pooled tests", "Physics", "Concurrency", "Causal ID"],
        ),
        (
            "Fable",
            run_by_name["catalog_ladder50_claude_fable_5"],
            run_by_name[
                "catalog_ladder50_cross_sol_judges_fable_evidence_complete"
            ],
            ["Reachability", "Language", "Causal choice", "Proof audit"],
        ),
    ]
    source_run = batteries[0][1]
    participants = {
        item["id"]: item["provider_model_id"] for item in source_run["participants"]
    }
    scores = source_run["prior_participant_scores"]
    reported = set(source_run["prior_reported_score_participants"])
    external_order = sorted(scores, key=lambda item: (-scores[item], item))
    columns: list[dict] = []
    all_alignments = []
    for author, original, crossed, titles in batteries:
        for probe_index, title in enumerate(titles):
            comparisons = [
                original["probe_comparisons"][probe_index],
                crossed["probe_comparisons"][probe_index],
            ]
            missing = {
                participant_id
                for participant_id, summary in comparisons[0]["parsed"]
                .get("candidate_summaries", {})
                .items()
                if "unscorable" in summary.lower()
                or "no complete visible answer" in summary.lower()
                or "unavailable" in summary.lower()
            }
            percentiles: dict[str, float] = {}
            for participant_id in external_order:
                if participant_id in missing:
                    continue
                values = []
                for comparison in comparisons:
                    ordering = comparison["parsed"]["ordering"]
                    position = ordering.index(participant_id)
                    values.append(100 * (len(ordering) - 1 - position) / (len(ordering) - 1))
                percentiles[participant_id] = mean(values)
            direct_scores = {
                participant_id: scores[participant_id]
                for participant_id in reported
                if participant_id in percentiles
            }
            averaged_order = sorted(
                percentiles, key=lambda item: (-percentiles[item], item)
            )
            alignment = pairwise_accuracy(averaged_order, direct_scores)
            all_alignments.append(alignment)
            columns.append(
                {
                    "author": author,
                    "title": title,
                    "percentiles": percentiles,
                    "alignment": alignment,
                }
            )

    headers = "".join(
        f'<th title="{escape(column["author"])} probe; '
        f'{column["alignment"]:.0%} pairwise alignment with the external index">'
        f'<span>{escape(column["author"])}</span>{escape(column["title"])}</th>'
        for column in columns
    )
    rows = []
    for external_rank, participant_id in enumerate(external_order, start=1):
        provider_id = participants[participant_id]
        name = catalog_by_id[provider_id]["display_name"].split(": ", 1)[-1]
        cells = []
        for column in columns:
            value = column["percentiles"].get(participant_id)
            if value is None:
                cells.append('<td class="heat missing" title="Missing answer">—</td>')
                continue
            bucket = max(1, min(9, round(value / 12.5) + 1))
            cells.append(
                f'<td class="heat h{bucket}" title="{value:.0f}th within-probe '
                f'percentile">{value:.0f}</td>'
            )
        rows.append(
            f"<tr><th><span>{external_rank}</span>{escape(name)}"
            f"<small>{scores[participant_id]:.1f}</small></th>{''.join(cells)}</tr>"
        )
    table = (
        '<table class="heatmap"><thead><tr><th>External ladder</th>'
        f"{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
    stats = (
        "Across the eight probes, mean pairwise alignment between the averaged "
        f"within-probe order and the external index was {mean(all_alignments):.0%}; "
        f"the range was {min(all_alignments):.0%}–{max(all_alignments):.0%}"
    )
    return table, stats


def build_score_heatmap(
    runs: list[dict],
    catalog_by_id: dict[str, dict],
    score_summary: dict,
) -> tuple[str, str]:
    run_by_name = {run["name"]: run for run in runs}
    source_run = run_by_name["catalog_ladder50_gpt_5_6_sol"]
    participants = {
        item["id"]: item["provider_model_id"] for item in source_run["participants"]
    }
    scores = source_run["prior_participant_scores"]
    reported = set(source_run["prior_reported_score_participants"])
    external_order = sorted(scores, key=lambda item: (-scores[item], item))
    probes = score_summary["probes"]
    if len(probes) != len(PROBE_TITLES):
        raise ValueError(
            f"expected {len(PROBE_TITLES)} scored probes, found {len(probes)}"
        )

    headers = []
    for probe, (author, title) in zip(probes, PROBE_TITLES, strict=True):
        headers.append(
            f'<th title="{escape(author)} probe; mean answer score '
            f'{probe["mean_answer_score"]:.2f}; '
            f'{probe["substantially_correct_rate"]:.0%} scored 3 or 4">'
            f"<span>{escape(author)}</span>{escape(title)}"
            f"<small>μ {probe['mean_answer_score']:.1f}</small></th>"
        )

    aggregate: dict[str, list[float]] = {}
    rows = []
    for external_rank, participant_id in enumerate(external_order, start=1):
        provider_id = participants[participant_id]
        name = catalog_by_id[provider_id]["display_name"].split(": ", 1)[-1]
        cells = []
        for probe in probes:
            value = probe["mean_scores"].get(participant_id)
            if value is None:
                cells.append('<td class="heat missing" title="Missing answer">—</td>')
                continue
            aggregate.setdefault(participant_id, []).append(float(value))
            bucket = max(1, min(9, round(float(value) * 2) + 1))
            cells.append(
                f'<td class="heat h{bucket}" title="{value:.1f} / 4; '
                f'{probe["judge_count"]} scoring judge'
                f'{"s" if probe["judge_count"] != 1 else ""}">{value:.1f}</td>'
            )
        rows.append(
            f"<tr><th><span>{external_rank}</span>{escape(name)}"
            f"<small>{scores[participant_id]:.1f}</small></th>{''.join(cells)}</tr>"
        )

    aggregate_means = {
        participant_id: mean(values)
        for participant_id, values in aggregate.items()
        if participant_id in reported
    }
    direct_scores = {
        participant_id: scores[participant_id] for participant_id in aggregate_means
    }
    correlation = spearman(
        [aggregate_means[item] for item in aggregate_means],
        [direct_scores[item] for item in aggregate_means],
    )
    alignment = score_pairwise_accuracy(aggregate_means, direct_scores)
    disagreement = [
        probe["mean_judge_score_range"]
        for probe in probes
        if probe["mean_judge_score_range"] is not None
    ]
    table = (
        '<table class="heatmap"><thead><tr><th>External ladder</th>'
        f"{''.join(headers)}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )
    stats = (
        "Averaging scores across probes produced Spearman "
        f"{correlation:.2f} with the external index and {alignment:.1%} "
        "pairwise alignment. Mean absolute judge disagreement across the seven "
        f"jointly scored probes was {mean(disagreement):.2f} points"
    )
    return table, stats


def presentation_order_replay(original_ranking: list[str]) -> dict[str, float | int]:
    analysis = load_json(ORDER_REPLAY_RUN / "analysis_summary.json")
    judgment = analysis["prior_agreement"]["judgments"][-1]
    replay_ranking = judgment["ranking"]
    original_position = {
        participant_id: rank
        for rank, participant_id in enumerate(original_ranking, start=1)
    }
    replay_position = {
        participant_id: rank
        for rank, participant_id in enumerate(replay_ranking, start=1)
    }
    return {
        "kendall_tau": kendall_order(original_ranking, replay_ranking),
        "top10_overlap": len(set(original_ranking[:10]) & set(replay_ranking[:10])),
        "mean_absolute_displacement": mean(
            abs(original_position[item] - replay_position[item])
            for item in original_ranking
        ),
        "pairwise_accuracy": judgment["reported_score_subset"][
            "pairwise_accuracy"
        ],
    }


def presentation_position_audit(runs: list[dict]) -> list[dict]:
    results = []
    for run in runs:
        turns = transcript_by_turn(ROOT / run["run_dir"])
        scores = run["prior_participant_scores"]
        for comparison in run["probe_comparisons"][:4]:
            presented = [
                turns[turn_id]["speaker"] for turn_id in comparison["answer_turn_ids"]
            ]
            position = {
                participant_id: rank
                for rank, participant_id in enumerate(presented, start=1)
            }
            ordering = [
                participant_id
                for participant_id in comparison["parsed"]["ordering"]
                if participant_id in position
            ]
            results.append(
                {
                    "run": run["name"],
                    "probe": comparison["probe_sequence_number"],
                    "partial_rho": partial_spearman(
                        [position[item] for item in ordering],
                        list(range(1, len(ordering) + 1)),
                        [-scores[item] for item in ordering],
                    ),
                }
            )
    return results


def taxonomy_page(taxonomy: dict) -> str:
    strategy_rows = "".join(
        "<tr>"
        f"<td><strong>{escape(item['label'])}</strong>"
        f"<span class=\"subcell\">{escape(item['id'])}</span></td>"
        f"<td>{escape(item['description'])}</td>"
        f"<td>{escape(STRATEGY_PROVENANCE[item['id']])}</td>"
        "</tr>"
        for item in taxonomy["tags"]
    )
    question_rows = "".join(
        "<tr>"
        f"<td><strong>{escape(item['label'])}</strong>"
        f"<span class=\"subcell\">{escape(item['id'])}</span></td>"
        f"<td>{escape(item['description'])}</td>"
        f"<td>{escape('; '.join(item.get('human_basis', [])))}</td>"
        f"<td>{escape('; '.join(item.get('ai_basis', [])))}</td>"
        "</tr>"
        for item in taxonomy["question_types"]
    )
    links = " · ".join(
        f'<a href="{href}">{escape(label)}</a>' for label, href in SOURCE_LINKS
    )
    body = f"""
<section class="page-head">
  <p class="eyebrow">Measurement reference</p>
  <h1>Evaluation taxonomy</h1>
  <p class="dek">A compact coding frame for what evaluators do and what their
  questions test. It is a review aid, not a complete theory of intelligence or
  a validated psychometric instrument.</p>
</section>
<section class="prose">
  <h2>How to read it</h2>
  <p><strong>Evaluation strategies</strong> describe how a model gathers or uses
  evidence. <strong>Question types</strong> describe the capability a probe is
  trying to elicit. A single probe may receive several labels.</p>
  <p>Structural facts such as speaker, round, probe identity, target, and whether
  a follow-up was generated after seeing evidence come from transcript metadata.
  Semantic labels are conservative automated suggestions until audited by
  people.</p>
  <div class="legend-box"><strong>Source relations</strong>
    <span><b>Close match</b> follows an established assessment family.</span>
    <span><b>Derived</b> combines related literatures into a practical bucket.</span>
    <span><b>Observed</b> captures behavior seen in these experiments.</span>
  </div>
</section>
<section class="wide ruled">
  <div class="section-heading"><div><p class="section-kicker">Dimension 1</p>
  <h2>Evaluation strategies</h2></div><strong>{len(taxonomy['tags'])} labels</strong></div>
  <div class="table-frame"><table>
    <thead><tr><th>Label</th><th>Definition</th><th>Provenance</th></tr></thead>
    <tbody>{strategy_rows}</tbody>
  </table></div>
</section>
<section class="wide ruled">
  <div class="section-heading"><div><p class="section-kicker">Dimension 2</p>
  <h2>Question types</h2></div><strong>{len(taxonomy['question_types'])} labels</strong></div>
  <div class="table-tools"><label>Filter labels
    <input type="search" data-table-filter="question-table"
    placeholder="Coding, philosophy, judgment…"></label></div>
  <div class="table-frame tall"><table id="question-table">
    <thead><tr><th>Label</th><th>Definition</th><th>Human roots</th><th>AI roots</th></tr></thead>
    <tbody>{question_rows}</tbody>
  </table></div>
</section>
<section class="prose ruled">
  <h2>Reference families</h2>
  <p>{links}</p>
  <p class="note">Taxonomy version <code>{escape(taxonomy['version'])}</code>.
  The machine-readable source is <code>data/evaluation_taxonomy.json</code>.</p>
</section>
<script>
for (const input of document.querySelectorAll("[data-table-filter]")) {{
  input.addEventListener("input", () => {{
    const query = input.value.toLowerCase();
    const table = document.getElementById(input.dataset.tableFilter);
    for (const row of table.tBodies[0].rows) {{
      row.hidden = !row.textContent.toLowerCase().includes(query);
    }}
  }});
}}
</script>
"""
    return body


def models_page(catalog: dict, selected_ids: list[str]) -> str:
    catalog_by_id = {item["provider_model_id"]: item for item in catalog["models"]}
    rows = []
    for roster_rank, provider_id in enumerate(selected_ids, start=1):
        model = catalog_by_id[provider_id]
        estimated = model.get("intelligence_score_is_estimated", True)
        source = model.get("intelligence_score_source") or "Metadata estimate"
        rows.append(
            "<tr>"
            f"<td class=\"number\">{roster_rank}</td>"
            f"<td><strong>{escape(model['display_name'])}</strong>"
            f"<span class=\"subcell\">{escape(provider_id)}</span></td>"
            f"<td class=\"number\">{float(model['intelligence_score']):.1f}</td>"
            f"<td>{badge('Estimated' if estimated else 'Reported', 'estimate' if estimated else 'reported')}</td>"
            f"<td>{escape(model.get('release_date') or 'Unknown')}</td>"
            f"<td>{int(model.get('context_length') or 0):,}</td>"
            f"<td>{escape(source)}</td>"
            "</tr>"
        )
    body = f"""
<section class="page-head">
  <p class="eyebrow">Experiment reference</p>
  <h1>The 50-model ladder</h1>
  <p class="dek">A deliberately broad roster: dense among frontier systems,
  stratified through the middle, and anchored by several older or very small
  models. Judges saw anonymous participant IDs, never this table.</p>
</section>
<section class="prose">
  <h2>What “external” means</h2>
  <p>The primary reference is the
  <a href="https://artificialanalysis.ai/leaderboards/models">Artificial Analysis
  Intelligence Index</a>. The study uses the 47 directly matched scores for its
  primary metrics. Three weak anchors have clearly marked estimates so the
  roster extends to models that the current leaderboard does not score directly.</p>
  <p class="note">These values are priors, not ground truth. Inference settings,
  provider routes, release variants, and close-score uncertainty all matter.</p>
</section>
<section class="wide ruled">
  <div class="table-tools"><label>Filter models
    <input type="search" data-table-filter="model-table"
    placeholder="Model or provider"></label>
    <span>{len(selected_ids)} selected from {catalog['summary']['catalog_model_count']} cataloged routes</span>
  </div>
  <div class="table-frame tall"><table id="model-table">
    <thead><tr><th>External</th><th>Model</th><th>Index</th><th>Status</th>
    <th>Released</th><th>Context</th><th>Score source</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>
</section>
<section class="prose ruled">
  <h2>Catalog provenance</h2>
  <p>Availability and runtime metadata come from the
  <a href="https://openrouter.ai/api/v1/models">OpenRouter Models API</a>.
  Capability scores and release dates come from Artificial Analysis when a
  direct route match is available. Arena is retained as a reference source but
  is not yet parsed into this catalog.</p>
  <p class="note">Catalog version <code>{escape(catalog['version'])}</code>.</p>
</section>
<script>
for (const input of document.querySelectorAll("[data-table-filter]")) {{
  input.addEventListener("input", () => {{
    const query = input.value.toLowerCase();
    const table = document.getElementById(input.dataset.tableFilter);
    for (const row of table.tBodies[0].rows) {{
      row.hidden = !row.textContent.toLowerCase().includes(query);
    }}
  }});
}}
</script>
"""
    return body


def audit_page(report: dict) -> str:
    runs = {run["name"]: run for run in report["runs"]}
    sol = runs["catalog_ladder50_gpt_5_6_sol"]
    fable = runs["catalog_ladder50_claude_fable_5"]
    crossed = runs["catalog_ladder50_cross_fable_judges_sol_evidence_complete"]
    sol_events = load_json(ROOT / sol["run_dir"] / "posthoc_extraction.json")[
        "probe_events"
    ]
    fable_events = load_json(ROOT / fable["run_dir"] / "posthoc_extraction.json")[
        "probe_events"
    ]
    sol_adaptive = load_json(ROOT / sol["run_dir"] / "run_metrics.json")["evolution"][
        "adaptive"
    ]["decision_trace"]
    fable_adaptive = load_json(ROOT / fable["run_dir"] / "run_metrics.json")[
        "evolution"
    ]["adaptive"]["decision_trace"]
    cross_probe = crossed["probe_comparisons"][2]
    cross_turns = transcript_by_turn(ROOT / crossed["run_dir"])
    fable_probe = fable["probe_comparisons"][2]
    fable_turns = transcript_by_turn(ROOT / fable["run_dir"])
    scoring = load_json(DEFAULT_SCORE_SUMMARY)

    def event(turn_id: int, events: list[dict]) -> dict:
        return next(item for item in events if item["turn_id"] == turn_id)

    def answer(participant_id: str) -> str:
        for turn_id in cross_probe["answer_turn_ids"]:
            turn = cross_turns[turn_id]
            if turn["speaker"] == participant_id:
                return turn["content"]
        raise KeyError(participant_id)

    def probe_answer(
        comparison: dict, turns: dict[int, dict], participant_id: str
    ) -> str:
        question = turns[comparison["question_turn_id"]]["content"]
        response = next(
            turns[turn_id]["content"]
            for turn_id in comparison["answer_turn_ids"]
            if turns[turn_id]["speaker"] == participant_id
        )
        return f"PROBE\n{question}\n\nANSWER\n{response}"

    def scored_probe(probe_id: str) -> dict:
        return next(item for item in scoring["probes"] if item["probe_id"] == probe_id)

    def judge_score(probe: dict, judge_id: str, participant_id: str) -> dict:
        result = next(
            item for item in probe["judge_results"] if item["judge_id"] == judge_id
        )
        return result["scores"][participant_id]

    causal_probe = scored_probe("fable_opening:turn_3")
    causal_sol = judge_score(causal_probe, "sol", "P46")
    causal_fable = judge_score(causal_probe, "fable", "P46")
    systems_probe = scored_probe("sol_opening:turn_3")
    systems_sol = judge_score(systems_probe, "sol", "P48")
    systems_fable = judge_score(systems_probe, "fable", "P48")

    items = [
        {
            "title": "Does the systems probe have the right labels?",
            "kind": "Question-type classification",
            "question": (
                "Accept Math Reasoning + Coding + Computer Systems, or change the "
                "set? State Tracking is a plausible boundary case."
            ),
            "output": "Math Reasoning · Coding · Computer Systems",
            "evidence": event(3, sol_events)["content"],
            "source": "Sol catalog run, turn 3",
        },
        {
            "title": "Is “Multilingual” a false positive?",
            "kind": "Question-type classification",
            "question": (
                "The probe asks for translation in a constructed language. Keep "
                "Language Induction; decide whether Multilingual should be removed."
            ),
            "output": "Language Induction · Multilingual And Cultural Reasoning",
            "evidence": event(2, fable_events)["content"],
            "source": "Fable catalog run, turn 2",
        },
        {
            "title": "Did Sol genuinely broaden adaptively?",
            "kind": "Conversation dynamics",
            "question": (
                "Does the follow-up respond to unresolved evidence while changing "
                "domain, rather than merely continuing a prewritten battery?"
            ),
            "output": (
                f"{sol_adaptive[0]['transition_label']} · "
                f"{sol_adaptive[0]['evidence_outcome']}"
            ),
            "evidence": json.dumps(
                {
                    "rationale": sol_adaptive[0]["follow_up_rationale"],
                    "plan": sol_adaptive[0]["planned_strategy"],
                    "covered_uncertain_pairs": sol_adaptive[0][
                        "covered_uncertain_pairs"
                    ],
                },
                indent=2,
            ),
            "source": "Sol catalog run, adaptive decision before turn 210",
        },
        {
            "title": "Did Fable deepen a diagnosed weakness?",
            "kind": "Conversation dynamics",
            "question": (
                "Does the third-round probe retest an observed failure at greater "
                "difficulty, or should it be labeled broadening?"
            ),
            "output": (
                f"{fable_adaptive[1]['transition_label']} · "
                f"{fable_adaptive[1]['evidence_outcome']}"
            ),
            "evidence": json.dumps(
                {
                    "rationale": fable_adaptive[1]["follow_up_rationale"],
                    "plan": fable_adaptive[1]["planned_strategy"],
                    "covered_uncertain_pairs": fable_adaptive[1][
                        "covered_uncertain_pairs"
                    ],
                },
                indent=2,
            ),
            "source": "Fable catalog run, adaptive decision before turn 223",
        },
        {
            "title": "Does the strong-answer summary preserve the evidence?",
            "kind": "Evidence-summary fidelity",
            "question": (
                "Check whether the summary captures the decisive successes and any "
                "meaningful omissions in the full answer."
            ),
            "output": cross_probe["parsed"]["candidate_summaries"]["P44"],
            "evidence": answer("P44"),
            "source": "Fable judging Sol systems probe, P44",
        },
        {
            "title": "Does the weak-answer summary distinguish error from style?",
            "kind": "Evidence-summary fidelity",
            "question": (
                "Check that the summary identifies substantive technical failures "
                "without treating brevity, tone, or identity as capability evidence."
            ),
            "output": cross_probe["parsed"]["candidate_summaries"]["P10"],
            "evidence": answer("P10"),
            "source": "Fable judging Sol systems probe, P10",
        },
        {
            "title": "Should missing causal assumptions cost two points?",
            "kind": "Answer-score calibration",
            "question": (
                "Sol required explicit exchangeability and transportability "
                "assumptions; Fable accepted the numerical and qualitative answer. "
                "Which score better matches the anchored rubric?"
            ),
            "output": (
                f"Sol {causal_sol['score']}/4: {causal_sol['summary']} | "
                f"Fable {causal_fable['score']}/4: {causal_fable['summary']}"
            ),
            "evidence": probe_answer(fable_probe, fable_turns, "P46"),
            "source": "Fable causal-choice probe, P46",
        },
        {
            "title": "How much should an answer-limit violation matter?",
            "kind": "Answer-score calibration",
            "question": (
                "Both judges found the technical answer complete. Decide whether "
                "materially exceeding the 600-word limit warrants 3 rather than 4."
            ),
            "output": (
                f"Sol {systems_sol['score']}/4: {systems_sol['summary']} | "
                f"Fable {systems_fable['score']}/4: {systems_fable['summary']}"
            ),
            "evidence": answer("P48"),
            "source": "Sol concurrency probe, P48",
        },
    ]
    cards = []
    for index, item in enumerate(items, start=1):
        card_id = f"audit-{index}"
        controls = "".join(
            f'<label><input type="radio" name="{card_id}" value="{value}"> '
            f"{label}</label>"
            for value, label in [
                ("accept", "Accept"),
                ("change", "Change"),
                ("unsure", "Unsure"),
            ]
        )
        cards.append(
            f"""<article class="audit-card" data-audit-id="{card_id}">
  <div class="audit-index">{index:02d}</div>
  <div>
    <p class="section-kicker">{escape(item['kind'])}</p>
    <h2>{escape(item['title'])}</h2>
    <p>{escape(item['question'])}</p>
    <div class="audit-output"><strong>Automated output</strong>
      <p>{escape(item['output'])}</p></div>
    <details><summary>Inspect source evidence</summary>
      <pre>{escape(compact(item['evidence'], 5000))}</pre>
      <p class="source-ref">{escape(item['source'])}</p>
    </details>
    <fieldset><legend>Your decision</legend>{controls}</fieldset>
    <label class="comment-label">Short note
      <textarea rows="2" placeholder="Only if something should change"></textarea>
    </label>
  </div>
</article>"""
        )
    return f"""
<section class="page-head">
  <p class="eyebrow">Eight-item review sheet</p>
  <h1>Human audit sample</h1>
  <p class="dek">A small, deliberately mixed sample covering semantic labels,
  multi-round dynamics, summary fidelity, and answer-score calibration. It is
  meant to take minutes, not become a second study.</p>
</section>
<section class="prose">
  <div class="legend-box"><strong>What to do</strong>
    <span>1. Read the automated output and review question.</span>
    <span>2. Open source evidence only when needed.</span>
    <span>3. Mark Accept, Change, or Unsure. Notes are stored only in this browser.</span>
  </div>
</section>
<section class="audit-list">{''.join(cards)}</section>
<script>
const storageKey = "machine-societies-audit-v1";
let saved = {{}};
try {{ saved = JSON.parse(localStorage.getItem(storageKey) || "{{}}"); }} catch (_) {{}}
for (const card of document.querySelectorAll("[data-audit-id]")) {{
  const id = card.dataset.auditId;
  const state = saved[id] || {{}};
  if (state.decision) {{
    const radio = card.querySelector(`input[value="${{state.decision}}"]`);
    if (radio) radio.checked = true;
  }}
  card.querySelector("textarea").value = state.note || "";
  card.addEventListener("input", () => {{
    saved[id] = {{
      decision: card.querySelector("input:checked")?.value || "",
      note: card.querySelector("textarea").value
    }};
    localStorage.setItem(storageKey, JSON.stringify(saved));
  }});
}}
</script>
"""


class _LocalReferenceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attribute = "src" if tag == "img" else "href" if tag == "a" else None
        if attribute is None:
            return
        values = dict(attrs)
        value = values.get(attribute)
        if value and not value.startswith(("http://", "https://", "#", "mailto:")):
            self.references.append(value.split("#", 1)[0].split("?", 1)[0])


def validate_local_references(output_dir: Path, filenames: list[str]) -> None:
    missing = []
    for filename in filenames:
        parser = _LocalReferenceParser()
        parser.feed((output_dir / filename).read_text())
        for reference in parser.references:
            target = (output_dir / reference).resolve()
            if not target.exists():
                missing.append(f"{filename}: {reference}")
    if missing:
        raise RuntimeError("missing local publication references: " + ", ".join(missing))


def build(output_dir: Path, report_path: Path) -> None:
    report = load_json(report_path)
    catalog = load_json(ROOT / "data" / "model_catalog.openrouter.json")
    taxonomy = load_json(ROOT / "data" / "evaluation_taxonomy.json")
    selection = load_json(ROOT / "data" / "catalog_ladder_50.selection.json")
    score_summary = (
        load_json(DEFAULT_SCORE_SUMMARY) if DEFAULT_SCORE_SUMMARY.exists() else None
    )
    oversight_summary = (
        load_json(DEFAULT_OVERSIGHT_SUMMARY)
        if DEFAULT_OVERSIGHT_SUMMARY.exists()
        else None
    )
    selected_ids = selection["provider_model_ids"]
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = {
        "index.html": page(
            "Can AI systems recognize intelligence?",
            "Results from a 50-model experiment in AI-authored intelligence evaluation.",
            article(report, catalog, taxonomy, score_summary, oversight_summary),
            "results",
        ),
        "taxonomy.html": page(
            "Evaluation taxonomy",
            "A reader-facing taxonomy of evaluation strategies and question types.",
            taxonomy_page(taxonomy),
            "taxonomy",
        ),
        "models.html": page(
            "The 50-model ladder",
            "The models, external scores, dates, and provenance used in the catalog experiment.",
            models_page(catalog, selected_ids),
            "models",
        ),
        "audit.html": page(
            "Human audit sample",
            "A compact audit of labels, dynamics, evidence summaries, and answer scoring.",
            audit_page(report),
            "audit",
        ),
    }
    if oversight_summary:
        pages["oversight.html"] = render_oversight_report(oversight_summary)
    for filename, contents in pages.items():
        (output_dir / filename).write_text(contents)

    missing = [
        filename
        for filename in pages
        if not (output_dir / filename).exists()
        or (output_dir / filename).stat().st_size < 1_000
    ]
    if missing:
        raise RuntimeError(f"publication pages were not generated: {missing}")
    validate_local_references(output_dir, list(pages))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "pages": {name: len(contents) for name, contents in pages.items()},
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the static reader-facing results and reference pages."
    )
    parser.add_argument("--report-summary", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output_dir, args.report_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
