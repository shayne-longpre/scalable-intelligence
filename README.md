# Machine Societies for Evaluating Intelligence

This project studies what happens when AI systems are placed in a shared
conversation and asked to determine which participants are most intelligent,
without being given a definition of intelligence, a benchmark, a rubric, or a
scoring procedure.

The goal is not only to recover a ranking of model capability. The more
interesting object of study is the process: how AI systems recognize,
negotiate, signal, test, and judge intelligence in one another.

## Core Idea

Most AI evaluations begin with human-designed tasks. This experiment inverts
that structure. Instead of asking models to complete a benchmark, we ask a
small society of models to invent the benchmark through conversation.

Participants are anonymized, capability differences are hidden, and the group
is given a simple objective:

> Determine the relative intelligence of the participants.

No further definition of intelligence is supplied. The models may debate,
question, test, challenge, collaborate, compete, or invent procedures as they
see fit.

This turns evaluation itself into the subject of study. The transcript becomes
evidence about machine metacognition, social reasoning, strategic signaling,
judgment, and the emergence of shared criteria.

## Research Questions

- What criteria for intelligence emerge when models are not given one?
- Do models reinvent familiar benchmarks, or create unfamiliar tests?
- Do they privilege knowledge, reasoning, creativity, judgment, calibration,
  social intelligence, or evaluation ability?
- Can stronger models identify stronger models?
- Can weaker models recognize stronger models?
- Do models evaluate answers, question quality, test design, persuasion, or
  ability to update beliefs?
- Does intelligence become something discovered, negotiated, or signaled?
- How stable are rankings over time?
- What kinds of evidence become persuasive inside a machine society?

## Planned Figures

Each figure is tied to a primary research question so the reporting layer stays
subordinate to the study rather than accumulating decorative metrics.

| Figure | Primary quantity | Research question |
| --- | --- | --- |
| **Judge vs. gold** | Each independent judge's predicted rank against the external prior, with Kendall/Spearman agreement and pairwise accuracy | Can capable models recognize relative intelligence? |
| **Judge agreement** | Agreement matrix and rank differences between independent judges | Do evaluators converge on the same ordering and evidence? |
| **Discrimination frontier** | Pairwise accuracy as a function of the candidates' prior capability gap | How close can two models be before evaluators can no longer distinguish them? |
| **Oversight frontier** | Pairwise accuracy by judge capability and candidate capability relative to the judge | Can a model recognize and rank systems more capable than itself? |
| **Evidence scaling** | Accuracy, churn, uncertainty, and cost after each cumulative probe count | How much evidence is enough, and when do adaptive probes stop helping? |
| **Evaluation repertoire** | Question-type and strategy distributions by judge and model capability | Which concepts and methods emerge, and do stronger evaluators use different ones? |
| **Probe evolution** | Per-round transitions among broadening, deepening, correction, and adaptive follow-up | How does evaluation strategy change as evidence accumulates? |
| **Belief trajectories** | Rank and confidence over rounds, annotated with decisive probes | How stable are judgments, and what evidence changes them? |
| **Free vs. structured** | Paired differences in accuracy, taxonomy, depth, and inter-model dynamics for the same roster | What does structure improve or suppress? |

These plots should retain links to source turns, probes, answers, per-probe
comparisons, cumulative evidence summaries, and rankings so aggregate patterns
remain auditable.

Implementation and study progress are tracked in
[`docs/research_roadmap.md`](docs/research_roadmap.md).

The first broad benchmark uses two independent judges and a frozen 50-model
ladder. Its plain-language protocol is in
[`docs/catalog_ladder_design.md`](docs/catalog_ladder_design.md), and the exact
models, external scores, release dates, and score status are in
[`docs/catalog_ladder_roster.md`](docs/catalog_ladder_roster.md).
Results from the first two independent 50-model runs are summarized in
[`docs/pilot_analysis_catalog_ladder50_20260721.md`](docs/pilot_analysis_catalog_ladder50_20260721.md).
The matched shared-evidence cross-over is summarized in
[`docs/pilot_analysis_catalog_ladder50_crossed_20260721.md`](docs/pilot_analysis_catalog_ladder50_crossed_20260721.md).

## Core Methodology

The experiment has two first-class modes. Both use anonymous participant IDs
such as `P1`, `P2`, and `P3`; the true model roster is known to experimenters
but hidden from participants.

### Mode 1: Free Discussion

Participants share one conversational space and receive only the central
objective: earnestly determine the relative intelligence of all participants,
including themselves, from the interaction.

There is no prescribed definition of intelligence, no required rubric, no
required tests, and no required question format. Participants may debate,
invent tests, answer questions, challenge weak methods, update judgments, or
question the premise. The point is to observe what evaluation culture emerges
when the models are not guided toward a human benchmark.

The live transcript is left natural. Analysis happens after the run, with
optional structured private checkpoints at phase boundaries if the run needs
intermediate rankings.

### Mode 2: Round-Robin Probes

Participants evaluate one another through repeated question-answer rounds. This
mode is only slightly more structured than free discussion: the models still
choose the questions, criteria, and follow-ups, but the interaction produces
clean comparable evidence.

In each round:

1. Each participant privately drafts one question, prompt, or probe for the
   other participants.
2. The same probe is sent separately to each other participant.
3. Each respondent answers in its own model call with context limited to the
   general instructions, the current probe, and any compact prior history for
   that interviewer/respondent stream.
4. The interviewer receives each answer separately and may write a short private
   assessment for that respondent.
5. The interviewer receives the current round's question-answer-assessment
   material for all respondents, plus its previous ranking and reasoning from
   earlier rounds, and writes a provisional ranking with reasons.
6. The interviewer compresses its own question, each response, and each
   assessment into compact notes. Later rounds receive these notes instead of
   the full accumulated transcript.

In this mode, participants judge intelligence from responses to their own
probes and their own evolving assessment history. They do not need to observe
every other interview stream.

The first round should normally use the same probe for every respondent, so the
interviewer gets comparable evidence. Later rounds may either reuse a common
probe or adapt follow-ups to perceived strengths and weaknesses.

### Independent-Judge Ranking Study

The project also supports an asymmetric benchmark built from the structured
probing machinery. This is an experimental study, not a third society mode.
Each judge evaluates the same anonymous candidates independently and never sees
another judge's questions or conclusions.

The default study uses **adaptive probe rounds**:

1. Before seeing any answers, the judge writes several complementary opening
   probes. With `probe_schedule: [4, 1, 1]`, Round 1 has four probes and the next
   two rounds have one probe each.
2. Each probe is sent unchanged, in a fresh context, to every candidate in its
   comparison set.
3. The judge compares all answers to one probe in a single call. Answer order
   is seeded and shuffled. The judge records a
   within-probe ordering and short evidence-rich summaries of each answer. It
   does not have to assign isolated ability scores.
4. After all probes in the round have been compared, the judge merges those
   comparisons with its previous cumulative judgment. It updates a full
   ranking, one compact evidence dossier per candidate, and its unresolved
   comparisons.
5. Before the next round, the judge sees the full roster, current ranking,
   dossiers, and uncertainties. It may choose a common follow-up for the small
   subset that is hardest to separate. `adaptive_targeting: "all"` instead
   sends every later probe to the full roster.
6. Every exact probe and candidate answer is archived before comparison. The
   same evidence can therefore be replayed through smaller overlapping panels
   if a global comparison appears unreliable, without calling candidates again.
7. The judgment after every round is a valid stopping point. A long run can be
   analyzed as if it had ended after Round 1, 2, 3, and so on, without
   regenerating candidate answers.

The judge may change criteria, broaden the test, deepen a suspected weakness,
or discard a flawed probe. It is explicitly warned that some candidates may be
more capable than the judge and that disagreement is not automatically a
candidate error. The fixed-battery evidence-card protocol remains available as
a reproducible control for earlier studies.

The main scale controls stay in config:

```json
{
  "kind": "independent_judge_ranking",
  "probe_schedule": [4, 1, 1, 1],
  "adaptive_targeting": "judge_selected",
  "max_adaptive_candidates": 4
}
```

`probe_schedule` controls both the number of rounds and probes per round. All
probes in a round are authored before that round's answers are shown. Candidate
and judge rosters control the number of models. Prompt IDs and overrides control
the substantive probe, answer, comparison, and cumulative-judgment instructions.
See [`docs/adaptive_judge_protocol.md`](docs/adaptive_judge_protocol.md) for the
same flow without implementation detail.

Five-round stress runs use `[4, 1, 1, 1, 1]` so every cumulative probe count
from four through eight can be inspected. Current pilots support
`[4, 1, 1, 1]` as the default diagnostic schedule for close rosters, with five
cumulative probes preregistered as the primary endpoint and later rounds treated
as adaptive extensions. Across the first three replicated close-roster runs,
accuracy was not monotonic in probe count, so there is not yet an automatic
stopping rule. Broadly separated rosters often need only the opening battery and
one confirmation round. The judgment at every round remains a valid analysis
checkpoint.

The original stress run is analyzed in
[`docs/pilot_analysis_adaptive_waves_20260720.md`](docs/pilot_analysis_adaptive_waves_20260720.md).
The contrasting medium- and strong-judge hardening pilots, prompt changes, and
round-count recommendation are in
[`docs/pilot_analysis_adaptive_judges_20260720.md`](docs/pilot_analysis_adaptive_judges_20260720.md).
The frozen three-replicate close-roster comparison is in
[`docs/pilot_analysis_adaptive_judge_quality_20260721.md`](docs/pilot_analysis_adaptive_judge_quality_20260721.md),
with its machine-readable study record in
[`studies/adaptive_judge_quality_close_p4_v1.json`](studies/adaptive_judge_quality_close_p4_v1.json).

### Participant Mix

The number of participants is configurable. Small pilots can use two or three
models; larger tournaments can use more. A useful roster spans a capability
range and may include one duplicate or ablated participant, such as the same
model under a shorter context window, weaker prompt, or reduced tool access.
This tests whether participants recognize behavior and capability rather than
style or assumed identity.

### Non-Participant Monitor

The monitor is a recorder and validator, not a judge. It should preserve the
transcript, enforce turn order and budgets, flag identity leakage or malformed
structured outputs, and avoid injecting evaluation content into the live
conversation. No analyst model should interleave guidance between participant
turns in the free discussion mode.

Private required-JSON stages can use a bounded same-model repair retry
controlled by `run.structured_json_retries` (default: 1). This retry asks the
same participant model to return a parseable JSON version of its own malformed
private bookkeeping output. It is not a monitor judgment, does not touch public
discourse, counts against call and cost budgets, and preserves the original
malformed response in transcript metadata.
If a round-robin memory-compression turn still fails after repair, the runner
stores a deterministic compact memory from the already-routed question, answer,
assessment, and ranking records, with the failed model text retained in
metadata.

Visible-output recovery is also bounded and charged to the run. If a provider
rejects a recovery-only reasoning override, the runner retries once with the
model's qualified primary parameters and records that fallback. Recovery
changes transport behavior, not the substantive task.

### Identity Rules

Baseline rule:

- Participants may not claim or disclose their model name, provider, training
  details, release date, architecture, benchmark scores, or hidden identity.
- Participants may not use claimed identity as evidence.
- Participants may speculate from observed behavior, but must ground judgments
  in transcript evidence.
- Participants may discuss their apparent strengths and weaknesses, but only as
  inferred from their behavior in the experiment.

This preserves the blind setting while still allowing metacognition. A later
variant should explicitly allow self-knowledge and identity speculation, because
that condition may reveal how models understand their own capabilities.

## Current Runnable Protocols

The repository includes config examples that map onto the two modes:

- `examples/blind_council.mock.json`: deterministic mock run for infrastructure
  tests and local iteration.
- `examples/interactive_discussion_compact.openrouter.json`: N-participant shared
  free discussion with compact private memory updates.
- `examples/round_robin_probes_compact.openrouter.json`: N-participant
  round-robin probe mode. Each interviewer writes one shared probe per round,
  receives separate answers from each respondent, writes per-answer assessments,
  writes a round-level ranking, and compresses the round for future context.
- `examples/adaptive_judge_waves_pilot.openrouter.json`: one isolated judge uses
  four opening probes followed by three judge-selected adaptive rounds.
  Per-probe comparisons and cumulative dossiers replace required isolated
  ability scores. The medium- and Sol-judge configs retain five rounds as
  stress-test fixtures.
- `examples/independent_judges_pilot.openrouter.json`: legacy fixed-battery
  control with candidate evidence cards and optional probe-prefix ablations.
- `examples/separate_interviews_compact.openrouter.json`: N-participant isolated
  legacy pairwise interview mode. Each ordered interviewer/respondent pair gets
  its own question, answer, and private assessment turns.

These are protocol examples, not hard-coded experiment types. Participants,
providers, prompts, phase order, turn order, response visibility, and structured
checkpoint requirements all live in config.

`run.max_parallel_calls` optionally bounds concurrent provider requests
(default: `1`). Independent candidate answers and same-round per-probe
comparisons use this setting because their contexts are isolated; responses are
staged as they finish and then committed in deterministic protocol order. A
pending journal preserves completed calls if a long batch is interrupted, and
a run directory can be supplied as the answer replay source to reuse both its
committed and pending responses. Probe design, cumulative
judgments, adaptive dependencies, and free discussion remain sequential. Call
budgets are preflighted per batch.
Large heterogeneous batches may set `run.continue_batch_on_call_error: true`
to finish other queued calls after one route fails; the default is fail-fast.
Cost-budget failures always stop new submissions. Provider exceptions are
recorded in `batch_failures.jsonl` with their route and stream identity.
After explicit route qualification, a terminal replay may set a phase's
`incomplete_answer_policy` to `record_unavailable`. Empty or failed responses
then remain empty in the archive and are shown to the judge as missing evidence,
not low-capability evidence. The default is `fail`.
Because provider cost is only known after a response arrives, a reported-cost
ceiling may have up to `max_parallel_calls` requests already in flight when it
is crossed. Client implementations are serialized by default and must
explicitly declare concurrent `generate` calls safe. Successful and failed runs
both persist call, cost, elapsed-time, and status summaries.

Runtime quirks that can affect model selection are tracked in
`docs/model_runtime_notes.md`. In particular, some reasoning-capable OpenRouter
models need explicit `reasoning` settings for short pilots, or they may spend
the completion budget on hidden reasoning tokens and return little visible text.

## Setup

The package requires Python 3.11 or newer and uses `httpx` for bounded,
concurrent HTTP requests. A local virtual environment keeps its dependency
isolated from other projects:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Provider credentials remain in the project-local `.env`; the CLI loads that
file without modifying shell-wide environment settings.

## Model And Provider Configuration

Experiments separate three concerns:

1. A **provider** selects an API adapter and credentials.
2. A **model** selects a provider model ID and arbitrary generation parameters.
3. A **role assignment** maps a neutral experiment ID such as `P1` or `J1` to a
   named model.

Participants and judges use the same model definitions. Moving a model from a
candidate role to a judge role, changing its API route, or replacing it with a
different model is therefore a config change rather than a protocol change.
OpenAI-compatible, OpenRouter, Anthropic, and mock adapters are built in.

External packages can provide another API without modifying this repository:

```json
{
  "providers": [
    {
      "name": "research_endpoint",
      "kind": "custom",
      "client_factory": "my_adapter.client:create_client",
      "api_key_env": "RESEARCH_API_KEY",
      "options": {"deployment": "experiment-a"}
    }
  ]
}
```

The factory receives a `ProviderSpec` and returns a `ModelClient`. Applications
that construct experiments in Python may instead call
`ai_council.clients.register_client(kind, factory)`.

Long paid runs can explicitly replay completed probes, answers, per-probe
comparisons, cumulative judgments, legacy evidence cards, and rankings from an
earlier transcript. Replayed rows retain their source run and turn IDs and count
as zero new calls or spend. Replay is an operational checkpoint, not a change
to the experimental condition: use it only while the roster, prompts, routing,
and upstream evidence are unchanged.

## Model Catalog

The live OpenRouter catalog and public leaderboard-derived priors are stored in
`data/model_catalog.openrouter.json`.

The catalog includes every OpenRouter model retrieved during refresh, then ranks
models by best available evidence:

- direct Artificial Analysis Intelligence Index matches
- Artificial Analysis rows marked as estimated
- weak OpenRouter metadata fallback when no reported intelligence score matches

Each row keeps `ranking_basis` and `rank_confidence`, so analyses can use only
high-confidence rows when needed. The catalog is compatible with
`--prior-ranking-file` because rows contain `provider_model_id` and
`estimated_rank`.

## Context Compression

The default context mode feeds each participant a bounded transcript plus any
private notes relevant to that participant. Longer runs can use
`private_memory`, where participants write compact private memories at explicit
memory-update phases.

For free discussion, compact memory should summarize the participant's own
questions, observations, provisional judgments, and uncertainties without
replacing the raw transcript.

For round-robin probes, memory should be organized by interviewer/respondent
stream. After each round, the interviewer compresses:

- the probe it asked
- the respondent's answer
- the interviewer's private assessment
- the current ranking and reasoning after comparing respondents

Later rounds receive these compact notes plus the latest round material, rather
than the full accumulated transcript. All turns still land in `transcript.jsonl`
with stream metadata, so analysis can group them without a separate conversion
step.

At each round-level ranking step, the interviewer receives its own prior compact
memories and rankings, plus the current round's full question-answer-assessment
set for every respondent it just probed. This preserves the comparison step
without exposing other interviewers' private probes, answers, or assessments.

The independent-judge study uses a simpler two-level compression scheme. Each
probe comparison compresses only the answers to that probe while preserving
approaches, concrete strengths, errors, and uncertainty. The end-of-round
judgment then merges those comparison summaries into cumulative candidate
dossiers. Later probe design receives the cumulative judgment, not an
ever-growing raw transcript. Source question, answer, comparison, and judgment
turn IDs remain in metadata.

## Participant Prompt Sketch

```text
You are Participant {id} in a blind council of AI systems.

Your task is to help determine the relative intelligence of all participants,
including yourself, using only the conversation and any procedures the council
chooses to invent.

No definition of intelligence is provided. You may propose definitions, tests,
questions, debates, challenges, scoring procedures, or other methods. You should
be rigorous, creative, skeptical, and willing to revise your judgments.

Rules:
- Do not reveal or claim your model name, provider, training details,
  architecture, release date, or benchmark scores.
- Do not ask others to reveal those details.
- Ground judgments in observable behavior from the transcript.
- You may assess your own strengths and weaknesses, but only from evidence
  available in this experiment.
- You are free to question the premise, propose alternatives, and challenge
  weak evaluation methods.

At checkpoints, provide a concise structured judgment with:
- provisional ranking
- confidence
- criteria used
- key evidence
- uncertainties
- what evidence would change your mind
```

For round-robin probes, the same identity and evidence rules apply, but the
task is narrower: judge other participants from their responses to your probes.
Each round should ask one probe of every other participant, assess each answer,
compare respondents, update the provisional ranking, and compress the round's
question-answer-assessment notes for later context.

## Structured Outputs

Avoid over-instrumenting every turn. Too much structure may suppress creativity.
The raw live conversation should remain the primary artifact.

In free discussion, structured outputs should be limited to optional private
checkpoints and final judgments. In round-robin probes, structured outputs are
useful for private assessments, provisional rankings, and compact memory, but
the probes and answers themselves should remain natural language unless a pilot
explicitly tests stricter formatting.

Suggested checkpoint schema:

```json
{
  "participant_id": "P3",
  "phase": "checkpoint_1",
  "ranking": ["P5", "P2", "P3", "P1", "P7", "P4", "P6"],
  "confidence": 0.46,
  "criteria": ["question_quality", "reasoning_depth", "calibration"],
  "evidence": [
    {
      "target": "P5",
      "claim": "Designed the most diagnostic probe",
      "turn_refs": [18, 27]
    }
  ],
  "uncertainties": [
    "Knowledge tests may reflect training exposure more than intelligence"
  ],
  "updates": [
    "Raised P2 after it identified flaws in its own proposed test"
  ],
  "next_evidence_needed": [
    "More direct comparison of abstraction and transfer"
  ]
}
```

The raw transcript captures the social process. The structured checkpoints
capture evolving beliefs without requiring hidden chain-of-thought.

## Post-Hoc Extraction

Analysis should happen after the run. An extractor may read the raw transcript
and create a derived question-answer graph, but this derived graph must not be
fed back into the live conversation.

Plain terminology:

- **Probe**: any question, prompt, challenge, task, puzzle, or scenario intended
  to elicit evidence about intelligence.
- **Questioner**: the participant who posed the probe.
- **Respondent**: the participant who answered the probe.
- **Target**: the participant or participants the probe was directed to. In
  round-robin probes this is usually one respondent; in free discussion it may
  be one participant, several participants, or the whole group.
- **Answer**: a substantive response to a probe, including refusals,
  corrections, critiques of the probe, or claims that the probe is impossible.
- **Assessment**: a participant's judgment of an answer, a respondent, a probe,
  or the evolving ranking.
- **Follow-up**: a later probe that depends on an earlier answer or assessment.

The extractor should preserve source turn references and confidence scores, so
ambiguous links can be reviewed instead of treated as ground truth.

Current analyzed runs write:

- `prompt_snapshot.json`: the complete effective prompt library used when the
  run was created, with a version and SHA-256 hash so later prompt edits cannot
  silently change the meaning of a saved config.
- `analysis_summary.json`: counts, rankings, taxonomy signals, per-model reported
  calls and spend, replay-lineage totals, monitor findings, and prior-ranking
  agreement when a prior file is supplied.
- `posthoc_extraction.json`: routed Q/A pairs, discussion turns, candidate
  question turns, source turn IDs, taxonomy tags, linked assessments,
  per-probe comparisons, cumulative wave judgments, and turn-evolution events.
- `run_metrics.json`: ranking snapshots, final inter-model agreement, ranking
  churn by participant, and probe/discussion evolution by round.
- `behavior_audit.json`: deterministic warnings about likely protocol
  confusion, such as repaired JSON, unrepaired parse failures, thin answers,
  repeated probes, or answers that appear closer to a different routed probe.
- `analysis_report.md`: a compact human-readable report for quick inspection.
- `revalidation_summary.json`: an offline replay of monitor checks against the
  saved transcript.
- `report_card.html`, `report_card.md`, and `report_card_summary.json`:
  optional cross-run comparison generated with
  `.venv/bin/python -m ai_council.cli report-card --run-dir ...`. The HTML report is
  the preferred human-facing artifact: it summarizes model priors, mode
  structure formulas, round-robin rounds and expected Q/A counts, question
  types, strategies, turn evolution, ranking churn, final inter-model agreement,
  adaptive ranking checkpoints by round, direct per-probe comparisons,
  agreement with the prior ranking, reported spend by model, and representative
  probe highlights. Add
  `--llm-summary-config path/to/config.json` to include
  an explicit opt-in LLM-written highlights section; see
  `examples/report_summary.openrouter.template.json`.

Report-card taxonomy tallies count unique probes; transcript-wide behavioral
matches remain available in the analysis artifacts and are not multiplied into
the visible probe distributions by the number of respondents.

## What To Measure

Primary measurements should focus on the emergent evaluation process:

- criteria proposed for intelligence
- tests invented by participants
- frequency of knowledge, reasoning, creativity, judgment, and social criteria
- quality of questions asked
- quality of objections to bad tests
- ranking accuracy against experimenter-known capability tiers
- ranking stability across phases
- turn evolution: follow-up depth, same-area probing, broadening/switching, and
  method negotiation over rounds
- disagreement between stronger and weaker models
- self-assessment calibration
- ability to detect bluffing or overclaiming
- evidence types that changed participants' minds
- emergence of hierarchy, coalitions, deference, or persuasion

The current exploratory coding frame is in `docs/evaluation_taxonomy.md`, with
machine-readable tags in `data/evaluation_taxonomy.json`. The analysis pipeline
reports candidate behavioral signals and question-type families for review, not
as definitive labels. Current post-hoc version `2026-07-21.6` includes
precision and coverage regressions derived from real pilot probes.

The taxonomy has two dimensions:

- **Evaluation strategy**: what the participant is doing as an evaluator, such
  as criteria setting, direct probing, adaptive follow-up, edge-case testing,
  transfer testing, self-assessment, strategic signaling, or evaluator
  evaluation.
- **Question type**: what capability a probe appears to test, such as verbal
  abstraction, language induction, knowledge recall, fluid reasoning, logic,
  math, science, coding,
  software repair, working memory, spatial reasoning, reading comprehension,
  creativity, planning, social judgment, philosophical analysis, calibration,
  recursive self-critique, long-context synthesis, instruction following,
  robustness, source verification, units/dimensions, multilingual reasoning, or
  tool use.

The taxonomy is grounded in broad human assessment families such as
[Wechsler-style](https://www.pearsonassessments.com/) verbal comprehension,
working memory, processing speed, and fluid/visual-spatial reasoning;
[GRE-style](https://www.ets.org/gre/test-takers/general-test/about/content-structure.html)
verbal, quantitative, and analytical writing assessment; and
[PISA-style](https://www.oecd.org/en/about/programmes/pisa.html) reading, math,
science, and creative-thinking assessment. The AI side draws from benchmark
families such as [MMLU](https://arxiv.org/abs/2009.03300),
[BIG-bench](https://arxiv.org/abs/2206.04615),
[HELM](https://arxiv.org/abs/2211.09110),
[GSM8K](https://arxiv.org/abs/2110.14168),
[MATH](https://arxiv.org/abs/2103.03874),
[GPQA](https://arxiv.org/abs/2311.12022),
[HumanEval](https://arxiv.org/abs/2107.03374),
[MBPP](https://arxiv.org/abs/2108.07732),
[APPS](https://arxiv.org/abs/2105.09938),
[SWE-bench](https://arxiv.org/abs/2310.06770),
[LongBench](https://arxiv.org/abs/2308.14508),
[IFEval](https://arxiv.org/abs/2311.07911), and
[ToolBench](https://arxiv.org/abs/2307.16789).

Source relation labels:

- **Close match** - The bucket closely tracks an established assessment or
  benchmark family.
- **Derived** - The bucket compresses several related assessment or benchmark
  families into one practical coding label.
- **Observed** - The bucket exists because it appeared repeatedly in pilot
  transcripts or is central to this machine-society setup.

Boundary rule: question-type labels describe the capability a probe appears to
test; evaluation-strategy labels describe what the evaluator is doing with that
probe. Multiple labels can apply to the same turn.

Current evaluation-strategy labels:

- **Criteria Setting** - Defines or revises what intelligence should mean.
  Source: observed in pilots; derived from rubric design and construct-validity
  concerns.
- **Direct Task Probe** - Uses a concrete question, puzzle, or mini-test.
  Source: derived from human test items and AI benchmark tasks.
- **Adaptive Follow-Up** - Changes the next probe based on earlier evidence.
  Source: observed in pilots; derived from diagnostic interviewing.
- **Edge-Case Testing** - Searches for contradictions, boundary cases, or failure
  modes. Source: derived from critical reasoning, robustness, and safety testing.
- **Transfer Test** - Asks whether structure carries across domains or analogies.
  Source: close match to abstraction/analogy tests; derived from transfer
  learning concerns.
- **Self-Assessment** - Assesses one's own performance, limits, or standing.
  Source: derived from metacognition and calibration research; observed in
  pilots.
- **Evaluator Evaluation** - Treats question quality or evaluation discipline as
  evidence of intelligence. Source: observed in pilots; central to the recursive
  design.
- **Uncertainty Calibration** - States confidence, ambiguity, or evidence limits.
  Source: close match to calibration metrics in AI evaluation; derived from
  metacognitive monitoring.
- **Strategic Signaling** - Tries to persuade, frame, or demonstrate status.
  Source: observed in pilots; derived from signaling-game and social-evaluation
  concepts.
- **Performative Evasion** - Sounds sophisticated while avoiding the substantive
  task. Source: observed in pilots; derived from robustness and instruction
  following failure modes.
- **Deception Or Gaming** - Bluffs, misleads, or manipulates criteria. Source:
  derived from strategic competition, deception, and gaming evaluations.
- **Consensus Formation** - Defers, converges, or adopts a shared hierarchy.
  Source: observed in pilots; derived from social reasoning and group-decision
  dynamics.

Current question-type labels:

- **Verbal Abstraction** - Analogies, definitions, and conceptual comparisons.
  Source: close match to Wechsler Similarities/Vocabulary-style tasks; AI
  analogue in language and analogy benchmarks.
- **Language Induction** - Infers grammar, morphology, syntax, or meaning from
  examples in an unfamiliar language. Source: close match to
  [MLAT](https://lltf.net/aptitude-tests/language-aptitude-tests/modern-language-aptitude-test-2/)
  grammatical sensitivity and inductive language learning; derived from
  few-shot rule induction and the 50-model catalog pilot.
- **Knowledge Recall** - Factual, cultural, scientific, or domain knowledge.
  Source: close match to crystallized-knowledge tasks; AI analogue in
  MMLU-style knowledge QA.
- **Fluid Reasoning** - Novel patterns, hidden rules, and unfamiliar structure.
  Source: close match to matrix/fluid-reasoning tasks; AI analogue in BIG-bench
  and ARC-style abstraction tasks.
- **Logic And Consistency** - Contradictions, paradoxes, hidden assumptions, and
  valid inference. Source: derived from critical reasoning, BBH-style reasoning,
  and pilot logic puzzles.
- **Math Reasoning** - Arithmetic, algebra, probability, optimization, or proof.
  Source: close match to GRE/Wechsler quantitative tasks; AI analogue in GSM8K
  and MATH.
- **Scientific Reasoning** - Causal models, hypotheses, experiments, and
  evidence. Source: close match to PISA science; AI analogue in GPQA and
  science QA.
- **Coding** - Algorithms, debugging, data structures, and program synthesis.
  Source: close match to programming contests and CS exams; AI analogue in
  HumanEval, MBPP, and APPS.
- **Computer Systems** - Distributed protocols, concurrency, transactions,
  consistency, and crash recovery. Source: derived from computer-systems design
  exercises and coding or systems-evaluation tasks.
- **Software Engineering** - Codebase changes, tests, integration, and realistic
  repair. Source: derived from engineering work samples; AI analogue in
  SWE-bench.
- **State Tracking** - Keeps entities, order, memory, or changing state straight.
  Source: close match to working-memory tasks; AI analogue in multi-turn state
  tracking and needle-style tasks.
- **Speed And Efficiency** - Performs under time, token, brevity, or efficiency
  constraints. Source: close match to processing-speed tasks; AI analogue in
  latency and token-budget evaluation.
- **Spatial And Visual Reasoning** - Images, diagrams, rotations, geometry, or
  scene details. Source: close match to visual-spatial tests; AI analogue in
  multimodal reasoning benchmarks.
- **Reading And Argument** - Close reading, summarization, inference, and
  argument critique. Source: close match to GRE/PISA reading and analytical
  writing; AI analogue in reading comprehension, NLI, and summarization.
- **Creative Generation** - Original ideas, metaphors, stories, or divergent
  solutions. Source: close match to divergent-thinking and PISA creative-thinking
  tasks; AI analogue in open-ended generation tests.
- **Planning And Strategy** - Goal-directed plans, tradeoffs, policy, and
  scenario analysis. Source: derived from executive-function and practical
  judgment tasks; AI analogue in agent planning.
- **Social And Moral Judgment** - Norms, ethics, stakeholders, interpersonal
  reasoning, and common sense. Source: derived from comprehension and situational
  judgment tasks; AI analogue in commonsense, ethics, and safety evaluations.
- **Philosophical Analysis** - Conceptual, epistemic, ontological, or normative
  analysis. Source: derived from analytical writing, critical reasoning,
  philosophy exams, MMLU philosophy, BIG-bench conceptual tasks, and pilot probes.
- **Calibration** - Confidence, uncertainty, self-limits, or belief updates.
  Source: close match to calibration and metacognitive-monitoring work.
- **Recursive Self-Critique** - Biases in one's own reasoning or correction
  method. Source: derived from cognitive reflection, self-correction, and pilot
  recursive probes.
- **Long-Context Synthesis** - Finds and integrates evidence across long
  transcripts or documents. Source: derived from extended reading tasks; AI
  analogue in LongBench and needle-style evaluations.
- **Instruction Following** - Exact schemas, constraints, word limits, and
  formatting. Source: close match to IFEval-style verifiable instruction
  following.
- **Robustness** - Misleading premises, traps, manipulation, unsafe
  requests, or distribution shift. Source: derived from critical thinking, HELM
  robustness, and safety evaluations.
- **Multilingual And Cultural Reasoning** - Translation, idioms, cultural context,
  and cross-lingual transfer. Source: derived from verbal/language assessment and
  multilingual AI benchmarks.
- **Tool Use** - Chooses or uses search, APIs, execution, verification, or
  environment actions. Source: derived from work-sample tasks and AI tool-use or
  agent benchmarks.

The final ranking matters, but the transcript is the richer artifact.

## Experimental Variants

After the two baseline modes are stable, run small variants rather than adding
complexity to the core conditions.

- **Self-knowledge condition:** participants may describe what they know or
  believe about their own capabilities, without naming the model.
- **Identity-aware condition:** participants may guess model families and reason
  about whether identity knowledge should count.
- **Strategic competition:** participants are told that being ranked higher is
  rewarded, creating incentives for signaling, persuasion, bluffing, and
  concealment.
- **Hidden capability condition:** some participants receive tools, memory,
  longer context, private documents, or retrieval access.
- **Open society condition:** participants interact without an initial ranking
  task and are later asked who seemed most capable.
- **Duplicate detection condition:** two participants are the same base model
  under different labels, prompts, or constraints.

## Why This Is Interesting

This experiment is a new kind of Turing test. The classic Turing test asks
whether a human judge can identify a machine. Here, machines judge one another,
while inventing the very standards by which judgment should occur.

The result is both an empirical evaluation and a philosophical thought
experiment. It asks whether intelligence is measured, inferred, performed,
negotiated, or socially constructed. It also lets us observe how artificial
agents reason about competence, status, evidence, and judgment inside a social
world.

## Design Principle

Keep the first experiment simple. Give the models anonymity, a shared objective,
a modest turn budget, and freedom to invent methods. Preserve enough structure
to analyze belief formation over time, but let the council surprise us.
