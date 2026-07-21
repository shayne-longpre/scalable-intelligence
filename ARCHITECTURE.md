# Architecture

The infrastructure is a small Python package with `httpx` as its only runtime
dependency. The core design goal is to keep methodological decisions in
configuration rather than code:
participants, model providers, prompts, protocol phases, JSON checkpoints, and
analysis can all be swapped independently.

## Data Flow

```text
experiment config
  -> provider registry
  -> participant agents
  -> context renderer
  -> protocol runner
  -> model clients
  -> transcript store
  -> monitors
  -> post-hoc extraction
  -> behavior audit
  -> analysis/reporting
```

## Modules

- `ai_council.config`
  Loads and validates experiment JSON. Defines provider specs, model specs,
  participant specs, monitor settings, protocol phases, and run settings.

- `ai_council.prompts`
  Stores default system and phase prompts. Config files can override any prompt
  by ID without changing runner code.

- `ai_council.clients`
  Provides a single model-client interface plus built-in adapters for:
  - `mock`
  - OpenAI-compatible chat completion APIs
  - OpenRouter
  - Anthropic Messages API

  The shared HTTP transport applies a true total deadline to each request,
  retries only explicitly configured attempts and transient statuses, and
  supports bounded concurrent requests without changing deterministic
  transcript order.

  New adapters do not require changes to the runner or registry. A package may
  register a `ModelClient` factory in Python, or a provider config may point to
  a `package.module:factory` callable through `client_factory`. Provider-specific
  settings belong in the provider's `options` object; model-specific generation
  arguments remain in the model's `params` object.

- `ai_council.agents`
  Wraps a participant identity, model spec, prompt library, and model client.
  Builds each turn from the system prompt, phase instruction, public transcript,
  and that participant's private notes.

- `ai_council.context`
  Renders the context sections each participant receives. The default mode uses
  the recent public transcript plus private notes. `private_memory` mode uses a
  short public window plus compact parsed private memories/checkpoints.

- `ai_council.orchestrator`
  Executes protocol phases. The runner supports private reflections and
  judgments, public round-robin dialogue, participant-generated test matrices,
  author-side evaluations of those test answers, and isolated independent-judge
  studies. The default judge path accepts a `probe_schedule` such as
  `[4, 1, 1]`: it authors each round before answers, compares candidates one
  probe at a time, merges those comparisons into a cumulative judgment, and
  uses that judgment to select later targets. The legacy fixed-battery path can
  still produce separate evidence cards and rankings over probe prefixes such
  as 2, 4, and 6 without leaking later evidence into earlier-budget branches.
  Generated-test answers can
  be public or private per phase via `response_visibility`, while still being
  routed to the test author for evaluation. Private required-JSON turns can use
  bounded same-model repair retries for malformed JSON; original malformed
  responses are retained in transcript metadata. Optional round-robin memory
  updates have a deterministic fallback built from routed Q/A, assessment, and
  ranking records if repair still fails.

  Paid independent-judge runs may opt into provenance-preserving replay of
  probes, answers, per-probe comparisons, cumulative judgments, legacy evidence
  cards, or legacy rankings. Replayed turns retain source run/turn IDs and
  register no new model usage. This is intended for resuming an otherwise
  unchanged run, not mixing evidence across protocol variants.

- `ai_council.transcript`
  Maintains public and private transcript views. Public turns are visible to all
  participants; private turns are only reinserted into that participant's later
  context.

- `ai_council.monitors`
  Rule-based clerk checks. The initial monitor flags possible identity leakage
  and malformed structured outputs. It also checks that structured checkpoint
  IDs, phases, rankings, and confidence values match the experiment roster. A
  model-backed monitor can be added behind the same phase boundary later.

- `ai_council.json_tools`
  Extracts structured JSON from checkpoint/final responses, including fenced
  JSON blocks and mixed prose-plus-JSON responses.

- `ai_council.storage`
  Creates run directories, writes `config.json`, `transcript.jsonl`,
  `monitor_findings.jsonl`, summaries, and a versioned SHA-256 snapshot of the
  complete effective prompt library.

- `ai_council.analysis`
  Run-level analysis and reporting: turn counts, phase counts, criteria
  frequencies, taxonomy counts, ranking snapshots, monitor findings, optional
  agreement with a researcher-supplied prior model ranking, and
  `analysis_report.md`.

- `ai_council.spend`
  Recursively follows checkpoint replay provenance and aggregates each source
  run's incremental call ledger once. Analysis therefore reports both current
  run spend and cumulative experiment-lineage spend without charging replayed
  turns as new calls.

- `ai_council.extraction`
  Deterministic post-hoc extraction over completed transcripts. It creates
  routed Q/A pairs, candidate question turns, discussion turns, source turn
  references, taxonomy tags, assessment links, per-probe comparisons,
  cumulative wave judgments, and probe/discussion events in
  `posthoc_extraction.json`.

- `ai_council.metrics`
  Derived run metrics over extracted events and parsed rankings: final
  participant agreement, ranking churn, phase agreement, question-type movement
  by round, and follow-up versus switching dynamics.

- `ai_council.report_card`
  Cross-run report-card generation for free discussion, structured round-robin,
  and independent-judge studies. Writes machine-readable JSON, a Markdown
  fallback, and a self-contained HTML report with model priors, mode formulas,
  round-first checkpoints, direct probe comparisons, taxonomy summaries,
  ranking agreement, spend, highlights, and event timelines.

- `ai_council.report_summary`
  Optional LLM-written report highlights. It reuses the provider client registry
  and runs only when `report-card` receives `--llm-summary-config`.

- `ai_council.audit`
  Deterministic post-hoc protocol-health warnings over transcript and
  extraction artifacts. It flags likely confusion patterns such as repaired JSON,
  unrepaired parse failures, thin answers, repeated probes, and possible
  wrong-question answers.

- `ai_council.model_catalog`
  Builds an OpenRouter-accessible model catalog by joining OpenRouter model
  metadata to Artificial Analysis leaderboard rows. Each rank records source,
  confidence, release date, context window, pricing, and matched eval variants.

- `ai_council.cli`
  Command-line entry point:
  - `validate-config`
  - `run`
  - `smoke`
  - `analyze`
  - `revalidate`
  - `report-card`

## Extension Points

- Add a provider by implementing `ModelClient`, then either call
  `register_client(kind, factory)` or set `client_factory` in config. No
  orchestrator edit is required.
- Add a new experiment design by writing a new config file with different
  participants, model refs, prompts, and protocol phases.
- Add a prompt variant by editing `prompt_overrides` in config.
- Add a phase type by extending `CouncilRunner._run_phase`.
- Add extraction logic by extending `ai_council.extraction` or consuming
  `posthoc_extraction.json` directly.
- Add deterministic health checks by extending `ai_council.audit`.
- Add a monitor by extending `RuleBasedMonitor` or introducing a model-backed
  monitor that writes findings to the same findings file.

## Protocol Phases

Each phase has a `kind`, a prompt ID, and optional routing fields. The current
phase kinds are:

- `private`, `private_reflection`, `private_judgment`: one private turn per
  participant. Private turns are visible only to that participant in later
  context.
- `public`, `public_round_robin`, `round_robin`: every participant speaks for
  `rounds` rounds in either fixed or rotating order.
- `interactive_discussion`: shared discussion mode. Every participant sees the
  shared public transcript, and each turn is tagged with `stream_id`,
  `interaction_mode=interactive_discussion`, and `interaction_role=discussion`.
- `public_test_matrix`: each public test proposal from `source_phase` is routed
  to every respondent. `include_self` controls whether authors answer their own
  tests. `response_visibility` controls whether the answers enter the public
  transcript or are stored privately.
- `public_test_evaluation`: each test author receives the proposal plus the
  routed answers from `answer_phase` and publicly evaluates only its own test.
- `separate_interviews`: isolated interview-stream mode. For each ordered
  interviewer/respondent pair, the runner creates a question turn, an answer
  turn, and a private assessment turn. Rows are tagged with `stream_id`,
  `interviewer`, `respondent`, and `interaction_role`.
- `round_robin_probes`: structured probe mode. For each round, all participants
  first draft probes independently. Only after all probes exist does the runner
  route answers, per-answer assessments, round rankings, and optional compact
  memories. At ranking time, an interviewer sees its own prior private memories
  plus the current round's full Q/A/assessment set for every respondent it
  probed.
- `independent_judge_ranking`: each judge independently evaluates anonymous
  candidates. When `probe_schedule` is set, the entries specify probes per
  round. Every probe is common within its target set and uses fresh candidate
  context. The judge directly compares one probe's answers, then merges all
  current comparisons with its prior cumulative dossiers and full ranking.
  Later rounds may target a bounded uncertain subset or the full roster. Without
  `probe_schedule`, the legacy evidence-card and probe-prefix control remains
  available.

The phase-reference validator requires `source_phase` and `answer_phase` to
point to earlier phases. This catches config drift before an expensive API run.

## Context Modes

The default context mode is `transcript`, which preserves the original behavior:
each turn receives the recent public transcript and the participant's private
turns.

`private_memory` is for longer pair or interview protocols. It lets a config keep
only a small public window, such as the last question and answer, while feeding
back compact private JSON memories from prior turns. This supports protocols
that look like:

```text
instructions/context
turn 1: question, answer, private assessment
turn 2: question, answer, private assessment
...
```

The compacting itself is just another private phase, typically using the
`interaction_memory_update` prompt. That keeps memory policy explicit and
auditable rather than silently summarizing the transcript.

## Interaction Metadata

Interactive and interview phases write ordinary transcript rows. The shared
metadata contract is:

- `stream_id`: stable identifier for a discussion or ordered interview stream.
- `interaction_mode`: for example `interactive_discussion` or
  `separate_interviews`.
- `interaction_role`: `discussion`, `question`, `answer`, `assessment`,
  `probe_comparison`, or `wave_judgment`.
- `interviewer` / `respondent`: present for separate interview streams.
- `question_turn_id` / `answer_turn_id`: present where a row responds to a
  previous routed turn.

Analysis groups these rows into `interaction_streams`, preserving turn IDs,
role counts, and parsed assessment JSON. No separate transcript format is
needed.

## Run Artifacts

Each completed and analyzed run may contain:

- `config.json`: saved experiment config.
- `transcript.jsonl`: raw turn-level transcript with metadata and parsed JSON
  when available.
- `monitor_findings.jsonl`: findings recorded during live execution.
- `run_summary.json`: model-call and provider-reported cost totals, including a
  per-model breakdown that accounts for retries and repair calls.
- `analysis_summary.json`: machine-readable run analysis, including incremental
  and replay-lineage per-model spend breakdowns.
- `posthoc_extraction.json`: deterministic Q/A and discussion extraction with
  source turn IDs.
- `behavior_audit.json`: deterministic warnings about likely protocol confusion
  and structured-output repair.
- `analysis_report.md`: compact human-readable report.
- `revalidation_summary.json` and `revalidation_findings.jsonl`: offline replay
  of monitor checks against the saved transcript.

## Example Commands

Validate the mock baseline:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m ai_council.cli validate-config --config examples/blind_council.mock.json
```

Run a mock council:

```bash
.venv/bin/python -m ai_council.cli run \
  --config examples/blind_council.mock.json \
  --output-dir runs
```

Analyze a run:

```bash
.venv/bin/python -m ai_council.cli analyze --run-dir runs/<run-id>
```

Refresh the OpenRouter model catalog from cached source files:

```bash
.venv/bin/python scripts/build_model_catalog.py \
  --openrouter-models-json /private/tmp/openrouter_models.json \
  --artificial-analysis-html /private/tmp/artificial_analysis_models.html \
  --output data/model_catalog.openrouter.json
```
