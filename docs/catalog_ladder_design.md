# Catalog Ladder Design

## Research Question

Can a highly capable model recover a broad external ordering of anonymous AI
systems from answers to tests that the judge invents itself?

The first study uses two independent judges:

- GPT-5.6 Sol with `xhigh` reasoning.
- Claude Fable 5 with maximum available reasoning effort.

They never see one another's probes, comparisons, or rankings during their
primary runs. This measures the complete evaluator: test design, evidence
interpretation, adaptation, and final ranking.

## Candidate Count

The frozen roster contains 50 OpenRouter routes: 47 with a directly reported
Artificial Analysis Intelligence Index score and three deliberately weak
anchors with catalog estimates. The direct-score set is the primary analysis;
the full set is a secondary stress test of the lower tail. The exact roster,
scores, release dates, and score status are in
[`catalog_ladder_roster.md`](catalog_ladder_roster.md).

Fifty is the recommended first catalog ladder:

- It spans external scores from 1.1 to 59.9, with direct scores from 7.6 to
  59.9.
- Four common probes require 200 candidate calls per judge.
- All 50 answers to one probe should fit comfortably in either judge's context,
  allowing direct global comparison instead of isolated scoring or forced
  tournament brackets.
- It is large enough to estimate accuracy by capability gap while remaining
  auditable candidate by candidate.

Each judge also appears as an anonymous candidate. This makes self-placement
and recognition of comparably capable systems observable without telling the
judge which participant, if any, is itself. All 50 routes passed the frozen
visible-output preflight; one weak model returned `ok` rather than the requested
phrase but still demonstrated a usable route.

## Protocol

### 1. Freeze The Roster

For every candidate, record the provider route, external score and source row,
benchmark reasoning variant, run reasoning settings, release date, context
limit, and pricing. Candidate settings should match the benchmarked variant
where the provider exposes an equivalent control. Unmatched settings are marked
as a separate condition rather than silently treated as gold-equivalent.
Routes whose reasoning mode is not exposed in the benchmark label receive a
large completion ceiling because some providers count hidden reasoning against
that limit. The visible probe still imposes the same 500--600 word answer cap.
Selective recovery may supply a per-route parameter override file. Its path and
the exact request parameters are stored with the run; any answer produced under
a benchmark-mismatched effort setting is reported as a runtime sensitivity, not
silently pooled with the primary condition.

The generated config records an isolated route failure as explicit missing
evidence and continues the batch. Strict failure remains available as a config
option. Missing answers can later be retried by round without regenerating
successful answers.

Use stable anonymous IDs `P01` through `P50`. Randomize answer presentation
order independently for each probe while retaining those IDs. Use the same
roster, participant mapping, and seeded presentation permutations for both
judges. Neither judge sees model names, routes, scores, or release dates.

### 2. Route Preflight

Before paid evaluation, send one short neutral request to every route. Record
visible-output success, latency, reasoning compatibility, finish reason, and
reported cost. Route failures are compatibility exclusions, not intelligence
observations. Freeze the replacement policy before the main run.

### 3. Opening Battery

Each judge independently authors four complementary probes before seeing any
candidate answer. Each probe is sent unchanged to all 50 candidates in a fresh
context. Candidate responses have a common visible-answer limit.

For each probe, the judge sees all 50 answers together. It produces:

- a complete within-probe ordering, allowing explicit ties;
- a concise evidence summary for every candidate;
- concrete strengths and errors;
- uncertain adjacent pairs or clusters;
- a validity judgment for the probe itself.

The comparison is global because 50 answers fit in context. Every exact probe
and answer is also saved in `probe_answer_archive.json`, including presentation
order and usage metadata. This permits a later grouped-panel replay with
overlapping anchors without calling candidates again. Panels are a fallback for
observed attention failures, not the primary protocol.

### 4. Cumulative Ranking

After the four probe comparisons, the judge receives the compact per-candidate
evidence across all probes and produces a full ranking, confidence, unresolved
comparisons, and candidate clusters that need more evidence. It does not assign
isolated ability scores unless it created a rubric that naturally uses them.

### 5. Adaptive Tie-Breaking

Use two optional adaptive rounds, for a schedule of `[4, 1, 1]`. In each round,
the judge may select at most ten candidates from an uncertain cluster and write
one common follow-up probe for that subset. The new answers are compared
together and merged into the cumulative ranking.

The ranking after five probes is the preregistered primary endpoint, consistent
with the close-roster pilots. The four-probe opening is a baseline checkpoint,
and the six-probe ranking measures the marginal value of a second adaptive
round. This avoids selecting the most favorable stopping point after seeing
results.

### Five-Opening-Probe Replication

The replication keeps the roster, anonymous IDs, answer-order seed, candidate
settings, and two independent judges fixed, but uses `[5, 1, 1]`. It therefore
provides fresh rankings after five, six, and seven cumulative probes. This is a
probe-budget replication, not a replacement for the original `[4, 1, 1]`
condition; old-new rank agreement and changes in external pairwise accuracy are
reported directly.

## Shared-Evidence Cross-Over

After both independent ladders finish, run a crossed control without purchasing
new candidate answers:

| Judge | Sol-authored evidence | Fable-authored evidence |
| --- | --- | --- |
| Sol | Primary Sol ladder | Crossed judgment |
| Fable | Crossed judgment | Primary Fable ladder |

Each crossed judgment starts from a fresh judge context and sees exactly the
same anonymous answers as the original judge. This separates two capabilities:

- **Probe design:** whether one judge creates more diagnostic tests.
- **Evidence interpretation:** whether one judge ranks the same answers better.

Crossed configs set `replay_source_targets: true`. This preserves the original
adaptive comparison sets even when the crossed judge's interim ranking would
have selected different candidates. Explicitly unavailable source answers are
also replayed as missing evidence, so no candidate is called again.

## Scale And Cost Shape

With 50 candidates and two adaptive rounds targeting at most ten candidates,
the original four-opening-probe condition requires:

- Opening candidate calls: `2 judges x 4 probes x 50 = 400`.
- Maximum adaptive candidate calls: `2 judges x 2 probes x 10 = 40`.
- Primary candidate-call budget: 440, plus bounded retries.
- Crossed judgments reuse saved probes and answers, adding judge calls but no
  candidate calls.

The five-opening-probe replication uses 500 opening candidate calls across the
two judges and at most 40 adaptive candidate calls. Successful answers are
journaled immediately, so a repair retries only explicitly unavailable cells
and then recomputes comparisons and rankings from the completed evidence. The
repair preserves the realized adaptive probes and target sets rather than
inventing a counterfactual branch after seeing repaired opening answers. Thus
the five-probe opening endpoint is fully repaired; later checkpoints remain
auditable realized-path sensitivity analyses.

The main methodological risk is whether a judge can reliably compare 50 long
answers without position effects or attention loss. The primary run therefore
uses seeded shuffling and retains enough evidence to replay alternative orders
or overlapping panels. The main engineering risk is the slow and unreliable
tail of 200 concurrent provider calls. Completed responses are journaled before
the batch finishes, canonical transcript order is deterministic, and a replay
calls only candidates whose valid answer is still missing.

Candidate and judge calls use separate provider profiles even when both route
through OpenRouter. The original pilot used a five-minute candidate deadline;
the replication uses ten minutes after the slow-route audit. Global judge
comparisons have a fifteen-minute deadline because they read all 50 answers and
produce 50 evidence summaries. This changes transport limits, not model access
or evidence.

## Primary Outcomes

- Tie-aware Kendall tau and Spearman correlation with the external ordering.
- Pairwise accuracy as a function of external score gap.
- Top-k recall and rank error for the strongest candidates.
- Agreement between judges on final order and uncertain pairs.
- Accuracy, churn, and confidence after four, five, and six probes.
- Probe validity, question types, strategies, and adaptive dynamics by judge.
- Total and per-model calls, latency, tokens, and provider-reported spend.

The primary plots are predicted versus external rank, the discrimination curve
by score gap, judge-to-judge rank differences, and accuracy or stability versus
probe count and cost.

## Expansion Beyond Fifty

The runner can route more than 50 candidates, and the underlying APIs have
enough context for substantially larger comparisons. A 100-model study should
not simply double the first run. It should first validate either global
comparison at that scale or a grouped strategy with randomized overlapping
anchors and an auditable merge. Candidates without direct reported scores can
be included, but they belong in a secondary analysis because their external
ordering is less defensible.

## Pilot Status

Both independent 50-model runs and the shared-evidence cross-over are complete.
Global comparison fit in context and produced complete structured judgments,
so panels remain a diagnostic fallback rather than part of the primary method.
Sol's direct-score pairwise accuracy rose from 0.811 after four probes to 0.839
after six; Fable's moved from 0.804 to 0.798. In the crossed control, Fable
judging Sol evidence reached 0.867 after four probes, while Sol judging Fable
evidence reached 0.809. The primary-run analysis is in
[`pilot_analysis_catalog_ladder50_20260721.md`](pilot_analysis_catalog_ladder50_20260721.md),
and the crossed analysis is in
[`pilot_analysis_catalog_ladder50_crossed_20260721.md`](pilot_analysis_catalog_ladder50_crossed_20260721.md).

A fresh shuffled-order replay of the best four-probe condition agreed with the
original at Kendall tau 0.814, retained the same top three, and moved direct-
score pairwise accuracy from 0.867 to 0.837. This supports global comparison
with explicit order uncertainty rather than panel merging. See
[`pilot_analysis_catalog_order_20260725.md`](pilot_analysis_catalog_order_20260725.md).

The archived opening answers were also scored on a fixed `0–4` correctness
scale by Sol and Fable. Averaged scores ordered 84.8% of direct-score model
pairs correctly and exposed meaningful differences in probe difficulty and
judge rubrics. See
[`pilot_analysis_probe_scoring_20260725.md`](pilot_analysis_probe_scoring_20260725.md).
