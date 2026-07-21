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

The current catalog snapshot contains 339 OpenRouter routes. Fifty-one have a
direct, non-estimated Artificial Analysis Intelligence Index score and at least
32K context. Excluding Fable from the candidate roster, because it is a judge,
leaves exactly 50 scored routes.

Fifty is the recommended first catalog ladder:

- It spans external scores from roughly 6.8 to 55.7.
- Four common probes require 200 candidate calls per judge.
- All 50 answers to one probe should fit comfortably in either judge's context,
  allowing direct global comparison instead of isolated scoring or forced
  tournament brackets.
- It is large enough to estimate accuracy by capability gap while remaining
  auditable candidate by candidate.

The roster contains two likely preview/production alias pairs with identical
scores. They should either be treated as external ties or replaced by clearly
labeled estimated-score entries. The catalog must be refreshed and every route
must pass a preflight before the roster is frozen.

## Protocol

### 1. Freeze The Roster

For every candidate, record the provider route, external score and source row,
benchmark reasoning variant, run reasoning settings, release date, context
limit, and pricing. Candidate settings should match the benchmarked variant
where the provider exposes an equivalent control. Unmatched settings are marked
as a separate condition rather than silently treated as gold-equivalent.

Use stable anonymous IDs `P01` through `P50`. Randomize answer presentation
order independently for each probe while retaining those IDs. Use the same
roster and seeded presentation permutations for both judges.

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

The comparison is global because 50 answers fit in context. Grouped comparison
with overlapping anchors remains a fallback for larger rosters or observed
attention failures, not the default.

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

## Scale And Cost Shape

With 50 candidates, four opening probes, and two adaptive rounds targeting at
most ten candidates:

- Opening candidate calls: `2 judges x 4 probes x 50 = 400`.
- Maximum adaptive candidate calls: `2 judges x 2 probes x 10 = 40`.
- Primary candidate-call budget: 440, plus bounded retries.
- Crossed judgments reuse saved probes and answers, adding judge calls but no
  candidate calls.

The main technical risk is not context capacity. It is whether a judge can
reliably compare 50 long answers without position effects or attention loss.
The first pilot should therefore audit ranking consistency under shuffled answer
order before increasing the roster beyond 50.

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
