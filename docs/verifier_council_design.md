# Verifier-Oriented Probes And Independent Judge Councils

This pilot tests two narrow mechanisms for making above-judge evaluation more
reliable. It extends the independent-judge ranking study; it does not add a
third machine-society mode.

## Research Questions

1. **Verifier-oriented probes:** Does asking a judge to design questions whose
   answers it can check reliably help it recognize candidates above its own
   capability?
2. **Independent councils:** Does deterministic aggregation of independent
   judges improve on the same strongest judge acting alone?
3. **Combined intervention:** Are verifier-oriented probes and councils
   complementary, redundant, or counterproductive?

## Matched Panels

The pilot reuses four nine-candidate panels previously evaluated by Grok 4.5,
DeepSeek V4 Pro, Kimi K2 Thinking, and Llama 4 Maverick. These judges span a
large capability range, and every panel contains the judge's anonymous
response plus candidates below, near, and above it.

For each panel, the verifier condition preserves:

- candidate roster and anonymous participant IDs;
- five-probe opening budget;
- fresh candidate context for each probe;
- candidate answer limits and prompts;
- answer-presentation seed;
- comparison and cumulative-ranking prompts.

Only the private probe-design guidance changes. It asks the judge to seek
concrete, inspectable evidence that it can evaluate even when producing the
best answer may exceed its own capability. It does not prescribe domains,
question formats, rubrics, or a definition of intelligence.

## Independent Council

The council contains three evaluators:

- GPT-5.6 Sol;
- Llama 4 Maverick;
- Qwen3.7 Max.

Members receive the same archived probes and candidate answers independently.
They do not see one another's assessments and do not deliberate. Answer order
is independently seeded by member. This isolates ensemble aggregation from
social influence and requires no new candidate calls.

Pairwise majority is the primary council decision. Mean ordinal rank produces
a complete deterministic display order, with pairwise wins and participant ID
used only as tie-breakers. No model is asked to synthesize the council.

## Endpoints

The analysis freezes three comparisons:

1. **Verifier effect for the author:** change in the proportion of externally
   stronger candidates ranked above the judge's anonymous response. Overall
   pairwise accuracy is secondary.
2. **Council effect:** change in candidate-pair accuracy from the Sol member
   alone to three-member majority on identical evidence.
3. **Interaction:** whether the council gain differs between ordinary and
   verifier-oriented batteries.

Results are reported panel by panel as well as pooled. The Artificial Analysis
Intelligence Index remains a noisy external reference rather than literal
ground truth. With four panels, this is a mechanism pilot, not a precise effect
estimate.

## Result

Verifier-oriented guidance did not improve the primary endpoint. Probe authors
placed 60% of externally stronger candidates above their anonymous selves with
ordinary probes and 55% with verifier-oriented probes. Their overall pairwise
accuracy rose slightly, from 63.9% to 66.0%.

Independent aggregation was more useful. On ordinary batteries, Sol alone
ordered 64.6% of candidate pairs correctly and the three-member council ordered
70.1%. On verifier batteries, the corresponding figures were 72.2% and 75.0%.
The interventions did not reinforce one another: the council's average gain
was 2.8 points smaller in the verifier condition.

See
[`pilot_analysis_verifier_council_20260727.md`](pilot_analysis_verifier_council_20260727.md)
for panel-level results, probe behavior, failures, costs, and limitations.
