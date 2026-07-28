# Research Roadmap

This file tracks the next empirical milestones. The README defines the stable
methodology; this roadmap records implementation and study status as the design
evolves.

## Current Objective

Test whether anonymous AI evaluators can infer relative general intelligence
from model-generated interactions, while preserving the evaluators' freedom to
invent their own criteria and probes.

## Milestones

| Step | Status | Deliverable | Main research question |
| --- | --- | --- | --- |
| 1. Measurement validity | Complete | Role-scoped taxonomy counts, source-turn provenance, and ranking metrics validated against hand-labeled transcripts | Are reported patterns real properties of the interaction rather than extraction artifacts? |
| 2. Conversation dynamics | Complete | Separate preplanned battery progression from evidence-conditioned follow-up and topical change | Do evaluators deepen, broaden, or adapt as evidence accumulates? |
| 3. Baseline freeze | Complete | Versioned adaptive-wave prompts, taxonomy, config, analysis schema, and three accepted replications | Can later experiments be compared without silent protocol drift? |
| 4. Independent-judge ladder | Complete, including order replay and answer scoring | Sol and Fable independently rank the same 50 anonymous candidates, followed by a shared-evidence cross-over | Can capable models recover a broad external capability ordering? |
| 5. Selective adaptivity | Five-probe 50-model replication complete | Opening batteries, direct per-probe comparison, cumulative evidence dossiers, bounded common follow-ups, and adaptive-decision traces | Can extra evidence resolve the difficult part of a ranking efficiently? |
| 5b. Probe-budget ablation | Initial pilot complete | Fresh evidence cards and rankings using only the first 2, 4, or 6 probes from one shared run | How much does additional probe evidence improve ranking accuracy? |
| 6. Oversight frontier | Four waves, 30 panels, ten judges, and matched relative-gap extension complete | Judges above, near, and below small candidate panels, with candidates on both sides of the judge's capability | Can a model recognize a system more capable than itself? |
| 7. Judge-strategy scaling | 147-probe author audit and 25-probe ceiling extension complete | Probe-type, difficulty, author solvability, adaptation, and same-evidence comparisons across judge capability | Do stronger judges ask systematically different or more diagnostic questions, including questions they cannot solve themselves? |
| 8. Free vs. structured | Later exploratory study | Paired runs with the same roster, budgets, and external prior | What does round-robin structure improve or suppress? |
| 9. Publication figures | Catalog results, fixed-scale heatmap, references, and audit sample generated | Auditable plots, tables, taxonomy, model provenance, and compact human-review samples | Which findings are robust, interpretable, and worth communicating? |
| 10. Scale beyond 50 | Paused | Extend the archived-answer and panel machinery only if a research question requires it | Can the method rank a much larger catalog tractably? |

## First Ladder Design

This section records the completed fixed-battery control. New adaptive studies
use the protocol below; the historical ladder remains reproducible.

- **Candidates:** 12 models spanning frontier systems through models released in
  2023, selected to cover broad capability gaps while retaining enough close
  neighbors to test discrimination.
- **Judges:** GPT-5.6 Sol at `xhigh` reasoning and one mid-2025 open-weight model.
  They run independently and never see one another's probes or judgments.
- **Baseline evidence:** Each judge designs six probes. Every probe is sent
  unchanged to every candidate in a fresh context.
- **Judgment:** Each judge creates a compact evidence card per candidate, then
  ranks all 12 candidates together.
- **Probe ablation:** Candidate calls are shared. Separate judgment branches see
  only probe prefixes 1-2, 1-4, or 1-6, preventing later evidence from leaking
  into lower-budget rankings.
- **Adaptive extension:** The full-evidence ranking may select a small uncertain
  set. A later common probe is sent only to that set, with the judge receiving
  the prior ranking and relevant evidence cards.

## Primary Outputs

- Kendall tau, Spearman rank correlation, pairwise accuracy, and top-k accuracy
  against the external prior.
- Inter-judge agreement and rank differences.
- Accuracy and confidence at 2, 4, and 6 probes.
- Pairwise accuracy by prior capability gap.
- Question-type, evaluation-strategy, and conversation-dynamic distributions.
- Rank and confidence trajectories with decisive source evidence. Legacy or
  judge-invented scores are retained when present but are not required.
- Total and per-model provider-reported spend.

## Core Experiment Program

The project should concentrate on four experiments. The first three test
scalable intelligence evaluation directly; the fourth is an exploratory social
demonstration.

### 1. Reference Judges Rank A Catalog

Two or more capable judges independently evaluate a broad anonymous candidate
catalog. They begin with a common portfolio of probes, preserve per-probe
comparisons, and use bounded adaptive tie-breakers for close candidates.

Primary plots: inferred rank or score versus the external prior; pairwise
accuracy by prior capability gap; judge-to-judge agreement; accuracy and cost by
candidate count.

The initial design uses independent GPT-5.6 Sol and Claude Fable 5 judges over a
50-model roster, followed by a shared-evidence cross-over. See
`docs/catalog_ladder_design.md`. Every exact answer is archived so the same
evidence can later be compared in smaller overlapping panels without another
candidate call.

### 2. The Oversight Frontier

Judges at several capability levels rank small panels containing candidates
below, near, and above the judge's own external score. Repeated anonymous
permutations estimate when a judge can recognize a system more capable than
itself.

Primary plots: a judge-capability by candidate-gap accuracy heatmap; probability
of correctly identifying a superior candidate; Kendall or pairwise accuracy by
judge score; confidence versus correctness above and below the judge's level.

### 3. Evidence And Adaptivity Scaling

Hold the roster and candidate answers fixed while varying how much evidence a
judge receives: probe prefixes, common follow-ups, and adaptive follow-ups. Give
different judges the same evidence in a crossed control to separate test design
from evidence interpretation.

Primary plots: accuracy, rank churn, uncertainty, and spend versus cumulative
probes; marginal accuracy gain per probe or dollar; resolution rate for targeted
uncertain pairs; broadening and deepening transitions over rounds.

### 4. Structured Versus Free Councils

Use only highly capable models and match roster, token budget, external prior,
and final-ranking requirements across structured probing and free discussion.
This experiment studies emergent evaluation culture and serves as a compelling
demonstration, but it is not the main scalable-ranking benchmark.

Primary plots: mode differences in accuracy and agreement; question-type and
strategy distributions; interaction or probe network; rank and confidence
trajectories annotated with decisive exchanges.

Before treating external scores as gold, candidate inference settings must
match the benchmarked variants or the analysis must use explicit uncertainty
bands and possible ties.

## Completed Ladder Pilot

The complete pilot is
`runs/20260719T034813Z_independent_judges_ladder12_6probe`; its report card is
`runs/report_cards/ladder12_complete_20260719_v2/report_card.html`. It contains 12
judge-authored probes, all 144 routed Q/A pairs, 72 evidence cards, and six
isolated 2/4/6-probe rankings. The resumable config now points to this complete
transcript, so protocol and analysis changes can be tested without purchasing
the model outputs again.

The stronger judge reached pairwise accuracy of 0.83, 0.92, and 0.91 after 2,
4, and 6 probes. The medium judge reached 0.62, 0.58, and 0.70. Both selected
the prior's top model after six probes, but their full-ranking Kendall agreement
was 0.52. The stronger judge used the full score range to discriminate models;
the medium judge assigned the same score to eight of twelve candidates, an
important failure mode for later judge selection and calibration analysis.

All expected artifacts completed. One candidate answer needed a bounded output
retry, and two medium-judge rankings needed same-model JSON repair. Revalidation
under the current monitor produces zero findings. The cumulative replay lineage
records at least $13.96 across eight attempts; the final resume added $1.09.
The full interpretation and classification audit are in
`docs/pilot_analysis_ladder12_20260719.md`.

DeepSeek V4, MiniMax M3, and Qwen routes were excluded from this baseline after
provider/reasoning behavior repeatedly produced no usable visible answer or
exceeded practical latency bounds. Those are protocol-compatibility exclusions,
not intelligence judgments.

## Design Guardrails

- Candidate identities and external rankings never enter judge context.
- Baseline probes are designed before candidate answers are observed.
- The same probe is used for every candidate in a comparison set.
- Lower-probe judgment branches receive no evidence or summaries from later
  probes.
- Adaptive labels require evidence that the later probe was generated after
  relevant prior answers or judgments existed; topical similarity alone is not
  evidence of adaptation.
- Taxonomy labels are post-hoc measurements and never influence live discourse.
- Every aggregate label and ranking remains traceable to transcript turn IDs.

## Adaptive Judge Baseline

The current baseline is one explainable loop:

1. The judge authors several complementary probes before seeing answers. The
   current catalog baseline uses five.
2. Each probe goes unchanged to every candidate in the selected comparison set.
3. The judge compares all answers to one probe and preserves concrete strengths,
   errors, implementation details, uncertainty, and a within-probe ordering.
4. The judge merges all current probe comparisons with its previous cumulative
   dossiers and updates the full ranking.
5. The judge selects a small uncertain subset and writes the next common probe.
6. The cumulative judgment after every round is analyzed as a possible early
   stopping point.

The initial stress-test schedule is `[4, 1, 1, 1, 1]`. This yields one common
four-probe opening portfolio and four opportunities for evidence-dependent
follow-up. The main stopping analysis compares rank accuracy, churn,
uncertainty, probe validity, and taxonomy movement after cumulative probe counts
4, 5, 6, 7, and 8.

The first stress pilot is complete. Its top candidate was stable after the four
opening probes; the first two follow-ups did not change the ranking, the third
corrected one middle-order inversion against the external prior, and the fourth
did not change it again. This is evidence that checkpointing works, not enough
evidence to fix a universal stopping rule. See
`docs/pilot_analysis_adaptive_waves_20260720.md`; the report card is
`runs/report_cards/adaptive_waves_pilot_20260720_final_v2/report_card.html`.

Two additional hardening pilots contrasted a medium judge over a broad roster
with Sol over a close frontier roster. They exposed probe saturation,
repetitive follow-ups, route-specific reasoning failures, and replay
assumptions; all are now represented in prompts, metrics, runtime policy, or
regression tests. The comparative report is
`runs/report_cards/adaptive_pilots_20260720_v4/report_card.html`, and the detailed
interpretation is `docs/pilot_analysis_adaptive_judges_20260720.md`.

## Replicated Judge-Quality Pilot

Three accepted runs independently compared GPT-5.6 Sol and GPT-5.4 Mini as
judges of the same four close frontier candidates. Candidate IDs were permuted
between runs; prompts, the `[4, 1, 1, 1]` schedule, and routing rules were
frozen. All 18 adaptive selections matched the judges' requested candidates and
covered their stated uncertain pair or pairs.

Sol produced informative comparisons for 18 of 21 probes, versus 11 of 21 for
Mini, and narrowed follow-ups to 2.22 candidates on average versus 3.00. That
clear test-design advantage did not translate into better agreement with the
external prior: aggregate pairwise accuracy was non-monotonic for both judges,
and Mini's best checkpoint was five probes. The result is not evidence that
Mini is the better judge because the roster is small and adjacent frontier
prior ranks are uncertain. It is evidence that probe quality, confidence, and
prior agreement must be reported separately.

The accepted runs cost $17.258874 across 272 model calls. Excluded hardening
attempts cost $7.287047 across 121 calls and remain recorded as operational
reliability evidence. Full analysis is in
`docs/pilot_analysis_adaptive_judge_quality_20260721.md`; the report card is
`runs/report_cards/adaptive_judge_quality_close_p4_v3/report_card.html`.

## 50-Model Catalog Pilot

The independent Sol and Fable catalog runs are complete. Both judges compared
all 50 answers globally for each opening probe, then used two common adaptive
probes over at most ten candidates. The 47-model direct-score primary analysis
reached final pairwise accuracy of 0.839 for Sol and 0.798 for Fable. Their
final rankings agreed at Kendall tau 0.605.

Accuracy depended strongly on capability gap. Both judges exceeded 0.91 on
pairs separated by more than ten external-score points, while pairs separated
by less than two points remained near chance. Sol improved across the two
adaptive checkpoints; Fable did not.

The shared-evidence cross-over is also complete. Fable judging Sol-authored
evidence produced the strongest externally aligned ranking, reaching 0.867
pairwise accuracy after the four-probe opening. Sol improved Fable-authored
evidence from 0.804 to 0.809 at that checkpoint. Same-evidence judge agreement
was substantially higher than same-judge agreement across batteries, indicating
that probe choice materially shaped the resulting order. Full results are in
`docs/pilot_analysis_catalog_ladder50_crossed_20260721.md`; the crossed report is
`runs/report_cards/catalog_ladder50_crossed_20260721_v1/report_card.html`.

All exact answers were archived, so alternative answer orders and overlapping
panels require judge calls but no new candidate calls. Full results and caveats
are in `docs/pilot_analysis_catalog_ladder50_20260721.md`; the combined report
card is
`runs/report_cards/catalog_ladder50_sol_fable_20260721_v5/report_card.html`.

The seeded order replay retained the same top three and agreed with the original
four-probe ranking at Kendall tau 0.814. External pairwise accuracy moved from
0.867 to 0.837, so order is a reported uncertainty but does not currently
justify panel merging as the primary method. See
`docs/pilot_analysis_catalog_order_20260725.md`.

Sol and Fable also assigned fixed `0–4` answer-quality scores to the eight
opening probes. Mean score across probes correlated 0.874 with the external
index and ordered 84.8% of direct-score model pairs correctly. One probe has
only Sol scores after repeated zero-content Fable responses. See
`docs/pilot_analysis_probe_scoring_20260725.md`.

## Oversight Frontier Pilot

Six independent judges spanning external scores from 15.9 to 58.9 each ranked
seven anonymous candidates distributed around the judge's own capability. Every
judge authored five common opening probes and one bounded adaptive follow-up.

Across 126 candidate pairs, final accuracy was 70.6%. Capability distance was
far more predictive than judge identity: accuracy was 40.9% for pairs separated
by less than two external-score points and 95.5% beyond ten points. Judges
placed 13 of 16 stronger candidates above their anonymous selves, but symmetric
self-relative accuracy was only 23 of 36 because several judges also placed
weaker candidates above themselves.

The adaptive probe improved three judges, left one unchanged, and worsened two;
aggregate accuracy moved from 71.4% to 70.6%. This does not support assuming
that a single follow-up is beneficial. The result also exposed a measurement
risk: adversarial false-premise questions were diagnostic when the judge
rewarded correction and misleading when it punished correction.

The complete interpretation is in
`docs/pilot_analysis_oversight_frontier_20260725.md`; the publication report is
`docs/site/oversight.html`.

## Oversight Replication

The frozen protocol has now run on two independent seven-candidate panels for
each of six judges. Across 252 candidate pairs, final pooled accuracy is 74.6%.
Capability separation is the most stable result: pooled accuracy rises from
57.8% below a two-point external-score gap to 96.6% beyond ten points. Judges
correctly ordered 49 of 72 candidate-versus-self comparisons. They placed 15 of
22 candidates at least two points above them in the external index above their
anonymous selves.

The new panel improved aggregate accuracy from 70.6% to 78.6%, but
condition-level movement ranged from -9.5 to +28.6 percentage points. This is
panel sensitivity, not evidence for a monotonic judge-capability frontier.
Across both panels, one adaptive probe improved six rankings, left four
unchanged, and worsened two; pooled accuracy moved from 72.6% to 74.6%.

Three judges also reranked exact archived evidence under new answer orders.
All 18 comparison orders changed and no candidate was called. Sol remained
stable at Kendall tau 0.90, while MiniMax and Mistral each reached 0.52. All
three retained the same top candidate. Full methods, results, caveats, and
links are in
`docs/pilot_analysis_oversight_frontier_replication_20260726.md`.

## Ten-Judge Oversight Extension

The third wave expands the study to ten judges and uses nine-candidate panels
that contain five candidates above the judge wherever the catalog permits.
The repaired pooled analysis orders 469 of 612 candidate pairs correctly
(76.6%) and places 50 of 78 externally stronger candidates above anonymous
self (64.1%). Symmetric candidate-versus-self accuracy is 109 of 152 (71.7%).
Recognition is highest for margins above ten external-score points, but
margin-bin intervals overlap and judge behavior remains heterogeneous. There is
no supported universal oversight threshold yet.

Eighteen of the third wave's 19 unavailable answers were recovered without
changing model settings. One Fable answer remains unavailable because the
provider repeatedly filtered the unchanged prompt. Raw and repaired outcomes
are reported separately. The repaired pooled lineage cost $78.68 across 1,518
model calls. Results and protocol caveats are in
`docs/pilot_analysis_oversight_frontier_above_heavy_20260726.md`.

## Five-Probe Catalog Replication

The fresh Sol and Fable runs use five opening probes followed by two selective
adaptive probes. Mean final pairwise accuracy remained stable, moving from
81.9% to 82.3%, but the judges moved in opposite directions. Old-new rank
agreement was 0.69 for Sol and 0.65 for Fable. Both fresh runs remained above
92% on model pairs separated by at least ten external-score points.

Neither judge improved materially after the five-probe checkpoint. Sol moved
from 80.9% to 81.1%; Fable moved from 84.6% to 83.4%. This supports treating
the opening battery as the clean primary endpoint and adaptive rounds as
explicitly evaluated extensions. See
`docs/pilot_analysis_catalog_ladder50_opening5_20260727.md`.

## Probe-Evolution Study

The fixed-protocol comparison now covers 108 probes across 18 runs from ten
judges. Stronger judges did not cover more topical families and were only
slightly more accurate on pairs they explicitly separated. They did use many
more ties and articulated somewhat broader adaptive intentions. Across the full
30-panel study, one adaptive probe improved ten rankings, left eleven
unchanged, and worsened nine.

The separate 50-model deep dive adds 14 Sol and Fable probes. Thirteen were
marked informative, but the adaptive probes did not improve either global
ranking materially.

The exact-evidence cross-over is also complete. Sol and Llama re-evaluated
unchanged answers to four batteries. Their aggregate five-probe accuracy
differed by only 1.4 points, while which evaluator led changed with the battery.
Battery means ranged from 56.9% to 73.6%. One battery per author is not enough
to infer stable probe-writing ability. See
`docs/pilot_analysis_probe_evolution_20260727.md` and
`docs/pilot_analysis_probe_design_cross_20260727.md`.

## Matched Oversight Extension

Two new matched panels for Grok, DeepSeek, Kimi, and Llama bring the pooled
oversight study to 30 panels. After five probes, judges placed 77 of 118
externally stronger candidates above anonymous self (65.3%), nearly identical
to 51 of 78 (65.4%) before the extension. Recognition rose to 13 of 14 for
leads above ten external-score points, while all smaller margin bins remained
between 57% and 65%.

Upper-third judges performed better descriptively on the observed
candidate-versus-self comparisons, but panel variation remains large. Kimi's
opening pair accuracy moved from 80.6% to 33.3% across two matched panels. The
next frontier study therefore needs repeated shared or closely matched evidence,
not just more one-off judges. See
`docs/pilot_analysis_oversight_matched_20260727.md`.

## Publication Synthesis

The first four primary questions now have versioned evidence:

1. Two independent five-probe frontier judges recover 82.7% of the comparable
   50-model pairs in the reference ordering. Accuracy reaches 92.3% for gaps of
   ten points or more, while the exact top five remains unstable.
2. Across 30 oversight panels, judges place 65.3% of stronger anonymous
   candidates above their own hidden responses. More capable judges do better
   descriptively after margin standardization, but the design does not isolate
   a causal judge-intelligence effect.
3. Across 146 fixed-reference probes, the taxonomy captures a broad repertoire
   but does not predict held-out probe effectiveness reliably. One adaptive
   follow-up has mixed effects.
4. In four matched mechanism panels, verifier-oriented guidance does not improve
   above-self recognition, while three independent evaluators improve ordinary
   pairwise accuracy from 64.6% to 70.1%. The two interventions show no positive
   interaction.

The claim decisions and unresolved gaps are in
`docs/claim_evidence_matrix.md`. Machine-readable results are in
`data/research_question_synthesis.json` and
`data/probe_effectiveness_results.json`. Mechanism results are in
`data/verifier_council_matched_v1_results.json`.

## Immediate Next Steps

1. Freeze the current article and four-question evidence synthesis; only change
   headline claims through the claim-to-evidence gate.
2. Audit the verifier batteries for concrete checkability, hidden
   contradictions, and the evidence that changed rankings. This uses archived
   transcripts and requires no model calls.
3. Decide whether a follow-up mechanism study targets the observed failure:
   better probe validation before routing, or diverse councils on held-out
   panels. Preregister one intervention rather than combining both.
4. Keep free discussion as a later qualitative study of social dynamics, not
   the primary scalable-ranking method.
