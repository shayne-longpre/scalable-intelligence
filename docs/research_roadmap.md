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
| 5. Selective adaptivity | Three stress pilots complete | Four opening probes, direct per-probe comparison, cumulative evidence dossiers, bounded common follow-ups, and adaptive-decision traces | Can extra evidence resolve the difficult part of a ranking efficiently? |
| 5b. Probe-budget ablation | Initial pilot complete | Fresh evidence cards and rankings using only the first 2, 4, or 6 probes from one shared run | How much does additional probe evidence improve ranking accuracy? |
| 6. Oversight frontier | Configured and mock-smoked; next paid experiment | Judges above, near, and below small candidate panels, with candidates on both sides of the judge's capability | Can a model recognize a system more capable than itself? |
| 7. Judge-strategy scaling | Fixed-scale catalog scoring complete; judge-capability comparison pending | Probe-type, difficulty, adaptation, and answer-quality comparisons across judge capability | Do stronger judges ask systematically different or more diagnostic questions? |
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

1. The judge authors four complementary probes before seeing answers.
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

## Immediate Next Steps

1. Review the eight-item human audit sample, make only evidenced taxonomy or
   summary changes, and freeze the measurement version for confirmatory work.
2. Run the first permutation for the Sol, GPT-5.4 Mini, and Claude Haiku 4.5
   oversight panels. Inspect probe validity, routing, and above-judge
   comparisons before purchasing the second permutations.
3. Run the second oversight permutations without prompt changes, then build the
   judge-capability by candidate-gap result.
4. Rerun the Sol and Fable catalog openings with five probes as a short
   methodology check. Treat evidence/adaptivity scaling as supporting analysis,
   not a separate large program.
5. Compare probe repertoire, difficulty, and adaptive behavior across judge
   capability. This is the next major analysis after the oversight frontier.
6. Keep structured-versus-free councils as later exploratory work. Do not scale
   beyond 50 candidates unless the substantive research question requires it.
