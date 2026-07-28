# Claim-to-Evidence Matrix

This document is the publication gate. A new analysis or model call should
support one of the claims below, change its interpretation, or remain deferred.

## RQ1: Can AI Judges Recover A Capability Ranking?

| Candidate claim | Best evidence | Current result | Strength | Remaining gap | Decision |
| --- | --- | --- | --- | --- | --- |
| Strong judges recover the broad capability ladder. | Independent Sol and Fable five-probe rankings of the same 50 anonymous models. | Mean pairwise agreement with the external index is 82.7%. | Strong descriptive evidence with a fresh replication. | External scores are noisy and three weak anchors are estimated. | Publish with the external index described as a reference, not ground truth. |
| Large differences are easier than close ones. | Pairwise accuracy stratified by external-score gap. | Accuracy rises from 54.3% below two points to 92.3% above ten points. | Strong and consistent across both judges. | The index's local scale may not be uniform. | Publish as the most stable catalog result. |
| The resulting fine-grained league table is stable. | Independent judges and old-versus-new battery comparisons. | Fresh judges have Kendall 0.74 but share only one top-five model; mean old-new Kendall is 0.67. | Evidence against a precise stable league table. | More batteries could estimate variance more precisely. | Publish the limitation; do not run another broad catalog merely to reduce variance. |

Primary artifacts:
`data/research_question_synthesis.json`,
`data/catalog_ladder50_opening5_stability.json`, and
`docs/figures/catalog-ladder50-opening5/`.

## RQ2: Can A Judge Recognize Models More Capable Than Itself?

| Candidate claim | Best evidence | Current result | Strength | Remaining gap | Decision |
| --- | --- | --- | --- | --- | --- |
| Models often recognize a stronger anonymous model. | Thirty panels, ten judges, and 118 above-judge candidate comparisons. | 77 of 118 stronger candidates, or 65.3%, were placed above anonymous self. | Moderate replicated descriptive evidence. | Panels and probe batteries vary by judge. | Publish "often," not "reliably." |
| Judge intelligence matters. | Margin-stratified results by judge capability third. | Standardized sub-ten-point recognition is 48.0%, 60.1%, and 85.8% from lower to upper third. | Suggestive; panel-bootstrap intervals remain wide. | Judge identity, provider, panel, and battery remain entangled. | Publish as suggestive evidence, not a causal model effect. |
| A larger candidate lead makes recognition easier. | Recognition by the candidate's external-score margin over the judge. | Leads of ten or more points are recognized 92.9% of the time; smaller bins are 57-65% and non-monotonic. | Strong only for the largest observed lead. | Sparse observations prevent a sharp threshold estimate. | Publish the large-margin result and explicitly reject a precise frontier threshold. |

Primary artifacts:
`data/research_question_synthesis.json`,
`data/oversight_frontier_synthesis_matched_results.json`, and
`docs/figures/research-synthesis/oversight-frontier.svg`.

## RQ3: What Tests Do Judges Invent, And Which Work?

| Candidate claim | Best evidence | Current result | Strength | Remaining gap | Decision |
| --- | --- | --- | --- | --- | --- |
| Judges use a broad repertoire rather than one benchmark family. | Taxonomy over 147 probes from 11 authors. | Math dominates, alongside science, planning, logic, abstraction, spatial reasoning, coding, philosophy, and practical judgment. | Strong descriptive evidence. | Automated labels require continued human audit. | Publish with examples and link the complete taxonomy. |
| Stronger judges write more diagnostic probes. | Fixed-evaluator answer scores and per-probe external-order accuracy. | Author intelligence has only rho 0.10 with probe accuracy. | Evidence does not support the simple claim. | Panels differ and one reference evaluator scores all answers. | Publish the null result cautiously. |
| Particular taxonomy labels identify better probes. | Leave-one-author-out label prediction. | Question types and the full label set order held-out probes at approximately chance. | Useful negative result. | More repeated batteries per author could reveal stable interactions. | Publish that labels describe behavior but are not yet a probe-selection rule. |
| Judges can write useful probes beyond their own performance. | Blind author solves scored beside stronger candidate answers. | 23 of 146 scored probes meet the operational beyond-author definition. | Intriguing exploratory evidence. | One author attempt is not a capability limit; the fixed evaluator may be biased. | Publish as exploratory, not as proof of unsolvability. |
| Adaptive follow-ups improve ranking. | Thirty oversight panels and two 50-model replications. | Follow-ups improve ten panels, leave eleven unchanged, and worsen nine; neither catalog ranking improves materially. | Evidence against a general benefit. | We have not tested verifier-oriented or council-based adaptation. | Publish the mixed result and make improved adaptation a future experiment. |

Primary artifacts:
`data/probe_effectiveness_results.json`,
`data/probe_self_study_results.json`,
`data/probe_evolution_results.json`, and
`docs/figures/probe-effectiveness/`.

## RQ4: Can We Improve Above-Level Judgment?

| Candidate claim | Best evidence | Current result | Strength | Remaining gap | Decision |
| --- | --- | --- | --- | --- | --- |
| Verifier-oriented probe guidance improves recognition above the judge's own capability. | Four matched nine-candidate panels; only private probe-design guidance changes. | Above-self recognition falls from 60% to 55%, while overall author accuracy rises from 63.9% to 66.0%. | Direct matched pilot evidence against the primary claim. | Four authors and one battery per condition cannot estimate a general effect. | Publish the null primary result and keep overall ranking separate. |
| Independent councils improve ranking on fixed evidence. | Sol, Llama, and Qwen independently rank identical archived answers; pairwise majority is compared with Sol alone. | Ordinary accuracy rises from 64.6% to 70.1% on all four panels; verifier accuracy rises from 72.2% to 75.0%, with one panel worse. | Consistent ordinary-condition pilot evidence. | Council composition is fixed and includes correlated model families. | Publish as promising, not universal. |
| Verifier guidance and councils reinforce one another. | Frozen 2x2 matched comparison. | The council gain is 5.6 points on ordinary evidence and 2.8 points on verifier evidence; mean interaction is -2.8 points. | Evidence against positive synergy in this pilot. | Interaction is noisy with four panels. | Do not claim complementarity. |

Primary artifacts:
`data/verifier_council_matched_v1_results.json`,
`docs/pilot_analysis_verifier_council_20260727.md`, and
`docs/figures/verifier-council/matched-results.svg`.

## Claims Not Ready

- A precise intelligence distance at which scalable oversight fails.
- A universal best question family.
- A universal optimal number of probes.
- Proof that a model cannot solve a probe it authored.
- A causal estimate of the effect of judge intelligence.
- A claim that free discussion is better or worse than structured evaluation.

## Experiment Gate

Before purchasing another run:

1. Name the claim and the exact figure or sentence the run could change.
2. Show why archived probes, answers, or crossed judgments cannot answer it.
3. Freeze the comparison, primary metric, stopping point, and exclusion rule.
4. Prefer a small matched design that isolates the missing factor.
5. Do not repeat a broad experiment solely because its variance is
   aesthetically inconvenient.

## Publication Audit Outcome

The current article can answer the four primary questions without another
paid run:

- **Retained:** strong judges recover broad capability differences; large
  differences are easier; judges often recognize clearly stronger candidates.
- **Qualified:** exact top rankings are unstable; judge-capability effects are
  suggestive rather than causal; probe labels are descriptive rather than
  prescriptive; independent councils are promising but composition-specific.
- **Removed:** claims of a precise oversight threshold, a universally best
  question type, a general benefit from one adaptive follow-up, or improved
  above-self recognition from verifier-oriented prompting.

The next paid study should target one observed mechanism failure, not repeat a
broad ranking or combine several interventions at once.
