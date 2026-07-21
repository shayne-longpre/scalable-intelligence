# Adaptive Judges: Contrasting Hardening Pilots

These pilots tested whether the adaptive protocol remains useful under two
different regimes: an affordable medium judge separating a broad capability
range, and a strong judge separating four close frontier candidates. They are
engineering and methodological hardening runs, not a controlled comparison of
judge quality. The rosters and prompt versions differ.

Comparative report:
`runs/report_cards/adaptive_pilots_20260720_v4/report_card.html`.

## Setups

| Pilot | Judge | Candidates | Schedule | Routed Q/A | Cost |
| --- | --- | --- | --- | ---: | ---: |
| Broad roster | GPT-5.4 Mini | Fable 5, GPT-5 Mini, GPT-4o, Llama 3.1 8B | `[4,1,1,1,1]` | 25 | $0.94 |
| Close roster | GPT-5.6 Sol | Fable 5, GPT-5.4, Gemini 3.5 Flash, Sonnet 4.6 | `[4,1,1,1,1]` | 28 | $7.01 lineage total |

Candidate identities and the external catalog were hidden from both judges.
The catalog is a fallible comparison prior, not ground truth. The broad pilot
used prompt set v1. The close pilot was resumed while runtime issues were fixed;
its replay provenance preserves the source prompts, and its final snapshot is
v3. Neither run should be used as a clean prompt-version ablation.

## Ranking Checkpoints

The broad judge recovered the catalog order after the opening battery and never
changed it:

| Round | Probes | Ranking | Pairwise accuracy | Confidence |
| ---: | ---: | --- | ---: | ---: |
| 1 | 4 | P2 > P4 > P1 > P3 | 1.00 | 0.95 |
| 2 | 5 | P2 > P4 > P1 > P3 | 1.00 | 0.96 |
| 3 | 6 | P2 > P4 > P1 > P3 | 1.00 | 0.93 |
| 4 | 7 | P2 > P4 > P1 > P3 | 1.00 | 0.94 |
| 5 | 8 | P2 > P4 > P1 > P3 | 1.00 | 0.95 |

Sol's close-roster ranking changed in three adaptive rounds:

| Round | Probes | Ranking | Pairwise accuracy | Confidence |
| ---: | ---: | --- | ---: | ---: |
| 1 | 4 | P2 > P1 > P4 > P3 | 0.67 | 0.79 |
| 2 | 5 | P2 > P4 > P1 > P3 | 0.83 | 0.86 |
| 3 | 6 | P2 > P1 > P4 > P3 | 0.67 | 0.89 |
| 4 | 7 | P2 > P4 > P1 > P3 | 0.83 | 0.91 |
| 5 | 8 | P2 > P4 > P1 > P3 | 0.83 | 0.90 |

The final disagreement is Sonnet 4.6 versus Gemini 3.5 Flash, adjacent catalog
ranks 8 and 7. That is weak evidence of judge error because the prior difference
is itself uncertain. Both judges selected the catalog's top candidate.

## What The Judges Did

The medium judge's opening battery included hidden-rule induction, warehouse
planning, solver auditing, and Bayesian value of information. Three of the four
comparisons were limited by an inconsistent example, an impossible grid, or
underspecified semantics. The value-of-information problem was clearly
diagnostic. Its later targeting was coherent, but every follow-up became a
register or sequence simulation. Round 3 was correctly called limited; Rounds 4
and 5 were described as informative even though both leading candidates were
fully correct and the ranking did not move. The judge separated a broad ladder
successfully, but its evaluation repertoire narrowed rather than developed.

Sol's opening probes covered hierarchical Bayesian inference and value of
information, an adjacent-pile optimization proof, causal principal-stratum
bounds, and an exactly-once payment impossibility argument. Three were
informative and one saturated. Later probes changed domain in response to
specific uncertainties: program semantics and aliasing, scientific-model
auditing under rounded measurements, deterministic recurrence algorithms, and
causal decision analysis. Sol also changed the comparison subset, temporarily
reintroducing the weakest candidate when a boundary needed fresh evidence.

This distinction is more informative than a raw count of adaptive turns. Both
judges chose all four target sets requested in their prior judgments and covered
all declared uncertain pairs. The medium judge produced zero rank-changing
follow-ups and repeated one capability area. Sol produced three rank-changing
follow-ups, three adaptive broadenings, and one final limited probe.

## Round Count

For this broad roster, Round 1 already contained the useful ordering. One
confirmation round would have been enough; by Round 3 the judge had produced a
limited repeat with no ranking change. For the close roster, evidence remained
useful through Round 4, or seven cumulative probes. Round 5 was limited, did not
change the ranking, and reduced confidence from 0.91 to 0.90.

The practical default is therefore:

- `[4,1,1,1]` for close-roster research runs;
- `[4,1]` or `[4,1,1]` for broad ladders when cost matters; and
- `[4,1,1,1,1]` for stress tests and stopping-policy development.

The protocol still records a complete judgment after every round. A future
stopping rule should be gold-blind and evaluated over replications. A plausible
diagnostic is stable ranking plus a targeted limited or invalid probe, but the
current sample is too small to freeze that rule.

## Refinements From These Pilots

The v3 judge prompt now requires an internal check for consistency,
answerability, difficulty, and feasible scope. After a limited probe, it asks
the judge to change capability area or raise difficulty unless a repeat tests a
specific unresolved signal. The comparison prompt now separates probe validity
from confidence and defines saturation and style-only differences as limited.
The cumulative judgment treats a bad probe as evidence about test quality, not
candidate ability.

The extraction layer now emits one adaptive-decision record per follow-up. It
links the prior uncertainty, requested and actual candidates, target changes,
planned strategy, all question-type labels, reported validity, resulting rank
change, and confidence delta. This makes question evolution inspectable without
an analyst model intervening in the live experiment.

The state-tracking taxonomy indicators were narrowed, generic domain words such
as `define`, `decision`, `rule`, and `API` were replaced with diagnostic
phrases, and a Computer Systems type now captures protocol and crash-recovery
probes. Primary question type selection uses the amount of matching evidence
rather than taxonomy file order. The behavior audit's `thin_answer` check was
also corrected to count actual words; it had previously counted unique long
content terms and falsely flagged numerical derivations.

## Runtime Reliability

The broad run completed in 49 paid calls. Three GPT-5.4 Mini probe-writing calls
used their entire reasoning budget without visible text and recovered on a
bounded low-reasoning retry. No structured JSON needed repair, and current-code
revalidation reports no errors.

The close run required several resumes while route-specific output budgets were
qualified. Its complete five-run lineage contains 78 calls and $7.01 in
provider-reported cost; the final resume made three new calls for $0.28. The
failures led to four concrete fixes:

1. completed concurrent responses are committed before a sibling error is
   raised;
2. replay validation uses semantic stream identities rather than local turn
   numbers;
3. a provider-rejected recovery override falls back once to qualified primary
   parameters and is recorded; and
4. reasoning and visible-output budgets can be qualified per model and route.

The completed close transcript has no JSON repairs, no remaining revalidation
findings, and no behavior-audit findings.

## Interpretation

The protocol is now producing the phenomenon it was designed to expose: not
just final rankings, but differences in test design, adaptation, saturation
recognition, candidate targeting, and belief revision. Stronger-looking
evaluation behavior appeared in the Sol transcript, but these two runs cannot
establish a judge-capability effect because judge, roster, and prompt version
were not held constant.

The next experiment should freeze v3 and compare judges on the same roster and
candidate evidence across at least three seeds. That is the point at which
question-evolution classifications and stopping behavior can be analyzed as
distributions rather than anecdotes.
