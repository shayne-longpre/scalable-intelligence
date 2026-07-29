# Robustness And Sensitivity Appendix

This appendix gathers the checks that qualify the headline results. It is
generated from the same frozen JSON artifacts as the public article; no
additional model calls are involved.

## External Reference And Estimated Anchors

The primary catalog metric uses 47 candidates with directly reported external
scores. Three deliberately weak anchors have estimated scores and are excluded
from that endpoint.

| Judge | Direct models | Direct pair accuracy | All models | Accuracy with estimates |
| --- | ---: | ---: | ---: | ---: |
| Sol | 47 | 84.6% | 50 | 86.4% |
| Fable | 47 | 88.2% | 50 | 89.5% |

Including the estimated lower-tail anchors raises measured accuracy slightly
because those candidates are easy to separate. It does not drive the primary
86.4% result.

## Opening Probe Budget

| Judge | 5 probes | 10 probes | Rank tau | Top-five overlap |
| --- | ---: | ---: | ---: | ---: |
| Fable | 84.6% | 88.2% | 0.85 | 80% |
| Sol | 80.9% | 84.6% | 0.78 | 60% |

The mean rank agreement is Kendall 0.81, with
70% mean top-five overlap. Aggregate accuracy is
higher with the larger opening battery; exact frontier ordering still varies.

## Answer Presentation Order

The 50-model catalog replay changed only answer order. Its ranking agreed with
the original at Kendall 0.81; pairwise accuracy
moved from 86.7% to
83.7%. The top three stayed fixed and
8 of the top ten overlapped.

3 smaller oversight panels were
also replayed under new answer orders with
exact evidence identity. Mean rank agreement was Kendall
0.65, and the top candidate
remained fixed in all 3 panels.
Order is therefore a real source of individual-rank uncertainty, but it did not
erase the broad signal.

## Probe Battery Versus Evidence Interpreter

With the same archived evidence, judge-to-judge Kendall agreement was
0.84 on Sol-authored evidence and
0.77 on Fable-authored evidence.
For the same judge across different batteries, agreement was
0.67 for Sol and
0.60 for Fable. In this cross-over,
the chosen evidence changed rankings more than swapping the interpreter of
fixed evidence.

## Probe Count And Adaptive Follow-Ups

In the catalog experiment, Sol moved from
84.6% after
10 probes to
84.1% after
two follow-ups. Fable moved from
88.2% to
85.6%.
Across 30 oversight panels, one follow-up improved
10 rankings, left
11 unchanged, and worsened
9. More evidence was not monotonically better.

## Oversight Uncertainty

The pooled superior-recognition estimate is
65.3% across
30 panels and 10 judges.
Wilson intervals are wide within margin cells; panel-bootstrap intervals for
the standardized sub-ten-point rates are:

| Judge band | Rate | Panel-bootstrap 95% interval |
| --- | ---: | ---: |
| Lower Third | 48.0% | 25.5%–69.7% |
| Middle Third | 60.1% | 43.3%–75.1% |
| Upper Third | 85.8% | 68.6%–96.9% |

The observed judge-capability pattern is suggestive, not causal: judge,
provider, panel, and probe battery remain entangled.

## Missing Evidence And Provider Failures

The primary catalog opening contains
1000 candidate answers, with
1 unavailable. The adaptive
rounds contain 40 answer
cells, with 3 unavailable. The
pooled oversight study retains 7 of
1345 unavailable answer cells
(0.5%);
the matched extension has none. Repairs reused successful evidence and preserve
lineage. Provider incompatibility is treated as missing evidence, never as an
intelligence score.

## Council Scope

The three-member council improved ordinary-battery accuracy from
64.6% to
70.1% across
4 panels. On
verifier-oriented evidence it moved from
72.2% to
75.0%. The interaction was
-2.8%; the interventions were not complementary.
This is a fixed-composition pilot, not evidence for every possible council.

## Remaining Limits

- The external intelligence index is a reference measurement, not ground truth.
- Nearby frontier scores and exact model settings are uncertain.
- Taxonomy labels are post-hoc and multi-label; they describe behavior but do
  not yet select reliably diagnostic probes.
- One fixed evaluator scored the 147-probe corpus.
- A single self-solve attempt cannot establish that a problem is impossible for
  its author.
- Free discussion remains a qualitative extension, not a quantitative claim
  about machine social order.

Primary artifacts:
`data/research_question_synthesis.json`,
`data/catalog_ladder50_opening10_stability.json`,
`data/oversight_frontier_v1_order_replay_results.json`,
`data/probe_effectiveness_results.json`, and
`data/verifier_council_matched_v1_results.json`.
