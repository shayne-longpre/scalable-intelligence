# Oversight Frontier Replication

## Question

Can an anonymous AI judge recognize models that are more capable than itself?
How much do the answer order and candidate panel change its verdict?

The frozen protocol used six independent judges from six organizations. Each
judge wrote five common opening probes for seven anonymous candidates,
including itself, then chose up to four uncertain candidates for one common
adaptive probe. The replication changed the candidate panels and all relevant
seeds, but not the protocol or judge settings.

The external catalog is a noisy reference measurement, not ground truth.
Results therefore separate close pairs from large capability gaps.

## Main Result

| Study | Correct pairs | Pair accuracy | Stronger above judge | All self-relative |
| --- | ---: | ---: | ---: | ---: |
| Panel 1 | 89/126 | 70.6% | 13/16 | 23/36 |
| Panel 2 | 99/126 | 78.6% | 7/16 | 26/36 |
| **Pooled** | **188/252** | **74.6%** | **20/32** | **49/72** |

The two panels disagree on the raw stronger-than-judge statistic because it
treats every positive catalog difference as exact. After requiring a candidate
to lead the judge by at least two external-score points, judges recognized 15
of 22. At five points, they recognized 7 of 10. Only one candidate was more
than ten points above its judge, so this design still has little evidence about
very large scalable-oversight gaps.

## Capability Separation

| External-score gap | Panel 1 | Panel 2 | Pooled |
| --- | ---: | ---: | ---: |
| Less than 2 | 9/22 (40.9%) | 17/23 (73.9%) | **26/45 (57.8%)** |
| 2 to 5 | 28/40 (70.0%) | 22/35 (62.9%) | **50/75 (66.7%)** |
| 5 to 10 | 31/42 (73.8%) | 24/31 (77.4%) | **55/73 (75.3%)** |
| More than 10 | 21/22 (95.5%) | 36/37 (97.3%) | **57/59 (96.6%)** |

Large gaps replicated. Fine ordering did not: sub-two-point accuracy moved by
33 percentage points between panels. The external index itself is least
defensible at that resolution, so near-ties should remain an uncertainty band,
not a strict correctness test.

Ranking two candidates that were both above the judge was harder than ranking
two below it: pooled accuracy was 20/30 (66.7%) above and 37/50 (74.0%) below.
Candidate-versus-self accuracy reached 8/8 for ten-point absolute gaps, although
those eight comparisons mostly involved candidates far below the judge.

## Panel And Order Sensitivity

Five judges improved on the second panel and one worsened. Individual changes
ranged from -9.5 points for Mistral to +28.6 for Haiku. Since every judge saw a
different local panel, this movement should not be interpreted as a stable
ordering of judge quality.

The separate order audit held every probe and candidate answer fixed. It
reshuffled all six probe-level answer presentations for Sol, MiniMax, and
Mistral:

| Judge | Kendall agreement | Same top | Accuracy change |
| --- | ---: | ---: | ---: |
| Sol 5.6 | 0.90 | Yes | +4.8 points |
| MiniMax M2.7 | 0.52 | Yes | +14.3 points |
| Mistral Large 3 | 0.52 | Yes | +4.8 points |

All 39 evidence items per condition matched their source streams exactly, and
the replays made zero candidate calls. The strongest judge was comparatively
stable; the two lower-capability judges were not. One seeded order is therefore
insufficient for a high-confidence fine ranking.

## Adaptation

Across both panels, the adaptive probe improved six rankings, left four
unchanged, and worsened two. Pooled accuracy rose from 72.6% after the five
opening probes to 74.6% after the follow-up. The second panel contained no
worsened ranking, but Panel 1 contained two. Keeping every checkpoint remains
more defensible than assuming the final adaptive round is always best.

## Probe Audit

The second panel again mixed strong tests with construct-validity problems:

- Sol's artificial-language probe was compositional, bidirectional, and
  directly checkable.
- Gemini's finite logician dialogue had a unique epistemic solution,
  `(2, 3, 10)`.
- Qwen's Kepler-symmetry problem was technically demanding, but arbitrary
  banned-word and heading constraints contaminated the mathematical signal.
- MiniMax asked candidates to infer a unique relation from `4:5`, although
  infinitely many rules fit one pair.
- Haiku's mentorship decision tested genuine value conflict but had no crisp
  external answer.
- Mistral required exact causal-effect percentages that were not identified by
  the supplied data.

These examples are a deliberately small audit, not an exhaustive validity
estimate. The versioned sample is
`data/oversight_frontier_v2_probe_audit.json`.

## Reliability And Cost

Panel 2 used 352 recorded model calls across all attempts and cost $16.3029 in
provider-reported usage. Two MiniMax attempts failed before the partial run was
resumed. Targeted repairs reused every successful archived answer and retried
only unavailable cells; four answers remained explicitly unavailable in the
selected runs. A malformed MiniMax validity field also exposed and motivated a
type-checking fix in the structured-output monitor.

Both panels together cost $35.8712. The three order replays added $1.5921 and
27 judge calls, with no candidate calls.

## Interpretation

The replicated result is not a single smooth oversight frontier. It is a
discrimination curve:

1. Models can usually recover broad capability differences.
2. Fine ordering is sensitive to panel, presentation order, and noise in the
   external reference.
3. Recognizing a clearly stronger model is possible, but these panels contain
   too few large above-judge gaps to estimate that capability precisely.
4. Probe quality is part of evaluator quality. A difficult-looking but
   underdetermined task can create confident ranking noise.

The combined report card is
`runs/report_cards/oversight_frontier_replication/report_card.html`. The
second-panel report is
`runs/report_cards/oversight_frontier_v2/report_card.html`, and the exact-order
audit is
`runs/report_cards/oversight_frontier_v1_order_replay/report_card.html`.
