# Probe Evolution Across Judge Capability

## Question

Do more capable judges invent different intelligence tests, and do their probes
change differently after they see candidate performance?

## Design

Ten judges spanning external intelligence scores from 14.3 to 58.9 each
evaluated a nine-model anonymous panel. The protocol was fixed: five probes
written before any answers were visible, followed by one evidence-conditioned
probe. The panels were centered on each judge rather than identical, so
associations with judge score are descriptive.

The analysis harmonizes all 60 probes under taxonomy version `2026-07-27.1`.
It distinguishes:

- topical breadth from whether a probe actually separates candidates;
- explicit ties from confident pair orderings;
- preplanned opening-battery progression from evidence-conditioned adaptation;
- the subject of a question from the evaluator's strategy.

## What Judges Asked

Math was the most common question family, but the repertoire was broad. A
single probe can receive more than one label.

| Question family | Tagged probes |
| --- | ---: |
| Math reasoning | 29 |
| Planning and strategy | 13 |
| Fluid reasoning | 12 |
| Logic and consistency | 12 |
| Scientific reasoning | 10 |
| Verbal abstraction | 6 |
| Spatial and visual reasoning | 4 |
| Coding | 3 |

The strongest judges often wrote compact, checkable technical problems. Sol
combined causal identification, distributed-systems semantics, algebraic
invariants, information theory, and constructed-language induction. Grok mixed
a meta-evaluation question with scientific model-building, logic, and game
theory. Other judges used a wider range of open scenarios: ethical choices,
counterfactual worlds, strategic planning, conceptual synthesis, and
self-correction.

## How Probes Evolved

Of the 40 transitions within the prewritten five-probe batteries, 28 broadened
to another topical family, eight stayed in the same area, and four formed a
clear progression. Judges generally used the opening battery to sample several
capabilities rather than repeatedly escalating one task.

After answers became visible, four judges broadened, four deepened an existing
area, and two wrote a more generic follow-up. Their stated adaptive plans most
often raised difficulty (8/10 judges), adversarially checked an answer (6/10),
retested a weakness (6/10), or changed domain (5/10). Eight of the 60 probes
were marked limited by their own judges; the rest were marked informative.

## Judge Capability

There was little association between judge score and topical breadth
(`rho = 0.19`) or accuracy on pairs the judge explicitly separated
(`rho = -0.05`). Stronger judges did, however, use many more explicit ties
(`rho = 0.70`) and describe more kinds of adaptive intent (`rho = 0.47`).

This tie behavior changes the apparent result. Giving half credit to tied pairs
produces a negative association between judge score and probe diagnosticity
(`rho = -0.40`). Once tied pairs are excluded, the association nearly
disappears. Sol tied 52.8% of opening-probe pairs and DeepSeek tied 82.2%;
most lower-scoring judges forced complete orders. This may reflect better
recognition of probe saturation, excessive caution, or rubric differences. One
run per judge cannot distinguish those explanations.

The single adaptive probe improved three final rankings, left four unchanged,
and worsened three. It did not supply evidence for a universal benefit from one
follow-up.

## Fifty-Model Deep Dive

The same classifier was applied separately to the new Sol and Fable catalog
runs. This is a qualitative deep dive, not part of the ten-judge correlation:
both judges are near the top of the external index and each evaluated 50 rather
than nine candidates.

The two judges wrote 14 probes spanning math, science, reading, planning,
coding, spatial reasoning, logic, philosophy, and practical judgment. Thirteen
were marked informative. On the five opening probes, both judges tied about 17%
of candidate pairs and ordered the pairs they did separate with about 84%
accuracy. Sol broadened its first adaptive probe and then wrote a general
follow-up; Fable deepened an earlier evaluation pattern twice.

Their adaptive probes were less diagnostic than their opening batteries. Sol's
final holistic ranking changed by `+0.2` percentage points and Fable's by
`-1.1` points relative to the five-probe checkpoint. Both judges articulated
targeted rationales, but richer adaptation did not guarantee a better global
ranking.

## Measurement Audit

The pilot led to three targeted classifier changes:

1. Explicit ties are now reported separately from decided-pair accuracy.
2. Policy or treatment “programs” no longer trigger the coding label.
3. Explicit exclusions such as “no tool use” no longer trigger tool-use labels;
   cube-rotation and statistical-uncertainty wording gained missing coverage.

These are post-hoc measurement changes only. They do not alter any live prompt,
answer, comparison, or ranking.

## Interpretation

The main difference in this sample is not that stronger judges simply cover
more domains or obtain higher per-probe agreement with the external ordering.
It is that they are more willing to say a probe did not discriminate, while
also articulating richer adaptive plans. The next useful test is repeated,
matched panels or shared answer evidence, which can separate judge behavior
from panel difficulty and sampling variance.

The interactive report, including both separately labeled cohorts, is
[`runs/report_cards/probe_evolution_20260727/report_card.html`](../runs/report_cards/probe_evolution_20260727/report_card.html).
Machine-readable results are in
[`data/probe_evolution_results.json`](../data/probe_evolution_results.json).
