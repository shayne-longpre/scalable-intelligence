# Probe Evolution Across Judge Capability

## Question

Do more capable judges invent different intelligence tests, and do their probes
change differently after they see candidate performance?

## Design

Eighteen fixed-protocol runs cover ten judges with external scores from 14.3 to
58.9. Each run contains five probes written before any answers were visible and
one evidence-conditioned follow-up. The panels are centered on each judge rather
than identical, so associations with judge score are descriptive.

The analysis harmonizes all 108 probes under taxonomy version `2026-07-27.2`.
It separates:

- the subject of a question from the evaluator's strategy;
- topical breadth from whether a probe actually separates candidates;
- explicit ties from accuracy on pairs the judge chose to separate;
- preplanned movement within the opening battery from evidence-conditioned
  adaptation.

## What Judges Asked

Math was the dominant family, but the repertoire was broad. A probe may receive
more than one label.

| Question family | Tagged probes |
| --- | ---: |
| Math reasoning | 50 |
| Scientific reasoning | 22 |
| Planning and strategy | 19 |
| Logic and consistency | 18 |
| Fluid reasoning | 15 |
| Verbal abstraction | 11 |
| Spatial and visual reasoning | 8 |
| Coding | 3 |
| Philosophy | 4 |

Eighty-nine of 108 probes posed a direct task. Stronger judges often wrote
compact, checkable technical problems, but sophisticated probes were not
exclusive to them. The batteries also included causal inference, systems
reasoning, constructed-language induction, experimental design, ethical
judgment, conceptual analogy, strategic planning, and self-critique.

## How Probes Evolved

Across the 72 transitions inside the prewritten opening batteries, 46 broadened
to a different topical family, 13 formed a progression, and 13 stayed in the
same area. Judges usually used the opening battery to sample complementary
capabilities rather than repeatedly escalating one task.

The 18 adaptive probes behaved differently: six broadened, seven deepened an
existing area, and five were generic follow-ups without a clear topical
relationship. Judges' stated plans most often raised difficulty, retested a
weakness, changed domain, or introduced an adversarial check. Their own
comparisons marked 92 probes informative and 16 limited.

One follow-up did not reliably improve holistic rankings. Across the full
30-panel oversight analysis, adaptation improved ten rankings, left eleven
unchanged, and worsened nine.

## Judge Capability

Judge score had almost no relationship with opening topical breadth
(`rho = -0.04`) or overall per-probe pair accuracy (`rho = -0.10`). The
association with accuracy on explicitly separated pairs was small
(`rho = 0.14`).

The clearest behavioral difference was tie use (`rho = 0.70`). Stronger judges
more often said that a probe had not distinguished two candidates. They also
described somewhat more varied adaptive intentions (`rho = 0.36`). This could
reflect better calibration, excessive caution, stricter rubrics, or probe
saturation; the current design cannot distinguish those explanations.

Models with higher external scores were less likely to call their own probes
informative (`rho = -0.62`). This is not evidence that their questions were
worse: “informative” is the judge's self-report, and stronger judges also used
far more ties. A future matched-evidence study should calibrate validity labels
independently.

## Same-Evidence Control

A crossed control gave Sol and Llama the exact archived answers from batteries
written by Grok, DeepSeek, Kimi, and Llama. Sol's mean five-probe pair accuracy
was 64.6%; Llama's was 63.2%. Which evaluator performed better changed with the
battery, while battery means ranged from 56.9% to 73.6%.

This indicates that probe design and evidence interpretation interact. It does
not establish a stable probe-author effect because each author supplied only one
battery. Full results are in
[`docs/pilot_analysis_probe_design_cross_20260727.md`](pilot_analysis_probe_design_cross_20260727.md).

## Fifty-Model Deep Dive

The same classifier was applied separately to the five-opening-probe Sol and
Fable catalog runs. This cohort contains 14 probes and is not included in the
18-run correlations because each judge evaluated 50 rather than nine
candidates.

The probes span math, science, reading, planning, coding, spatial reasoning,
logic, philosophy, and practical judgment. Thirteen were marked informative.
Both judges tied about 17% of opening-probe candidate pairs and ordered the
pairs they did separate with roughly 84% accuracy. Their adaptive probes were
less diagnostic than their opening batteries and did not materially improve
either global ranking.

## Measurement Audit

The expanded audit led to narrow classifier changes:

1. Locally negated indicators no longer create labels, while an independent
   positive requirement in the same prompt remains detectable.
2. Structural uses of words such as “translation” and “strategy” no longer
   trigger language or planning labels without supporting context.
3. Planning, mathematical construction, structural analogy, and self-critique
   gained phrases observed in the new probes.
4. Explicit ties remain separate from decided-pair accuracy.

These changes affect post-hoc labels only. They do not alter prompts, answers,
comparisons, or rankings.

## Interpretation

Stronger judges in this sample do not simply ask more diverse questions or
produce probes that align more closely with the external index. They differ
most clearly in how readily they declare a probe non-discriminating and, more
weakly, in the range of intentions they articulate for a follow-up. Probe
battery, candidate panel, and evaluator interact enough that future claims
about “better test designers” require repeated crossed batteries.

The interactive report is
[`runs/report_cards/probe_evolution_20260727_v2/report_card.html`](../runs/report_cards/probe_evolution_20260727_v2/report_card.html).
Machine-readable results are in
[`data/probe_evolution_results.json`](../data/probe_evolution_results.json).
