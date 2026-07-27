# Probe Effectiveness And Held-Out Generalization

## Question

Which model-authored probes actually distinguish candidate intelligence, and
can the taxonomy identify effective probes for a new judge?

## Design

The analysis uses 146 scored probes from 11 authors. A fixed evaluator scored
each archived candidate answer from zero to four, allowing a common per-probe
ordering to be compared with the external intelligence index.

Raw category averages are confounded by judge and panel. Two controls reduce
that problem:

1. Question-type effects are centered on each author's own mean probe accuracy,
   then averaged equally across authors.
2. The held-out test estimates label effectiveness using every *other* author
   and asks whether those estimates correctly order the excluded author's
   probes. No probe contributes to its own label estimate.

This is predictive validation, not a causal estimate of question type.

## Results

Mean per-probe pairwise accuracy was 58.4%. Probe-author intelligence had only
`rho = 0.10` association with diagnostic accuracy.

Planning and spatial probes were respectively 6.9 and 5.2 percentage points
above their authors' average probe accuracy. Their author-bootstrap intervals
were positive in this dataset. However, that pattern did not generalize into a
reliable selection rule:

| Held-out features | Mean author-level concordance | 95% author bootstrap |
| --- | ---: | ---: |
| Question types | 46.4% | 38.5-53.6% |
| Strategies and stage | 56.3% | 43.5-68.6% |
| All recorded labels | 48.8% | 40.6-55.9% |

Chance is 50%. Taxonomy labels are therefore useful descriptions of what
judges tried, but not yet validated predictors of which probes will work.

Author-declared objective probes were more often reference-valid than mixed
probes, 78.3% versus 65.6%, but their mean discrimination was nearly identical,
58.4% versus 58.2%. Probes intended for stronger models were more often
operationally beyond their author, 27.3% versus 10.9% for peer-targeted probes,
without a meaningful gain in overall pair accuracy.

## Multi-Round Behavior

Across the 18 fixed-protocol runs, prewritten batteries mostly broadened:
46 of 72 opening transitions changed capability area, while 13 progressed
within an area and 13 stayed in the same area. Adaptive probes split between
six broadening, seven deepening, and five generic follow-ups.

The fixed evaluator scored adaptive probes lower than opening probes on
external-order accuracy, 52.4% versus 59.5%. That comparison is not causal:
adaptive probes intentionally target smaller and harder uncertain subsets.
The cleaner ranking endpoint is also mixed, with ten oversight panels
improving, eleven unchanged, and nine worsening after one follow-up.

## Interpretation

The study supports three publication claims:

- Judges invent a broad repertoire of tests.
- Stronger authors do not automatically write more diagnostic tests.
- Broad taxonomy categories alone are insufficient for choosing the next
  probe.

The next probe-design experiment should test a specific mechanism rather than
another category sweep. The clearest candidate is verifier-oriented probing:
ask judges to prefer tasks whose answer quality they can check reliably even
when producing the best answer may exceed their own capability. A council of
independent judges is a second, separable intervention.

## Artifacts

- `data/probe_effectiveness_results.json`
- `data/probe_self_study_results.json`
- `docs/figures/probe-effectiveness/question-type-effects.svg`
- `docs/figures/probe-effectiveness/held-out-labels.svg`
