# Oversight Frontier Design

## Research Question

Can an AI judge recognize and correctly rank anonymous candidate models that
are more capable than the judge itself?

## First Study

Three judges cover distinct external capability levels:

| Judge | External index | Panel |
| --- | ---: | --- |
| GPT-5.6 Sol | 58.9 | One candidate just above; six at or below |
| GPT-5.4 Mini | 40.0 | Three above; three below; judge included |
| Claude Haiku 4.5 | 29.6 | Three above; three below; judge included |

Each panel has seven candidates. The exact routes and external scores are in the
selection files referenced by `studies/oversight_frontier_v1.json`.

The judge:

1. authors four complementary probes before seeing any answer;
2. sends each probe unchanged to all seven candidates;
3. compares the seven answers directly, probe by probe;
4. merges the evidence into a full ranking;
5. selects at most four uncertain candidates for one common adaptive probe;
6. updates the ranking after the follow-up.

Each condition has two runs. Only anonymous participant IDs and seeded answer
order change. The prompts, candidates, inference settings, and schedule remain
fixed.

## Primary Analysis

Candidate pairs are grouped by their position relative to the judge's external
score:

- both below the judge;
- crossing the judge, with one candidate below and one above;
- both above the judge;
- near the judge, reported separately because the external index is noisy.

The central plot is pairwise accuracy by judge capability and candidate gap
from the judge. Supporting results report whether the judge identifies a
superior candidate, agreement across the two permutations, rank churn after
the adaptive probe, probe taxonomy, and spend.

This first study does not estimate a sharp capability threshold. Three judges,
small panels, and two permutations are enough to expose the basic pattern and
failure modes before adding more judge levels.

## Reproducibility

The study manifest is `studies/oversight_frontier_v1.json`. Its six generated
configs are `examples/oversight_frontier_{sol,mini,haiku}_r{1,2}.openrouter.json`.
All use the existing provider-neutral independent-judge protocol.
