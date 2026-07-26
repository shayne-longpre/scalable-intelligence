# Oversight Frontier Design

## Research Question

Can an AI judge recognize and correctly rank anonymous candidate models that
are more capable than the judge itself?

## Study

Six popular model families span a wide range of external capability estimates:

| Judge | Organization | External index | Candidate panel |
| --- | --- | ---: | --- |
| GPT-5.6 Sol | OpenAI | 58.9 | 46.0–59.9 |
| Gemini 3.5 Flash | Google | 50.2 | 44.4–59.9 |
| Qwen 3.7 Max | Alibaba | 46.0 | 40.0–50.2 |
| MiniMax M2.7 | MiniMax | 38.1 | 29.6–42.2 |
| Claude Haiku 4.5 | Anthropic | 29.6 | 21.4–34.7 |
| Mistral Large 3 | Mistral AI | 15.9 | 7.6–29.6 |

Every panel contains seven candidates, including the judge behind an anonymous
participant ID. Each panel also contains models above and below the judge on
the external index. The routes, parameters, seeds, and exact panels are frozen
in [`studies/oversight_frontier_v1.json`](../studies/oversight_frontier_v1.json).

The judge:

1. authors five complementary probes before seeing any answer;
2. sends each probe unchanged to all seven candidates;
3. compares the seven answers directly, one probe at a time;
4. merges the comparisons into a full ranking;
5. selects at most four uncertain candidates for one common adaptive probe;
6. updates the ranking after the follow-up.

Candidate calls use fresh context for every probe. Judges never see provider
names, model names, catalog scores, or release information.

## Primary Analysis

Candidate pairs are grouped by their position relative to the judge:

- both below the judge;
- crossing the judge or involving the judge's anonymous candidate;
- both above the judge;
- close in external score, reported separately because the index is noisy.

The main outcomes are final pairwise accuracy, the share of externally stronger
candidates placed above the judge's anonymous self, and the difference between
the five-probe ranking and the ranking after adaptation. The report also shows
probe types, adaptive targeting, malformed-output recovery, and
provider-reported spend.

The judge's own claim that a probe was informative is not treated as ground
truth. Probe validity is audited separately: a judge can create an impossible
or underspecified task and confidently prefer the wrong answer.

The completed pilot results and interpretation are in
[`pilot_analysis_oversight_frontier_20260725.md`](pilot_analysis_oversight_frontier_20260725.md);
the reader-facing scorecard is
[`site/oversight.html`](site/oversight.html).

## Reproducibility

Resolved configs are build artifacts rather than duplicated source files:

```bash
python -m scripts.build_adaptive_judge_study \
  --study studies/oversight_frontier_v1.json \
  --output-dir runs/configs/oversight_frontier_v1
```

After all six conditions complete:

```bash
python -m scripts.analyze_oversight_frontier \
  --study studies/oversight_frontier_v1.json \
  --runs-root runs \
  --output-dir runs/report_cards/oversight_frontier_v1 \
  --probe-audit data/oversight_frontier_probe_audit.json \
  --published-json data/oversight_frontier_results.json
```

An order-robustness check must replay the exact same probes and answers under a
new presentation order. Regenerating probes or candidate answers is a new
replication, not an order test.
