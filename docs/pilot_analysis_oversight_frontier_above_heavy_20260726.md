# Oversight Frontier: Ten-Judge Extension

## Question

Can models across the capability spectrum recognize anonymous candidates that
are more capable than themselves, and how large must the capability margin be?

## Design

The third wave keeps the earlier protocol fixed and expands the sampling layer:

- ten independent judges from ten model providers;
- nine anonymous candidates per judge;
- five common opening probes authored before answers are visible;
- one judge-selected adaptive probe for at most four candidates;
- five candidates above the judge, anonymous self, and three below, where the
  catalog permits;
- one model above Sol and seven below it, because the catalog contains no other
  model scored above Sol.

The exact routes, parameters, panels, and seeds are frozen in
[`studies/oversight_frontier_v3_above_heavy.json`](../studies/oversight_frontier_v3_above_heavy.json).

## Repaired Third-Wave Results

The ten judges correctly ordered 281 of 360 candidate pairs (78.1%). They placed
30 of 46 externally stronger candidates above their anonymous selves (65.2%)
and got 60 of 80 candidate-versus-self relations correct (75.0%). Ordering two
candidates that were both above the judge reached 64 of 90 pairs (71.1%).

The adaptive follow-up improved three rankings, left four unchanged, and
worsened three. Aggregate pairwise accuracy was 78.1% both after the five
opening probes and after adaptation. A single follow-up is therefore not
reliably beneficial.

The original analysis used 19 explicitly unavailable candidate answers. A
bounded replay recovered 18 while preserving every probe, anonymous ID, and
adaptive target. The last cell was rejected by the Fable route's content
filter. Recomputed rankings moved final pair accuracy from 77.2% to 78.1% but
left the primary superior-recognition result exactly unchanged at 30/46.
Individual judges moved more than the aggregate because their comparison and
ranking calls were sampled again; repair is not a deterministic judgment
replay.

## All Three Waves

Pooling the two earlier seven-candidate panels and the new nine-candidate panel
gives 22 judge panels, ten distinct judges, and 612 candidate pairs:

| Outcome | Result |
| --- | ---: |
| Final pair ordering | 469/612 (76.6%) |
| Stronger candidate above judge | 50/78 (64.1%) |
| All candidate-versus-self relations | 109/152 (71.7%) |
| Adaptive improved / unchanged / worsened | 9 / 8 / 5 |

Recognition of stronger candidates did not reveal a sharp capability threshold:

| Candidate lead over judge | Recognized | Rate | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| <2 points | 16/26 | 61.5% | 42.5%-77.6% |
| 2-5 points | 12/20 | 60.0% | 38.7%-78.1% |
| 5-10 points | 15/24 | 62.5% | 42.7%-78.8% |
| 10+ points | 7/8 | 87.5% | 52.9%-97.8% |

The intervals overlap substantially. Capability margin helps at the largest
separation, but judge-specific behavior and panel composition remain at least
as important in this sample.

## Judge Coverage

“Stronger tested” counts candidate appearances across panels. “Unique stronger”
deduplicates models reused across a judge's panels.

| Judge | External score | Panels | Stronger recognized | Unique stronger | All-pair accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol | 58.9 | 3 | 1/3 | 1 | 74% |
| Grok 4.5 | 53.8 | 1 | 5/5 | 5 | 86% |
| Gemini 3.5 Flash | 50.2 | 3 | 9/11 | 7 | 78% |
| Qwen 3.7 Max | 46.0 | 3 | 8/11 | 7 | 68% |
| DeepSeek V4 Pro | 44.3 | 1 | 4/5 | 5 | 92% |
| MiniMax M2.7 | 38.1 | 3 | 8/11 | 9 | 82% |
| Kimi K2 Thinking | 32.7 | 1 | 1/5 | 5 | 58% |
| Claude Haiku 4.5 | 29.6 | 3 | 4/11 | 8 | 72% |
| Mistral Large 3 | 15.9 | 3 | 5/11 | 6 | 77% |
| Llama 4 Maverick | 14.3 | 1 | 5/5 | 5 | 89% |

The striking result is heterogeneity, not a monotonic frontier. Grok, MiniMax,
and Llama were relatively well calibrated in these panels, while Kimi placed
only one of five externally stronger candidates above itself. One panel per new
judge is not enough to treat these as stable model traits.

## Protocol Health

The original third wave used 621 model calls and cost $32.07 according to
provider reports. Including bounded repairs, the evidence lineage used 809
calls and cost $42.81. One of 487 routed candidate answers remains unavailable:
a route-level content-filter rejection. Two selected repairs changed runtime
parameters and are explicitly marked as sensitivity runs; one increased a
completion ceiling for a response that had exhausted it, without changing the
probe or reasoning effort.

DeepSeek completed without missing evidence or JSON repair, but its x-high
judge condition took about one hour. This is a valid route for a primary study,
not a practical setting for rapid iteration.

## Interpretation

The experiment supports three conclusions:

1. Models can recognize systems above themselves, but do so inconsistently.
2. A large external-score lead improves recognition only weakly in this sample;
   there is no defensible universal margin threshold yet.
3. Probe design, self-placement behavior, and panel composition matter enough
   that additional independent panels are more valuable than fitting a smooth
   frontier to these heterogeneous observations.

The repaired third-wave report is
[`runs/report_cards/oversight_frontier_v3_repaired/report_card.html`](../runs/report_cards/oversight_frontier_v3_repaired/report_card.html).
The repaired pooled report is
[`runs/report_cards/oversight_frontier_synthesis_repaired/report_card.html`](../runs/report_cards/oversight_frontier_synthesis_repaired/report_card.html),
with machine-readable results in
[`data/oversight_frontier_synthesis_repaired_results.json`](../data/oversight_frontier_synthesis_repaired_results.json).
