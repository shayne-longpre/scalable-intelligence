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

## Third-Wave Results

The ten judges correctly ordered 278 of 360 candidate pairs (77.2%). They placed
30 of 46 externally stronger candidates above their anonymous selves (65.2%)
and got 58 of 80 candidate-versus-self relations correct (72.5%). Ordering two
candidates that were both above the judge reached 60 of 90 pairs (66.7%).

The adaptive follow-up improved four rankings, left three unchanged, and
worsened three. Aggregate pairwise accuracy moved from 78.3% after the five
opening probes to 77.2% after adaptation. A single follow-up is therefore not
reliably beneficial.

## All Three Waves

Pooling the two earlier seven-candidate panels and the new nine-candidate panel
gives 22 judge panels, ten distinct judges, and 612 candidate pairs:

| Outcome | Result |
| --- | ---: |
| Final pair ordering | 466/612 (76.1%) |
| Stronger candidate above judge | 50/78 (64.1%) |
| All candidate-versus-self relations | 107/152 (70.4%) |
| Adaptive improved / unchanged / worsened | 10 / 7 / 5 |

Recognition of stronger candidates did not reveal a sharp capability threshold:

| Candidate lead over judge | Recognized | Rate | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| <2 points | 17/26 | 65.4% | 46.2%-80.6% |
| 2-5 points | 11/20 | 55.0% | 34.2%-74.2% |
| 5-10 points | 16/24 | 66.7% | 46.7%-82.0% |
| 10+ points | 6/8 | 75.0% | 40.9%-92.9% |

The intervals overlap substantially. Capability margin helps at the largest
separation, but judge-specific behavior and panel composition remain at least
as important in this sample.

## Judge Coverage

“Stronger tested” counts candidate appearances across panels. “Unique stronger”
deduplicates models reused across a judge's panels.

| Judge | External score | Panels | Stronger recognized | Unique stronger | All-pair accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5.6 Sol | 58.9 | 3 | 1/3 | 1 | 73% |
| Grok 4.5 | 53.8 | 1 | 5/5 | 5 | 78% |
| Gemini 3.5 Flash | 50.2 | 3 | 9/11 | 7 | 76% |
| Qwen 3.7 Max | 46.0 | 3 | 8/11 | 7 | 68% |
| DeepSeek V4 Pro | 44.3 | 1 | 4/5 | 5 | 92% |
| MiniMax M2.7 | 38.1 | 3 | 10/11 | 9 | 85% |
| Kimi K2 Thinking | 32.7 | 1 | 0/5 | 5 | 50% |
| Claude Haiku 4.5 | 29.6 | 3 | 4/11 | 8 | 73% |
| Mistral Large 3 | 15.9 | 3 | 4/11 | 6 | 81% |
| Llama 4 Maverick | 14.3 | 1 | 5/5 | 5 | 89% |

The striking result is heterogeneity, not a monotonic frontier. Grok, MiniMax,
and Llama were well calibrated relative to anonymous self in these panels;
Kimi ranked all five externally stronger candidates below itself. One panel per
new judge is not enough to treat these as stable model traits.

## Protocol Health

The third wave used 621 model calls and cost $32.07 according to provider
reports. Nineteen of 488 routed candidate answers (3.9%) remained unavailable
after bounded retries. They were single-probe gaps: every affected candidate
still answered at least four opening probes, and no anonymous self candidate
was missing. Missingness was concentrated in slow reasoning routes, so it is
not random. The raw rankings remain interpretable, but repaired evidence should
be reported separately before treating small differences as definitive.

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

The third-wave report is
[`runs/report_cards/oversight_frontier_v3_above_heavy/report_card.html`](../runs/report_cards/oversight_frontier_v3_above_heavy/report_card.html).
The pooled report is
[`runs/report_cards/oversight_frontier_synthesis/report_card.html`](../runs/report_cards/oversight_frontier_synthesis/report_card.html),
with machine-readable results in
[`data/oversight_frontier_synthesis_results.json`](../data/oversight_frontier_synthesis_results.json).
