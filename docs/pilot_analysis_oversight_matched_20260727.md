# Oversight Frontier: Matched-Panel Extension

## Question

Can models recognize anonymous candidates more capable than themselves, and
does recognition depend on the judge's capability or the candidate's lead?

## Design

The extension adds two independently sampled nine-candidate panels for each of
four judges that previously had one panel: Grok 4.5, DeepSeek V4 Pro, Kimi K2
Thinking, and Llama 4 Maverick. Every new panel contains anonymous self plus:

- two candidates less than two external-score points above the judge;
- one candidate two to five points above;
- two candidates at least five points above;
- one near, one moderately, and one substantially weaker candidate.

Each judge wrote five probes before seeing any answer, compared all candidates,
and then received one bounded opportunity to probe an uncertain subset. The
five-probe judgment is the primary endpoint; the adaptive judgment is
secondary. The manifest freezes the exact panels and settings in
[`studies/oversight_frontier_v4_matched_extension.json`](../studies/oversight_frontier_v4_matched_extension.json).

## Matched Extension

All eight accepted panels have complete candidate evidence.

| Judge | Panel | Five probes | After follow-up | Stronger recognized |
| --- | --- | ---: | ---: | ---: |
| Grok 4.5 | B | 58.3% | 58.3% | 4/5 |
| Grok 4.5 | C | 75.0% | 72.2% | 5/5 |
| DeepSeek V4 Pro | B | 61.1% | 63.9% | 3/5 |
| DeepSeek V4 Pro | C | 80.6% | 80.6% | 2/5 |
| Kimi K2 Thinking | B | 80.6% | 77.8% | 1/5 |
| Kimi K2 Thinking | C | 33.3% | 33.3% | 2/5 |
| Llama 4 Maverick | B | 55.6% | 52.8% | 4/5 |
| Llama 4 Maverick | C | 58.3% | 55.6% | 5/5 |

Across these panels, opening pairwise accuracy was 62.8% and judges recognized
26 of 40 stronger candidates. Kimi's 47-point opening swing between two matched
panels is the clearest warning against interpreting one panel as a stable model
trait.

## All Four Waves

The pooled study now contains 30 panels, ten judges, and 900 candidate pairs.

| Outcome | Five probes | After follow-up |
| --- | ---: | ---: |
| Pair ordering | 645/900 (71.7%) | 647/900 (71.9%) |
| Stronger candidate above judge | 77/118 (65.3%) | 76/118 (64.4%) |

The central superior-recognition estimate was stable: it was 51/78 (65.4%)
before this extension and is 77/118 (65.3%) after it. Overall pair accuracy
fell from 75.8% to 71.7% because the new matched panels deliberately include
many close and above-judge comparisons.

| Candidate lead over judge | Recognized after five probes | Rate |
| --- | ---: | ---: |
| Less than 2 points | 26/42 | 61.9% |
| 2 to 5 points | 16/28 | 57.1% |
| 5 to 10 points | 22/34 | 64.7% |
| At least 10 points | 13/14 | 92.9% |

Distance matters clearly only at the largest separation. The three smaller
bins overlap substantially, so the data do not support a sharp oversight
threshold.

The upper third of judges recognized 83-91% of stronger candidates in the
observed sub-ten-point bins. Lower and middle judges were much less consistent.
This is suggestive evidence that judge capability matters, not a causal
estimate: judge identity, provider, panel, and probe battery remain entangled,
and the cells contain few independent observations.

## Adaptation And Reliability

One follow-up improved ten panels, left eleven unchanged, and worsened nine.
Aggregate pairwise accuracy moved only 0.2 percentage points. The adaptive
stage should remain a reported extension, not be silently folded into the
primary result.

The eight-panel extension used 557 calls and $16.33 in provider-reported spend.
Five accepted runs reused exact probes and successful answers to repair or
resume specific missing stages. Four retained the original model settings.
DeepSeek-B preserved its high-reasoning opening judgment but used the
prespecified low-reasoning recovery profile for the adaptive stage after the
high-reasoning follow-up timed out. This does not affect the primary five-probe
endpoint. The pooled four-wave lineage records 2,075 calls and $95.01; seven
unavailable answers remain in older accepted panels, out of 1,345 candidate
answers. The new extension contributes none of them.

## Interpretation

Three findings survive the expanded sample:

1. Models often recognize a more capable system, but approximately one third
   of stronger candidates are still placed below anonymous self.
2. A very large capability lead is easier to recognize; smaller leads do not
   yet form a clean monotonic curve.
3. Judge capability may help, but battery and panel effects are large. More
   matched replication is necessary before assigning a stable oversight
   frontier to an individual model.

The eight-panel report is
[`runs/report_cards/oversight_frontier_v4_matched_extension/report_card.html`](../runs/report_cards/oversight_frontier_v4_matched_extension/report_card.html).
The pooled report and machine-readable results are
[`runs/report_cards/oversight_frontier_synthesis_matched/report_card.html`](../runs/report_cards/oversight_frontier_synthesis_matched/report_card.html)
and
[`data/oversight_frontier_synthesis_matched_results.json`](../data/oversight_frontier_synthesis_matched_results.json).
