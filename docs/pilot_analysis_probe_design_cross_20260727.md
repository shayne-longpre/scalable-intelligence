# Probe Design Versus Evidence Interpretation

## Question

When candidate evidence is held fixed, how much ranking quality comes from the
model that designed the probes, and how much comes from the model that interprets
the answers?

## Design

Four models supplied archived five-probe batteries and one follow-up from the
matched oversight study: Grok 4.5, DeepSeek V4 Pro, Kimi K2 Thinking, and Llama
4 Maverick. GPT-5.6 Sol and Llama 4 Maverick independently re-evaluated each
battery.

Within a battery, both evaluators received the exact same anonymous candidate
answers in the same seeded order. No candidate was called again. The
five-opening-probe ranking is primary; the ranking after the archived follow-up
is secondary.

The source author's original judgment is retained as an ecological baseline.
It used the source run's presentation order, so it is not part of the strict
same-order comparison.

## Results

| Probe author | Source author, 5 | Sol, 5 | Llama, 5 | Sol, final | Llama, final |
| --- | ---: | ---: | ---: | ---: | ---: |
| Grok 4.5 | 58.3% | 69.4% | 77.8% | 72.2% | 75.0% |
| DeepSeek V4 Pro | 61.1% | 63.9% | 50.0% | 55.6% | 63.9% |
| Kimi K2 Thinking | 80.6% | 61.1% | 66.7% | 61.1% | 63.9% |
| Llama 4 Maverick | 55.6% | 63.9% | 58.3% | 72.2% | 52.8% |
| **Mean** | **63.9%** | **64.6%** | **63.2%** | **65.3%** | **63.9%** |

The main findings are:

- **Interpretation quality is conditional on the test.** Sol led Llama on the
  DeepSeek and Llama batteries after five probes; Llama led on the Grok and Kimi
  batteries.
- **Battery choice mattered more than the aggregate evaluator gap.** Averaging
  the two crossed evaluators, five-probe accuracy ranged from 56.9% on the
  DeepSeek battery to 73.6% on the Grok battery. Sol's overall lead over Llama
  was only 1.4 percentage points.
- **More evidence was not reliably better.** The follow-up improved three of
  eight evaluator-battery cells, worsened four, and left one unchanged. Mean
  accuracy increased by 0.7 points.
- **Test authors were not consistently the best readers of their tests.** Both
  crossed evaluators exceeded Grok's source judgment, while both trailed Kimi's.
  This comparison is descriptive because the source-author answer order differed.

These results argue against treating either probe quality or judge quality as a
single model-level scalar. The observed ranking is produced by an interaction
among the battery, candidate panel, evaluator, and follow-up.

## Reliability And Cost

All eight accepted cells used complete archived evidence. The crossed control
made 67 evaluator calls and cost $5.155864 in provider-reported usage. Two
earlier Grok-C exploratory replays were excluded after the source selection was
returned to the prespecified Grok-B panel; they made 22 calls and cost
$1.889314.

The report card is
[`runs/report_cards/oversight_probe_design_cross_v1/report_card.html`](../runs/report_cards/oversight_probe_design_cross_v1/report_card.html).
Machine-readable results are
[`data/oversight_probe_design_cross_v1_results.json`](../data/oversight_probe_design_cross_v1_results.json).

## Limits

There is one battery per probe author and one candidate panel per battery.
Battery-level differences therefore do not identify general probe-writing
ability. The external intelligence index is also an uncertain reference rather
than ground truth, especially for close model pairs. Replication should cross
multiple independently authored batteries with multiple evaluators before
estimating author or evaluator effects.
