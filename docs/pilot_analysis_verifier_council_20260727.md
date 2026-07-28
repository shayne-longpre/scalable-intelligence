# Verifier-Oriented Probes And Independent Councils

## Question

Can a judge recognize models above its own capability more reliably if it
designs answers it can verify? Does aggregating three independent evaluators
improve the ranking? Do those interventions reinforce one another?

## Design

Four judges each evaluated the same nine anonymous candidates under two
five-probe batteries. The ordinary battery came from the matched oversight
study. In the verifier battery, only private probe-design guidance changed:
judges were asked to seek concrete, inspectable evidence they could assess even
when the best answer exceeded their own ability.

GPT-5.6 Sol, Llama 4 Maverick, and Qwen3.7 Max then independently ranked the
same archived probes and answers. They did not deliberate. Pairwise majority is
the council endpoint; Sol alone is the preregistered single-evaluator baseline.

## Main Results

| Battery | Author: stronger above self | Author: all pairs | Sol: all pairs | Council: all pairs | Council gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary | 60.0% | 63.9% | 64.6% | 70.1% | +5.6 pp |
| Verifier-oriented | 55.0% | 66.0% | 72.2% | 75.0% | +2.8 pp |

Verifier guidance did **not** improve the primary scalable-oversight outcome:
above-self recognition fell by 5 points. It slightly improved the authors'
overall ordering and improved Sol's ordering by 7.6 points. This suggests that
the new evidence was sometimes easier for a stronger outside evaluator to use,
without making self-evaluation more reliable.

The council result is cleaner. It improved on Sol in all four ordinary panels.
With verifier evidence it improved two panels, tied one, and worsened one. The
two interventions were partly redundant rather than complementary: the council
gain was 2.8 points smaller with verifier batteries.

## Panel Results

| Probe author | Battery | Author accuracy | Stronger above self | Sol accuracy | Council accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Grok 4.5 | Ordinary | 58.3% | 80% | 69.4% | 72.2% |
| Grok 4.5 | Verifier | 55.6% | 60% | 66.7% | 72.2% |
| DeepSeek V4 Pro | Ordinary | 61.1% | 60% | 63.9% | 69.4% |
| DeepSeek V4 Pro | Verifier | 72.2% | 20% | 75.0% | 75.0% |
| Kimi K2 Thinking | Ordinary | 80.6% | 20% | 61.1% | 72.2% |
| Kimi K2 Thinking | Verifier | 69.4% | 40% | 61.1% | 69.4% |
| Llama 4 Maverick | Ordinary | 55.6% | 80% | 63.9% | 66.7% |
| Llama 4 Maverick | Verifier | 66.7% | 100% | 86.1% | 83.3% |

The panel spread is the result, not a nuisance to average away. DeepSeek's
overall ordering improved while its above-self recognition collapsed. Llama's
verifier battery helped substantially, but adding two evaluators diluted Sol's
exceptionally strong ranking. Kimi's ordinary author ranking was strong even
though it placed only one of five stronger models above itself.

## What The Judges Asked

- **Grok** wrote five tightly constrained construction and proof tasks:
  distributed protocols, algebraic structures, optimal stopping, group testing,
  and hidden-state control. Sol judged only one fully informative and four
  limited.
- **DeepSeek** used familiar linguistic induction, string rewriting, logic-grid,
  experimental-design, and causal-paradox tasks.
- **Kimi** mixed game design, lock logic, chemical identification, resilient
  networks, and statistical testing. Strong candidates exposed flaws in some
  requested guarantees, making probe validity itself an intelligence signal.
- **Llama** favored broad counterfactual world-building, artifact analysis,
  causal explanation, research planning, and remote-station design. These were
  diverse but less mechanically checkable.

The guidance therefore changed the surface form of the batteries without
creating one uniform verification strategy. Probe authors marked 70% of
verifier probes informative, versus 75% of ordinary probes. Sol marked 70% in
both conditions.

## Reliability And Cost

Accepted source runs contain every expected answer. Provider-level failures
were isolated and replayed without rerunning available candidate
answers. One recovery attempt confused probe number with experimental round;
the repair builder now rejects requested rounds with no unavailable evidence.
The final analyzer also verifies that every council member judged the same
source run and that the declared evaluator matches the run configuration.

Reconstructing all accepted evidence lineages and evaluator judgments costs
$29.46 as reported by the providers: $17.68 for evidence lineages and $11.78
for council evaluations. Evaluator costs were $9.90 for Sol, $1.76 for Qwen,
and $0.12 for Llama. A discarded repair branch added $0.11 and is not included
in the accepted-artifact total.

## Interpretation

This four-panel pilot rejects a simple story that telling judges to write
verifiable questions solves above-level oversight. It provides better evidence
for independent aggregation, especially on ordinary batteries, but not for a
universal council advantage or positive interaction.

The external intelligence index is a noisy reference, nearby models are hard
to order, and authors, panels, providers, and probe styles remain entangled.
The next study should isolate one observed failure: validate probes before
routing them, or test diverse councils on held-out panels. It should not combine
both interventions without first identifying which mechanism it intends to
change.
