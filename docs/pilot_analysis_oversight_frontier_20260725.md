# Oversight Frontier Pilot

## Question

Can an anonymous AI judge recognize and rank candidate models that are more
capable than the judge itself?

Six independent judges, from six organizations and spanning external capability
scores from 15.9 to 58.9, each evaluated seven candidates around their own
level. The judge was also present as an anonymous candidate. Each judge authored
five common opening probes, compared all answers probe by probe, formed a
ranking, selected up to four uncertain candidates, and asked one common adaptive
follow-up.

The external catalog is a noisy reference rather than ground truth. Close pairs
are therefore reported separately.

## Results

| Judge | External score | Opening pair accuracy | Final pair accuracy | Stronger above self | Weaker below self |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sol 5.6 | 58.9 | 66.7% | 71.4% | 0/1 | 5/5 |
| Gemini 3.5 Flash | 50.2 | 76.2% | 81.0% | 3/3 | 1/3 |
| Qwen 3.7 Max | 46.0 | 61.9% | 52.4% | 3/3 | 0/3 |
| MiniMax M2.7 | 38.1 | 81.0% | 81.0% | 3/3 | 1/3 |
| Claude Haiku 4.5 | 29.6 | 52.4% | 57.1% | 1/3 | 2/3 |
| Mistral Large 3 | 15.9 | 90.5% | 81.0% | 3/3 | 1/3 |

The aggregate final ranking ordered 89 of 126 candidate pairs correctly
(70.6%). Accuracy depended strongly on the external-score gap:

| Score gap | Correct pairs | Accuracy |
| --- | ---: | ---: |
| Less than 2 | 9/22 | 40.9% |
| 2 to 5 | 28/40 | 70.0% |
| 5 to 10 | 31/42 | 73.8% |
| More than 10 | 21/22 | 95.5% |

The first result is therefore not a clean monotonic "oversight frontier."
Large capability differences were easy to recognize, including for modest
judges, while near-ties were effectively unresolved. Judge quality alone did
not explain the condition-level result because each judge saw a different local
panel with a different difficulty distribution.

Judges placed 13 of 16 externally stronger candidates above their anonymous
self. That asymmetric statistic is too generous by itself: Qwen placed every
stronger candidate above itself, but also every weaker candidate. Across both
directions, judges made 23 of 36 self-relative comparisons correctly.

## Adaptation

One adaptive probe improved pairwise accuracy for three judges, left one
unchanged, and worsened two. Aggregate accuracy decreased from 71.4% after the
five-probe opening to 70.6% after adaptation. The follow-up mechanism worked
operationally, but one follow-up should not be assumed to improve a ranking.
Future studies should retain every checkpoint and treat stopping as an empirical
choice.

The judges used a broad repertoire: quantitative and scientific reasoning,
coding and systems design, argument analysis, linguistic induction, planning,
and social judgment. Probe diversity did not guarantee validity or difficulty.
The human audit found both productive and harmful false-premise tests:

- Gemini asserted that a game had no pure equilibrium; candidates that
  challenged the premise were correctly rewarded.
- Mistral asked for an impossible doubling of approvals under the stated
  rejection rate, then penalized a candidate for identifying the contradiction.
- Qwen posed underspecified ecology and finite-state inference tasks where the
  requested guarantees or unique answer did not follow from the evidence.
- MiniMax included a valid but low-ceiling hat puzzle, while Haiku used a
  thoughtful but weakly verifiable value-pluralism prompt.

Probe validity must therefore be audited independently from the judge's own
confidence or ranking.

## Reliability

The study used 357 recorded model calls across all attempts and cost $19.5683
according to provider-reported usage. Four malformed structured judge outputs
were repaired by the same model under a bounded recovery prompt. One MiniMax
attempt returned no usable text and was rerun. Two candidate answers remained
unavailable after bounded retries; both are explicit in the report rather than
silently dropped.

The reusable analysis separates paid model failures from local setup failures,
selects a repair run only when it reduces missing evidence, aggregates all
attempt costs, and preserves source run paths. Exact-answer order replays remain
the next robustness check; regenerating probes or answers would be a new
replication instead.

## Interpretation

This pilot supports three provisional conclusions:

1. AI judges can often recognize models above their own external capability,
   especially when the capability gap is substantial.
2. Fine-grained ranking is much less reliable than broad separation, and close
   external scores should not be treated as strict ground truth.
3. Better interrogation requires both hard questions and the judgment to
   recognize valid objections. A flawed test can invert the intelligence signal.

These findings motivate a confirmatory study with repeated panels and exact
evidence replays. They do not yet establish a universal capability threshold or
a monotonic relationship between judge intelligence and evaluation accuracy.
