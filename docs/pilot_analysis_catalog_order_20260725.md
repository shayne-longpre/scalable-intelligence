# Catalog Order Replay

## Question

Does changing the order in which Fable sees the same 50 anonymous answers
materially change its four-probe ranking?

## Design

The replay reused the exact four Sol-authored probes and archived candidate
answers from the best catalog condition. Candidate calls were not repeated.
Only answer presentation order changed, using seed `20260814`. Fable judged the
evidence from a fresh context with the original prompt and settings.

## Results

| Measure | Original | Shuffled replay |
| --- | ---: | ---: |
| Pairwise accuracy, 47 direct-score models | 86.7% | 83.7% |
| Kendall tau with external index | 0.734 | 0.674 |
| Spearman with external index | 0.885 | 0.827 |
| Top external model selected | Yes | Yes |

The two judged orders agreed at Kendall tau `0.814` and Spearman `0.928`.
Their top three were identical, four of the top five overlapped, and eight of
the top ten overlapped. Median absolute displacement was `1.5` ranks and mean
displacement was `3.44`.

The largest movement was Grok 4.5, from judged rank 8 to 29. GPT-5 moved from
20 to 6, Inkling from 30 to 18, and three other models moved nine places.
These are large individual uncertainties despite strong aggregate stability.

The replay cost `$6.57588` and made seven judge calls. Three unavailable source
answers remained explicitly unavailable; no candidate was called.

## Decision

Keep global comparison as the 50-model primary method. It preserves direct
comparison, fit within context, retained the top of the ladder, and avoided the
extra anchoring and merge assumptions introduced by panels.

Treat presentation order as a real uncertainty:

- use seeded shuffling and record the seed;
- include one prespecified alternate-order replay for key confirmatory results;
- report agreement and rank displacement across orders;
- use overlapping panels only if larger rosters no longer fit, repeated orders
  become substantially less stable, or a specific attention failure is found.

This is evidence of bounded order sensitivity, not order invariance.
