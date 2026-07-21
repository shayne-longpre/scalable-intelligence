# Adaptive Judge-Quality Pilot

## Question

Can a strong and a medium model independently rank the same close frontier
roster, and how do their probes, adaptation, confidence, and accuracy change as
evidence accumulates?

## Frozen Design

- Judges: GPT-5.6 Sol and GPT-5.4 Mini, isolated from one another.
- Candidates: Claude Fable 5, GPT-5.4, Gemini 3.5 Flash, and Claude Sonnet 4.6.
- External prior: Fable 5 > GPT-5.4 > Gemini 3.5 Flash > Sonnet 4.6.
- Schedule: four opening probes, then three single adaptive probes (`[4,1,1,1]`).
- Routing: every probe is unchanged across its selected candidates; every
  candidate receives a fresh context.
- Replication: three anonymous-ID permutations with frozen prompts and runtime
  settings.

The judges authored different probes, so this measures complete evaluator
performance: test design, answer interpretation, adaptation, and ranking. It
does not isolate judgment quality on shared evidence.

## Accepted Runs

| Run | Calls | Reported cost | Turns | Final judge tau |
| --- | ---: | ---: | ---: | ---: |
| r1 | 92 | $6.367568 | 85 | 0.67 |
| r2 | 92 | $6.147476 | 83 | 0.67 |
| r3 | 88 | $4.743829 | 83 | 0.00 |
| **Total / mean** | **272** | **$17.258874** | **251** | **0.44** |

## Accuracy By Evidence Budget

Accuracy is against the external prior, whose adjacent frontier ordering is
uncertain.

| Judge | Probes | Pairwise accuracy | Kendall tau | Top-1 rate | Confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-5.4 Mini | 4 | 0.89 | 0.78 | 0.67 | 0.72 |
| GPT-5.4 Mini | 5 | **0.94** | **0.89** | 1.00 | 0.71 |
| GPT-5.4 Mini | 6 | 0.83 | 0.67 | 0.67 | 0.76 |
| GPT-5.4 Mini | 7 | 0.89 | 0.78 | 1.00 | 0.78 |
| GPT-5.6 Sol | 4 | **0.89** | **0.78** | 0.33 | 0.92 |
| GPT-5.6 Sol | 5 | 0.83 | 0.67 | 0.33 | 0.92 |
| GPT-5.6 Sol | 6 | **0.89** | **0.78** | 0.67 | 0.91 |
| GPT-5.6 Sol | 7 | 0.83 | 0.67 | 0.33 | 0.91 |

More evidence did not monotonically improve either judge. Mini peaked after the
first adaptive probe; Sol tied its best aggregate pairwise accuracy at four and
six probes. Confidence did not track prior agreement: Sol remained near 0.92
even when later rankings moved away from the prior.

## Probe Quality And Adaptation

| Judge | Informative | Limited | Invalid | Mean adaptive target | Rank changes | Direct judge cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.4 Mini | 11 | 10 | 0 | 3.00 | 4 | $0.995034 |
| GPT-5.6 Sol | 18 | 3 | 0 | 2.22 | 3 | $6.556625 |

Sol's probes were much more often diagnostic and its adaptive rounds focused on
smaller uncertain subsets. Mini was cheaper and sometimes equally or more
accurate against the prior, but half its probes were limited by saturation or
weak differentiation. This is the clearest replicated judge-quality difference
in the study.

All 18 adaptive routing decisions matched the requested candidate IDs and
covered the uncertain pair or pairs named by the judge. Seven follow-ups changed
a ranking; the rest corroborated or failed to resolve the current order. Later
probes showed both intended dynamics: topical broadening after weak initial
coverage and deeper same-area challenges after a specific failure.

## What They Tested

Math dominated: 29 of 42 probes received a Math Reasoning label. Science and
Planning each appeared on eight probes, Coding on seven, with smaller numbers of
systems, logic, fluid reasoning, state tracking, reading, spatial, and
robustness probes. Labels are multi-label, so totals exceed 42.

Sol tended to use harder causal inference, concurrency, formal construction,
and algorithmic edge-case probes. Mini used more conventional puzzles and
state traces. Neither judge needed domain suggestions in the live prompt to
produce broad tasks, but both favored problems with verifiable answers. That
bias is itself an experimental result; adding example domains now would erase
it and is not justified by these pilots.

## Runtime Reliability

The accepted runs completed with no audit warnings or errors. Bounded recovery
handled 14 empty-visible-output responses, four truncated responses, and three
malformed structured outputs. Most empty-output recoveries came from the medium
judge. These recoveries remain visible in the audit rather than being silently
discarded.

Four attempts were excluded before the final three-run batch:

| Failure | Calls | Reported cost | Resolution |
| --- | ---: | ---: | --- |
| HTTP-200 body containing a provider 500 error | 3 | $0.330775 | Detect and retry embedded provider errors. |
| JSON repair changed the fixed judge identity | 93 | $5.187735 | Make stage identity fields non-overridable during repair. |
| Candidate route content-filtered one common probe | 23 | $1.768537 | Exclude rather than rephrase for one candidate or score refusal as low intelligence. |
| Checkpoint confirmed the same filter | 2 | $0.000000 | Preserve the failure as route-compatibility evidence. |

Excluded attempts total 121 calls and $7.287047. They are operational failures,
not candidate intelligence observations.

## Analysis Refinements

The post-hoc taxonomy moved from frozen version `2026-07-20.4` to audited
version `2026-07-21.5`. Real pilot examples removed ambiguous lexical matches:
coding theory no longer implies programming, mini-language no longer implies
multilingual ability, executing a paper trace no longer implies tool use, and
"exactly once" in a combinatorics proof no longer implies distributed systems.
Positive coverage was added for counting proofs, modular mathematics, mechanics,
constraint satisfaction, and symbolic state traces. Regression tests preserve
both precision and recall.

The report card now aggregates each judge condition across anonymous-ID
permutations, with accuracy trajectories, probe validity, adaptive target size,
ranking churn, calls, and spend. Model priors are deduplicated by provider model
ID.

## Decision

Keep four opening probes. Use five cumulative probes as the primary endpoint for
the next small-roster study, because it was the best aggregate checkpoint here
and limits cost. Preserve rounds six and seven as explicitly adaptive extensions
when the judge still names an unresolved pair. Do not add automatic stopping
yet: three close-roster replications are insufficient to validate a stopping
rule.

Do not change the substantive judge or candidate prompts from this evidence.
The observed math/verifiability preference and judge differences are outcomes,
not prompt defects. The next clean comparison should hold evidence constant
across judges, then test whether these conclusions survive uncertainty in the
external prior.
