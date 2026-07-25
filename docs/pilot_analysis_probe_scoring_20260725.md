# Opening-Probe Answer Scoring

## Question

How difficult were the judges' opening probes, and do fixed answer-quality
scores recover the external capability ladder?

## Design

Sol and Fable independently scored the archived answers to all eight opening
probes. Each scoring call showed one judge every available anonymous answer to
one probe and requested:

- a short probe-specific correctness rubric;
- an anchored integer score from 0 to 4;
- a short evidence-grounded assessment and error tags.

The scale measures answer quality on that probe, not general intelligence.
Judges were told not to reward style, verbosity, or familiar phrasing. The
same seeded answer order was used for both judges.

## Results

| Probe | Mean score | Score ≥3 | Score/index Spearman | Pairwise alignment |
| --- | ---: | ---: | ---: | ---: |
| Sol: pooled tests | 2.15 | 38% | 0.707 | 75% |
| Sol: mechanics | 3.25 | 80% | 0.599 | 68% |
| Sol: concurrency | 2.81 | 68% | 0.730 | 77% |
| Sol: causal identification | 3.07 | 74% | 0.793 | 75% |
| Fable: reachability | 3.32 | 82% | 0.561 | 68% |
| Fable: invented language | 2.66 | 56% | 0.705 | 74% |
| Fable: causal choice | 3.01 | 78% | 0.627 | 71% |
| Fable: proof audit | 2.92 | 80% | 0.656 | 72% |

Mean score across all available probes correlated `0.874` with the external
index and ordered `84.8%` of direct-score model pairs correctly. This is close
to the best holistic four-probe ranking (`86.7%`) despite using a generic fixed
scale and deterministic averaging.

The pooled-test probe was hardest and the concurrency probe was most
discriminative by pairwise alignment. Reachability was easiest and least
discriminative. Difficulty and discrimination are therefore distinct.

Seven probes have scores from both judges. Six had mean absolute disagreement
between `0.14` and `0.38` points. On Fable's causal-choice probe, disagreement
was `1.00`: Sol required explicit causal exchangeability and transportability
assumptions, while Fable often accepted numerically correct observational
comparisons. This is a substantive rubric difference, not random formatting
noise.

## Reliability

Fifteen of sixteen judge-probe jobs completed. Four initial Fable calls exhausted
their visible-output budget or returned empty provider responses; evidence-
complete low-reasoning recovery completed three. Fable repeatedly returned an
empty, zero-cost response for Sol's pooled-test probe, so that column currently
uses Sol alone. Total reported scoring spend, including failed attempts, was
`$21.9874025`.

The runner now journals each job, resumes only missing work, includes full
evidence in repair requests, applies a per-judge recovery profile, and counts
failed-attempt spend. Raw attempts remain in the journal; the analysis summary
contains only compact rubrics, scores, and assessments.
The sanitized, versioned summary used by the publication site is
`data/catalog_ladder50_probe_scores.json`.
