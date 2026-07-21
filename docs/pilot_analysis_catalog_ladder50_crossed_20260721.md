# 50-Model Shared-Evidence Cross-Over

## Design

Sol and Fable each first authored and judged an independent six-probe battery
over the same 50 anonymous candidates. The crossed control then swapped only
the judge:

| Evidence | Original judge | Crossed judge |
| --- | --- | --- |
| Sol probes and answers | Sol | Fable |
| Fable probes and answers | Fable | Sol |

The probe text, candidate answers, anonymous IDs, answer order, unavailable
answers, and adaptive target sets were frozen within each evidence condition.
The crossed judges independently wrote answer comparisons and cumulative
rankings. No candidate was called again. Results below use the 47 candidates
with directly reported external Intelligence Index scores.

## Results

| Evidence | Judge | 4 probes | 5 probes | 6 probes | Final Kendall tau | Final rank-score R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Sol | Sol | 81.1% | 83.2% | 83.9% | 0.678 | 0.689 |
| Sol | Fable | **86.7%** | **85.8%** | **85.2%** | **0.704** | **0.716** |
| Fable | Fable | 80.4% | 79.9% | 79.8% | 0.597 | 0.622 |
| Fable | Sol | 80.9% | 82.4% | 82.1% | 0.641 | 0.674 |

The strongest checkpoint was Fable's first ranking of Sol-authored evidence:
86.7% of candidate pairs were ordered consistently with the external index.
Sol also ranked Fable-authored evidence better than Fable did at every
checkpoint, although by a smaller margin.

Judges agreed substantially when they saw identical evidence. On Sol evidence,
their ranking agreement rose from Kendall tau 0.758 after four probes to 0.837
after six. On Fable evidence it remained between 0.767 and 0.780. By contrast,
the same judge's final rankings across the two evidence batteries agreed less:
0.667 for Sol and 0.602 for Fable. In this single cross-over, the chosen evidence
changed rankings more than swapping the interpreter of fixed evidence.

## What The Probes Tested

Sol's opening battery emphasized exact, technical stress tests: noisy pooled
testing and optimal experiment design, variable-mass mechanics, concurrent
algorithm correctness, and causal identification. Its adaptive probes then
tested planning under partial observability and a minimal streaming automaton
with executable pseudocode and proof obligations.

Fable's opening battery emphasized epistemic range: arithmetic reachability and
ill-posed questions, induction of an invented language while recognizing
underdetermination, causal judgment under Simpson's paradox, and adversarial
proof auditing. Its follow-ups tested induction of hidden program semantics and
a difficult combinatorial existence proof.

Both batteries were broad and substantive. The Sol battery nevertheless
produced more externally aligned rankings under both judges in this run. That
could reflect more diagnostic problems, better coverage of capabilities in the
external index, or ordinary judging and presentation variance; one realization
cannot distinguish these explanations.

## Adaptation

Sol improved as evidence accumulated on both batteries: by 2.8 percentage
points on its own evidence and 1.2 points on Fable's. Fable declined by 1.5
points on Sol evidence and 0.6 points on its own. More probes therefore did not
provide a monotonic benefit. The fifth probe remains the preregistered primary
endpoint; the sixth is a marginal-evidence diagnostic.

The crossed adaptive rounds intentionally retained the evidence author's target
set. This is necessary to hold evidence fixed, but it means a crossed judge
could not redirect the frozen follow-up toward its own uncertain candidates.
The result estimates interpretation of a fixed adaptive trajectory, not the
quality of a fully judge-specific adaptive policy.

## Reliability And Spend

Every accepted crossed transcript contains 6 probes, 220 replayed candidate
answers, 6 comparisons, and 3 cumulative rankings. Source target counts were
exactly 50, 10, and 10. All replayed question and answer entries retain source
provenance.

An orchestration bug discovered during the first crossed attempt allowed a
fresh interim ranking to disable later probe replay. The run was stopped before
an adaptive probe was generated. Probe and ranking replay state are now
independent, with a multi-round regression test covering this failure. The four
valid opening comparisons and ranking from each stopped attempt were reused.

The crossed judgments added nine judge calls per model and no candidate calls.
Provider-reported incremental spend was $9.5527 for Fable and $4.0583 for Sol,
including the reused opening work and resumed adaptive work. Report-card lineage
also includes the original evidence-generation runs.

## Interpretation

The clean conclusion is not that one model is the better judge. It is that
probe authorship and evidence interpretation are separable, measurable sources
of performance. The surprising best cell, Fable interpreting Sol's evidence,
also shows why self-authored batteries alone confound those sources.

The next experiment should replay at least one frozen battery under multiple
seeded answer orders, with no new candidate calls. If the ranking is stable,
move to the oversight-frontier study. If it is materially order-sensitive,
compare the global 50-answer judgment against overlapping anchored panels.

## Artifacts

- Crossed report: `runs/report_cards/catalog_ladder50_crossed_20260721_v1/report_card.html`
- Fable judging Sol evidence: `runs/20260721T224736Z_catalog_ladder50_cross_fable_judges_sol_evidence_complete`
- Sol judging Fable evidence: `runs/20260721T224737Z_catalog_ladder50_cross_sol_judges_fable_evidence_complete`
- Crossed accuracy figure: `docs/figures/catalog-ladder50/crossed-judge-accuracy.svg`

