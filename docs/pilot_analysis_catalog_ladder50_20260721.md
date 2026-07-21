# 50-Model Catalog Ladder Pilot

## Study

GPT-5.6 Sol and Claude Fable 5 independently evaluated the same 50 anonymous
models. Each judge authored four opening probes before seeing any answer, then
used two adaptive probes for a judge-selected set of at most ten candidates.
Every probe was sent unchanged to its comparison set. All exact prompts,
answers, presentation orders, comparisons, and usage records were archived.

The primary analysis covers the 47 candidates with directly reported
Artificial Analysis Intelligence Index scores. The three estimated weak
anchors are included only in the secondary 50-model analysis. External scores
are a useful prior, not ground truth, especially for closely spaced frontier
models and settings that do not exactly match the benchmarked reasoning mode.

## Main Results

| Judge | Probes | Kendall tau | Pairwise accuracy | Rank-score R2 |
| --- | ---: | ---: | ---: | ---: |
| Sol | 4 | 0.623 | 0.811 | 0.635 |
| Sol | 5 | 0.663 | 0.832 | 0.675 |
| Sol | 6 | 0.678 | 0.839 | 0.689 |
| Fable | 4 | 0.608 | 0.804 | 0.633 |
| Fable | 5 | 0.599 | 0.799 | 0.625 |
| Fable | 6 | 0.597 | 0.798 | 0.622 |

The judges' final rankings had Kendall tau 0.605 with one another and selected
different top models. Sol placed its anonymous instance first and Fable third;
Fable placed its anonymous instance fifth and Sol sixth. Self-placement is
descriptive because candidate and judge inference settings are not identical.

The clearest result is discrimination by capability gap. At six probes, Sol's
pairwise accuracy was 0.60 for candidates separated by less than two index
points, 0.66 for gaps of 2-5, 0.77 for gaps of 5-10, and 0.93 for gaps above
ten. Fable reached 0.49, 0.63, 0.67, and 0.91 in the same bins. Broad capability
differences are therefore much easier to recover than fine frontier ordering.

Adaptive evidence had judge-dependent value. Sol improved at both checkpoints.
Fable changed its ranking and became more confident, but agreement with the
external prior declined slightly. This does not by itself show that Fable's
updates were worse: the external prior is noisy, and each adaptive probe
covered only ten candidates. It does show that confidence, ranking change, and
external agreement must be reported separately.

## Probe Behavior

Sol opened with combinatorial experimental design, quantitative physical
reasoning, concurrent-program correctness, and causal inference. It then
targeted an uncertain cluster with a hidden-state planning problem and a
program-synthesis and verification task.

Fable opened with formal reachability, invented-language induction, causal
inference, and adversarial proof auditing. It followed with a stack-language
induction problem and an exact combinatorics problem. Both judges changed
domain as well as difficulty and required checkable constructions, error
analysis, or proofs. The taxonomy now distinguishes language induction from
ordinary multilingual competence so this behavior is not lost in aggregation.

## Reliability And Cost

Both opening batteries obtained 197 substantive answers out of 200. Remaining
route failures are stored as explicit missing evidence, never as low-capability
answers. Interrupted batches can be resumed from an append-only journal and
committed in deterministic transcript order. A replay can retry only missing
rounds, while the exact-answer archive supports new answer orders or panel
comparisons without repurchasing candidate evidence.

The recorded Fable lineage is complete: 235 model calls and $18.0002. The Sol
lineage records 261 calls and at least $17.1403, including failed and recovery
attempts; one temporary replay source was not retained, so this is a lower
bound. Direct judge calls account for $10.6104 for Fable and $6.0854 for Sol.

## Decisions

- Keep the `[4, 1, 1]` schedule for the next controlled study. Four probes are
  the baseline, five remain the preregistered primary endpoint, and six are a
  secondary marginal-evidence endpoint.
- Keep global per-probe comparison as the primary 50-model procedure. It
  completed without evident format or context failure. Do not introduce panels
  until an order-randomization replay measures material position instability.
- Run the shared-evidence cross-over next. Each judge should rank the other
  judge's frozen evidence, separating probe design from interpretation.
- Replay at least one frozen battery under a second answer order. If rankings
  are unstable, compare global judgment with overlapping anchored panels using
  the same archived answers.
- Treat close-score pair accuracy and the oversight frontier as primary
  scientific targets. Overall ladder accuracy is dominated by easy large-gap
  pairs.

## Artifacts

- Sol run: `runs/20260721T085927Z_catalog_ladder50_gpt_5_6_sol`
- Fable run: `runs/20260721T102640Z_catalog_ladder50_claude_fable_5`
- Sol report: `runs/report_cards/20260721T092518Z/report_card.html`
- Fable report: `runs/report_cards/20260721T103008Z/report_card.html`
- Combined report: `runs/report_cards/catalog_ladder50_sol_fable_20260721_v5/report_card.html`
- Predicted versus external capability: `docs/figures/catalog-ladder50/predicted-vs-external.svg`
- Accuracy by capability gap: `docs/figures/catalog-ladder50/discrimination-by-gap.svg`
- Accuracy by cumulative evidence: `docs/figures/catalog-ladder50/evidence-scaling.svg`

The figures can be regenerated with `scripts/plot_catalog_ladder.py` from a
combined report-card summary and model catalog.
