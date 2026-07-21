# Adaptive Probe-Wave Stress Pilot

This pilot tested the proposed independent-judge loop at its initial maximum
schedule: four opening probes followed by four one-probe adaptive rounds. It
also treated every round as a virtual early stopping point.

Run artifacts: `runs/20260720T034603Z_adaptive_judge_waves_p4_round_stress`.
Human-facing report:
`runs/report_cards/adaptive_waves_pilot_20260720_final_v2/report_card.html`.

## Setup

- **Judge:** Claude Fable 5 (`J1`).
- **Candidates:** GPT-5.4 (`P1`), GPT-5.4 Mini (`P2`), Gemini 3.5 Flash (`P3`),
  and Claude Sonnet 4.6 (`P4`). Identities and external priors were hidden.
- **Schedule:** `[4, 1, 1, 1, 1]`.
- **Evidence:** 8 probes, 28 routed Q/A pairs, 8 direct probe comparisons, and
  5 cumulative judgments.
- **External comparison prior:** `P1 > P3 > P4 > P2`. This catalog ordering is
  a fallible research prior, not verified ground truth.

## Checkpoints

| Round | Cumulative probes | Judge ranking | Kendall tau | Pairwise accuracy | Confidence |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 4 | P1 > P4 > P2 > P3 | 0.33 | 0.67 | 0.60 |
| 2 | 5 | P1 > P4 > P2 > P3 | 0.33 | 0.67 | 0.65 |
| 3 | 6 | P1 > P4 > P2 > P3 | 0.33 | 0.67 | 0.68 |
| 4 | 7 | P1 > P4 > P3 > P2 | 0.67 | 0.83 | 0.60 |
| 5 | 8 | P1 > P4 > P3 > P2 | 0.67 | 0.83 | 0.62 |

The judge selected the prior's top model after the opening battery and never
changed its top choice. Rounds 2 and 3 did not alter the ordering. Round 4
corrected the `P2/P3` inversion; Round 5 added evidence without another rank
change. The final remaining disagreement was `P4/P3`.

This does not establish that seven probes are generally sufficient. It shows
that the checkpoint design exposes both diminishing returns and a late useful
correction. A stopping rule should be chosen without looking at the external
prior, then evaluated over repeated judges and rosters.

## Probe Evolution

The opening portfolio covered four distinct approaches:

1. An inconsistent assignment puzzle with perturbation and self-audit.
2. Hidden-rule induction, calibrated hypotheses, and discriminating experiments.
3. Causal flaw detection in an air-purifier study, followed by study repair and
   explanations for technical and public audiences.
4. Scheduling optimization, proof of optimality, sensitivity analysis, and a
   counterexample to standard lower bounds.

The follow-ups were genuinely evidence-conditioned:

- **Round 2:** a classifier briefing with embedded count and arithmetic errors,
  competing rules, a decisive test, and a claim ledger.
- **Round 3:** a hidden-label problem intended to separate logical restraint
  from overclaiming. The judge later marked the probe limited because its own
  epistemic categories overlapped.
- **Round 4:** the smallest partition of `{1,...,N}` balancing sums and squared
  sums, with complete minimality ledgers. This was difficult enough to change
  the middle ranking, although the comparison remained low-confidence.
- **Round 5:** greedy coin-system failures plus an intentionally underdetermined
  denomination question. It was informative but did not change the ranking.

Five comparisons were marked informative and three limited. The judge did not
blindly preserve bad evidence: it identified saturation, ambiguity, and cases
where an ordering rested on judgment rather than a checkable error. One observed
risk was repeated use of claim ledgers and premise traps. The baseline prompt
now allows replication when it tests generalization while warning against using
one preferred checklist as a proxy for general intelligence.

## Context And Reliability

Compression worked as intended. After the larger first-round merge, cumulative
judge prompts stayed roughly flat at 4.5k to 4.9k input tokens. Candidate
dossiers remained about 90 to 126 words each by Round 5 instead of replaying all
raw answers.

All structured probe comparisons and wave judgments parsed without JSON repair.
Offline revalidation found zero errors. Four truncated candidate completions and
two empty-visible-output completions recovered through bounded retries; the
final transcript contains complete answers for every routed request.

The stress attempt also exposed and fixed two infrastructure failures:

- malformed or truncated provider response bodies now use the configured HTTP
  retry policy;
- successful calls in a concurrent batch are committed before another call's
  failure is re-raised, so resumes do not repurchase completed work.

GLM 5.2 repeatedly exhausted its answer budget without visible output and was
replaced by GPT-5.4 Mini. This is recorded as a route/protocol compatibility
failure, not an intelligence result.

## Cost

The completed run added 47 model calls and approximately `$4.37` in
provider-reported cost. Including the failed and resumed attempts, the complete
replay lineage contains 86 calls and approximately `$6.00`. The final report
retains both incremental and lineage totals by model.

## Assessment

The design is ready for repeated research pilots. It preserves judge freedom,
supports several opening probes, compares candidates directly within each
probe, keeps cross-probe evidence compact, targets uncertain subsets, and yields
auditable early-stopping checkpoints.

The next methodological step is replication rather than another protocol
rewrite: run the same schedule with additional independent judges and rosters,
estimate how often later rounds improve or degrade rankings, then preregister a
gold-blind stopping policy for the main experiments.
