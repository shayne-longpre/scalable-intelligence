# Adaptive Judge Protocol

This study asks one AI judge to rank anonymous candidate models from probes the
judge invents. Multiple judges can run independently over the same roster. They
do not communicate or share evidence.

## The Loop

1. **Ask.** The judge writes all probes for the current round before seeing any
   answers from that round.
2. **Answer.** Each probe is sent unchanged and in a fresh context to every
   candidate selected for that round.
3. **Compare.** For one probe at a time, the judge sees every candidate answer
   side by side. It records a within-probe ordering and compact summaries that
   retain concrete reasoning, implementation details, strengths, errors, and
   uncertainty.
4. **Merge.** After all probes in the round are compared, the judge combines
   them with its previous cumulative judgment. It updates every candidate's
   evidence dossier, the full ranking, and unresolved comparisons.
5. **Adapt.** The judge reviews the full roster before choosing the next target
   set and probe. It may deepen a close comparison, test whether a weakness
   generalizes, broaden coverage, or replace a bad test.

Candidate identities and external rankings are hidden. Candidates never see one
another's answers. The judge is warned that a candidate may be more capable
than the judge and that disagreement is not automatically an error.
Repeated formats are allowed when they test whether a signal generalizes, but
the judge is warned not to confuse compliance with one preferred rubric or
checklist with general intelligence.

The probe-writing instruction also asks the judge to check answerability,
internal consistency, difficulty, and scope. A limited or invalid previous
probe should normally trigger a substantial difficulty increase or capability
change. The candidate answer limit is a real design constraint, so a probe
should not demand more proof obligations than a careful answer can complete.

Every new run stores the exact effective prompt library with a version and
content hash. A saved config therefore remains interpretable after the default
prompts evolve.

## Scale Controls

`probe_schedule` is the main control. `[4, 1, 1]` means four opening probes,
then one probe in each of two adaptive rounds. The schedule determines the
number of rounds; every round ends with a complete ranking checkpoint.

`adaptive_targeting` controls later routing:

- `judge_selected` sends a later common probe to the judge's uncertain subset.
- `all` sends every later probe to the complete roster.

`max_adaptive_candidates` caps a judge-selected subset. Participant and judge
lists set the numbers of candidates and independent judges. Prompt IDs and
prompt overrides can change any stage without changing routing code.

## Why Compare Before Merging

Candidates are compared directly on each probe rather than assigned isolated
ability scores. This preserves domain differences: one candidate can lead on a
coding probe and trail on a philosophical or scientific probe. The later merge
can weigh those differences without flattening them too early.

Comparison summaries are shorter than raw answers but must preserve the facts a
later ranking needs. The cumulative dossiers are shorter again. Every layer
keeps source turn IDs, so a researcher can trace a conclusion back to the exact
probe and answer.

## Early Stopping

A maximum-length run supports several virtual stopping points. For a schedule
of `[4, 1, 1, 1, 1]`, compare the rankings after 4, 5, 6, 7, and 8 cumulative
probes. This reveals when rankings stabilize, whether later probes correct or
degrade the ordering, and when adaptive questioning stops adding useful
evidence.

Current pilot guidance is `[4, 1, 1, 1]` for close four-candidate rosters and
`[4, 1]` or `[4, 1, 1]` for broad rosters. Keep `[4, 1, 1, 1, 1]` for stress
tests and stopping-policy development. This is a configurable default, not a
claim that one schedule is optimal across judges and capability gaps.

## Decision Trace

Each adaptive probe is linked to the prior cumulative judgment, requested and
actual target sets, uncertain pairs, planned strategy, current per-probe
comparison, and resulting ranking. The analysis layer reports:

- whether routing matched the judge's requested targets;
- whether the targets covered the judge's declared uncertain pairs;
- which candidates were retained, added, or dropped;
- whether the probe broadened, deepened, or repeated an earlier area;
- all matched question types, not only one primary label;
- judge-reported probe validity and deterministic ranking change; and
- the change in cumulative confidence.

Probe validity is intentionally separate from confidence. `informative` means
the answers exposed a material capability difference; `limited` includes
saturation, style-only differences, or ambiguity; `invalid` means the probe
cannot support a sound comparison. These labels are judge reports and remain
auditable rather than being treated as verified ground truth.

Replay provenance uses semantic stream identities such as probe, candidate,
and judgment relationships. Local turn numbers may change when a partial run
is resumed, so they are source references rather than cross-run identities.

## Fixed-Battery Control

The older fixed-battery protocol remains available for reproducibility and
probe-count ablations. It creates candidate-specific evidence cards and may ask
for scores. The adaptive protocol above is the default for new studies because
it uses direct comparison, supports changing probes, and preserves a clean
round-by-round belief trajectory.
