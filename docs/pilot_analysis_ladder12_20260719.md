# Independent-Judge Ladder: Initial Pilot

## Design

Two judges independently designed six probes and ranked the same 12 anonymous
candidate models. Every probe was sent unchanged to every candidate in a fresh
context. Separate judgment branches saw only probes 1-2, 1-4, or 1-6. Candidate
answers were shared across branches, but evidence cards and rankings were
regenerated from the permitted prefix.

The external model catalog is a comparison prior, not verified ground truth.
Judge identities, candidate identities, and catalog ranks were hidden during
the experiment.

## Ranking Results

| Judge | Probes | Kendall tau | Spearman rho | Pairwise accuracy | Score R2 | Prior top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GPT-5.6 Sol | 2 | 0.67 | 0.80 | 0.83 | 0.58 | No |
| GPT-5.6 Sol | 4 | 0.85 | 0.95 | 0.92 | 0.74 | Yes |
| GPT-5.6 Sol | 6 | 0.82 | 0.91 | 0.91 | 0.75 | Yes |
| GLM-4.5-Air | 2 | 0.24 | 0.38 | 0.62 | 0.29 | No |
| GLM-4.5-Air | 4 | 0.15 | 0.27 | 0.58 | 0.22 | No |
| GLM-4.5-Air | 6 | 0.39 | 0.59 | 0.70 | 0.39 | Yes |

At six probes, both judges selected `P4` (Claude Fable 5), the prior's top
model. Their full rankings agreed at Kendall tau 0.52. Sol placed seven models
in their exact prior positions and got 60 of 66 pairwise comparisons right.
GLM got 46 of 66.

More evidence was not uniformly beneficial. Sol's ordinal agreement peaked at
four probes, although score R2 improved slightly at six. GLM deteriorated at
four and then improved materially at six. This is one run, so it establishes a
failure mode and a replication target, not an optimal probe count.

The judges differed sharply in score resolution. Sol assigned 12 distinct
scores at every budget. GLM used only four distinct scores at two probes, three
at four, and four at six; eight candidates shared a score of 85 in its final
ranking. Its evidence cards often used generic strengths and weaknesses, while
Sol identified concrete errors tied to requested constructions, proofs, and
calculations. Valid JSON was therefore not the limiting capability for GLM;
evidence discrimination was.

Both judges substantially overranked `P3` (GPT-OSS-120B) relative to the prior;
Sol ranked it third rather than seventh and GLM ranked it second. This may be a
judge error, a task-distribution effect favoring formal reasoning, or a weakness
in the external prior. It should not be labeled incorrect without replication
or independent answer grading.

## Probe Repertoire

| Judge | Probe | Audited question types |
| --- | ---: | --- |
| Sol | 1 | Math reasoning |
| Sol | 2 | Scientific and causal reasoning |
| Sol | 3 | Coding and algorithmic reasoning |
| Sol | 4 | Logic, consistency, and state tracking |
| Sol | 5 | Math, planning, and adaptive decision-making |
| Sol | 6 | Scientific and spatial reasoning |
| GLM | 1 | Abductive logic and contradiction handling |
| GLM | 2 | Counterfactual science and decision judgment |
| GLM | 3 | Scientific evaluation, argument, metacognition, and robustness |
| GLM | 4 | Verbal abstraction and cross-domain transfer |
| GLM | 5 | Fluid reasoning, logic, creative synthesis, and edge cases |
| GLM | 6 | Creative generation and resource planning |

All 12 probes were direct task probes because that behavior is required by this
protocol and should not be interpreted as emergent strategy. Two probes tested
cross-domain transfer, one explicitly tested metacognitive self-assessment, and
one used paradoxes as an edge-case test. The batteries were preauthored before
answers: two opening probes were followed by nine broad topic changes and one
same-area new angle. There was no adaptive follow-up in this baseline.

## Classification Audit

The initial lexical pass produced systematic false positives: *code* in
error-correcting code, *speed* in physics, *state* in "state precisely,"
*uncertainty* in causal identification, *algorithm* in a stock-model evaluation,
and *social* in social contagion. The taxonomy indicators were narrowed and
regression tests added. The audited labels above now match the substantive
content of all 12 probes.

Confidence differs by layer:

- **High:** Q/A linkage, role, judge isolation, probe budget, replay provenance,
  preplanned/adaptive status, rankings, and spend. These come from metadata.
- **Moderate to high for this pilot:** broad probe question types after complete
  manual review.
- **Moderate:** secondary multi-label boundaries such as fluid versus creative
  reasoning or argument evaluation versus scientific judgment.
- **Exploratory:** transcript-wide behavioral keyword signals such as evasion,
  signaling, deception, or calibration. These require a frozen rubric and
  labeled validation set before aggregate scientific claims.

## Operational Health

The final transcript contains all 144 Q/A pairs, 72 evidence cards, and six
rankings. One candidate answer required a bounded visible-output retry. Two GLM
rankings returned a string where a list was required and were repaired by the
same model. Current-code revalidation reports zero remaining findings.

The final resume made 116 paid calls and cost $1.09. Including replay sources
and failed attempts, the recorded lineage is at least 385 calls and $13.96. The
lineage is marked incomplete because the first failed run lacks a run summary;
its successful transcript costs are included, but any failed request cost may
be missing.

## Next Test

Freeze this protocol and taxonomy version, replicate the ladder with at least
one new randomization, then run selective follow-up on the strongest judge's
uncertain middle group. The adaptive probe should be common within the selected
comparison set and evaluated against the six-probe baseline. Separately grade a
sample of candidate answers to distinguish prior error from judge error,
especially for GPT-OSS-120B.
