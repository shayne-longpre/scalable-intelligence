# Ceiling-Aware Probe Study

This study reused five completed nine-candidate panels. Each judge kept its
original five opening probes, wrote five new probes after being told candidates
might be substantially stronger than itself, and then produced a fresh
ten-probe ranking. Original candidate calls were replayed; only the additions
were purchased.

## Ranking Result

| Judge | External score | Five probes | Ten probes | Change | New-probe pair accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama 4 Maverick | 14.3 | 55.6% | 55.6% | 0.0 pp | 45.3% |
| Kimi K2 Thinking | 32.7 | 80.6% | 72.2% | -8.3 pp | 53.5% |
| DeepSeek V4 Pro | 44.3 | 61.1% | 66.7% | +5.6 pp | 58.9% |
| Grok 4.5 | 53.8 | 58.3% | 75.0% | +16.7 pp | 67.8% |
| GPT-5.6 Sol | 58.9 | 75.0% | 66.7% | -8.3 pp | 58.9% |

Mean ranking accuracy moved from 66.1% to 67.2%. Two judges improved, two
worsened, and one was unchanged. The 25 new probes reached 57.0% pair accuracy
as individual discriminators. The extension used 288 model calls and cost
$18.54.

Probe style changed markedly with judge capability. Llama favored broad
advisory scenarios; Kimi mixed communication, mathematics, language induction,
and spatial reasoning; DeepSeek used classic ciphers and logic puzzles; Grok
used proof-heavy mathematical puzzles; and Sol used advanced mathematics,
algorithms, economics, and physics. Sixteen of the 25 additions received a
quantitative-mathematics label. Harder and more formal did not reliably mean
more diagnostic.

## Author Solvability Audit

The accepted study catalog contains 147 unique author-probe pairs. Each author
assessed every probe in a separate context; 146 also produced a blind fresh
solution. A fixed Sol evaluator jointly scored each available solution and the
archived anonymous candidate answers. One Qwen probe is excluded from scored
rates because repeated provider failures left both its blind solution and
reference evaluation unavailable.

- 105 of 146 scored probes were reference-valid (71.9%).
- 23 of 146 (15.8%) were **beyond author**: the author scored 0-2 while a
  candidate with a higher external intelligence score received 3-4.
- Authors overpredicted their own score by 0.52 points on average on the
  anchored 0-4 scale; mean absolute error was 0.70.
- Authors labeled 101 probes as peer-level, 34 for stronger systems, eight
  mixed, and four for weaker systems.
- Three authors said a probe was not solvable. All three also judged the probe
  invalid; the fixed evaluator rated the two scored cases limited rather than
  valid. No author explicitly identified a valid probe it could not solve.

The ceiling-aware additions show the same gap between intention and outcome.
Authors described 22 of 25 as peer-level and 19 as fully solvable. Only two
were intended for stronger systems. Nevertheless, five became strict
beyond-author cases: three from Llama, one from DeepSeek, and one from Grok.

## Interpretation

The instruction created broader difficulty portfolios but did not reliably
produce intentional tests above the author's own ceiling. Extra probes also had
heterogeneous marginal value: Grok's additions materially improved its ranking,
while equally difficult-looking additions hurt Sol and Kimi. Probe count alone
is therefore a poor stopping rule; direct per-probe discrimination and ranking
stability should determine whether more evidence is useful.

The author audit does show a scalable-intelligence signal. Weaker and
mid-capability authors sometimes wrote valid probes that stronger candidates
answered better than the author itself. That is different from writing an
impossible or internally inconsistent task, and the analysis keeps those cases
separate.

These are descriptive pilot results. The external score is a noisy prior, the
fixed evaluator may have model-specific biases, candidate panels differ by
judge, and one fresh solution is not proof of a model's capability limit.
Author assessments used two prompt versions; the second changed only the
placement of required JSON fields after some models nested otherwise valid
answers.

## Artifacts

- [Ceiling-extension report](../runs/report_cards/probe_ceiling_extension_v1/report_card.html)
- [Probe self-study report](../runs/probe_self_study/accepted_probes_v1/report_card.html)
- [Combined probe-evolution report](../runs/report_cards/probe_evolution_accepted_v2/report_card.html)
- [Ceiling-extension data](../data/probe_ceiling_extension_results.json)
- [Probe self-study data](../data/probe_self_study_results.json)
- [Deduplicated probe catalog](../data/probe_catalog.json)
