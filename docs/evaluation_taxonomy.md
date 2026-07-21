# Evaluation Taxonomy

This is a compact coding frame for machine-society experiments. It is not a
complete theory of intelligence and should not be treated as a validated
psychometric model. Its purpose is to help reviewers notice what models test,
how they test it, and which social or metacognitive behaviors emerge.

The machine-readable version lives in `data/evaluation_taxonomy.json`. Analysis
reports include keyword-based strategy and question-type signals. These are
candidate labels for review, not final classifications.

The taxonomy has two dimensions:

- **Evaluation strategy** - What the participant is doing as an evaluator.
- **Question type** - What capability a prompt or probe appears to elicit.

Boundary rule: question types describe the target capability; evaluation
strategies describe how the evaluator uses the probe. Multiple labels can apply
to the same turn.

Automated labels use conservative lexical indicators. Headline report-card
counts are computed from unique evaluator probes, not routed answers or JSON
field names. Standard protocol constraints such as candidate answer limits do
not make every probe an instruction-following test. Likewise, domain phrases
such as *physical speed*, *error-correcting code*, *training data*, and *state
precisely* are not classified as processing speed, programming, model
self-disclosure, or working-memory testing merely because they share a word.

Structural labels such as probe identity, speaker, evidence dependency,
preplanned versus adaptive generation, and Q/A linkage come from transcript
metadata and can be treated as deterministic. Semantic question and strategy
labels remain multi-label coding suggestions until audited. Publication-scale
claims should use a frozen rubric, a hand-labeled validation sample, and report
inter-coder or classifier agreement.

## Provenance

The taxonomy is grounded in four source families:

- Human cognitive assessment: [Wechsler/Pearson assessment families](https://www.pearsonassessments.com/),
  including verbal comprehension, working memory, processing speed, fluid
  reasoning, and visual-spatial reasoning.
- Educational and admissions assessment: [GRE General Test](https://www.ets.org/gre/test-takers/general-test/about/content-structure.html)
  and [PISA](https://www.oecd.org/en/about/programmes/pisa.html), including
  reading, quantitative reasoning, analytical writing, science, and creative
  thinking.
- AI benchmarks: [MMLU](https://arxiv.org/abs/2009.03300),
  [BIG-bench](https://arxiv.org/abs/2206.04615),
  [HELM](https://arxiv.org/abs/2211.09110),
  [GSM8K](https://arxiv.org/abs/2110.14168),
  [MATH](https://arxiv.org/abs/2103.03874),
  [GPQA](https://arxiv.org/abs/2311.12022),
  [HumanEval](https://arxiv.org/abs/2107.03374),
  [MBPP](https://arxiv.org/abs/2108.07732),
  [APPS](https://arxiv.org/abs/2105.09938),
  [SWE-bench](https://arxiv.org/abs/2310.06770),
  [LongBench](https://arxiv.org/abs/2308.14508),
  [IFEval](https://arxiv.org/abs/2311.07911), and
  [ToolBench](https://arxiv.org/abs/2307.16789).
- Observed machine-society behavior: criteria negotiation, evaluator evaluation,
  strategic signaling, performative evasion, consensus formation, and recursive
  self-assessment from pilot transcripts.

Source relation labels:

- **Close match** - The bucket closely tracks an established assessment or
  benchmark family.
- **Derived** - The bucket compresses several related families into one practical
  coding label.
- **Observed** - The bucket exists because it appeared in pilots or is central to
  this experimental design.

## Evaluation Strategies

- **Criteria Setting** - Defines or revises what intelligence should mean.
  Source: observed; derived from rubric design and construct validity.
- **Direct Task Probe** - Uses a concrete question, puzzle, or mini-test.
  Source: derived from human test items and AI benchmarks.
- **Adaptive Follow-Up** - Changes the next probe based on earlier evidence.
  Source: observed; derived from diagnostic interviewing.
- **Edge-Case Testing** - Searches for contradictions, boundary cases, or failure
  modes. Source: derived from critical reasoning, robustness, and safety testing.
- **Transfer Test** - Tests whether structure carries across domains or analogies.
  Source: close match to abstraction and analogy tests; derived from transfer
  learning concerns.
- **Self-Assessment** - Assesses one's own performance, limits, or standing.
  Source: derived from metacognition and calibration; observed.
- **Evaluator Evaluation** - Treats question quality or evaluation discipline as
  evidence of intelligence. Source: observed; central to the recursive design.
- **Uncertainty Calibration** - States confidence, ambiguity, or evidence limits.
  Source: close match to calibration metrics; derived from metacognitive
  monitoring.
- **Strategic Signaling** - Tries to persuade, frame, or demonstrate status.
  Source: observed; derived from signaling-game and social-evaluation concepts.
- **Performative Evasion** - Sounds sophisticated while avoiding the substantive
  task. Source: observed; derived from robustness and instruction-following
  failure modes.
- **Deception Or Gaming** - Bluffs, misleads, or manipulates criteria.
  Source: derived from strategic competition, deception, and gaming evaluations.
- **Consensus Formation** - Defers, converges, or adopts a shared hierarchy.
  Source: observed; derived from social reasoning and group-decision dynamics.

## Question Types

- **Verbal Abstraction** - Analogies, definitions, and conceptual comparisons.
  Source: close match to Wechsler Similarities/Vocabulary-style tasks.
- **Language Induction** - Infers grammar, morphology, syntax, or meaning from
  examples in an unfamiliar language. Source: close match to
  [MLAT](https://lltf.net/aptitude-tests/language-aptitude-tests/modern-language-aptitude-test-2/)
  grammatical sensitivity and inductive language learning; derived from
  few-shot rule-induction tasks in [BIG-bench](https://arxiv.org/abs/2206.04615)
  and the 50-model catalog pilot.
- **Knowledge Recall** - Factual, cultural, scientific, or domain knowledge.
  Source: close match to crystallized-knowledge tasks; AI analogue in MMLU.
- **Fluid Reasoning** - Novel patterns, hidden rules, and unfamiliar structure.
  Source: close match to matrix/fluid-reasoning tasks; AI analogue in BIG-bench
  and ARC-style abstraction.
- **Logic And Consistency** - Contradictions, paradoxes, hidden assumptions, and
  valid inference. Source: derived from critical reasoning, BBH-style reasoning,
  and pilot logic puzzles.
- **Math Reasoning** - Arithmetic, algebra, probability, optimization, or proof.
  Source: close match to GRE/Wechsler quantitative tasks; AI analogue in GSM8K
  and MATH.
- **Units And Dimensions** - Unit analysis, dimensional consistency, and
  equation repair. Source: derived from physics/engineering problem solving and
  pilot dimensional-analysis probes.
- **Scientific Reasoning** - Causal models, hypotheses, experiments, and
  evidence. Source: close match to PISA science; AI analogue in GPQA and science
  QA.
- **Source Verification** - Citation discipline, source reliability, retraction
  under uncertainty, and empirical-claim verification. Source: derived from
  research-literacy tasks, retrieval-grounded QA, and pilot source-discipline
  probes.
- **Coding** - Algorithms, debugging, data structures, and program synthesis.
  Source: close match to programming contests and CS exams; AI analogue in
  HumanEval, MBPP, and APPS.
- **Computer Systems** - Distributed protocols, concurrency, transactions,
  consistency, and crash recovery. Source: derived from computer-systems design
  exercises and coding or systems-evaluation tasks.
- **Software Engineering** - Codebase changes, tests, integration, and realistic
  repair. Source: derived from engineering work samples; AI analogue in
  SWE-bench.
- **State Tracking** - Keeps entities, order, memory, or changing state straight.
  Source: close match to working-memory tasks; AI analogue in multi-turn state
  tracking and needle-style tasks.
- **Speed And Efficiency** - Performs under time, token, brevity, or efficiency
  constraints. Source: close match to processing-speed tasks; AI analogue in
  latency and token-budget evaluation.
- **Spatial And Visual Reasoning** - Images, diagrams, rotations, geometry, or
  scene details. Source: close match to visual-spatial tests; AI analogue in
  multimodal reasoning benchmarks.
- **Reading And Argument** - Close reading, summarization, inference, and
  argument critique. Source: close match to GRE/PISA reading and analytical
  writing.
- **Creative Generation** - Original ideas, metaphors, stories, or divergent
  solutions. Source: close match to divergent-thinking and PISA creative-thinking
  tasks.
- **Planning And Strategy** - Goal-directed plans, tradeoffs, policy, and
  scenario analysis. Source: derived from executive-function and practical
  judgment tasks; AI analogue in agent planning.
- **Social And Moral Judgment** - Norms, ethics, stakeholders, interpersonal
  reasoning, and common sense. Source: derived from comprehension and situational
  judgment tasks; AI analogue in ethics, commonsense, and safety evaluations.
- **Philosophical Analysis** - Conceptual, epistemic, ontological, or normative
  analysis. Source: derived from analytical writing, critical reasoning,
  philosophy exams, MMLU philosophy, BIG-bench conceptual tasks, and pilot probes.
- **Calibration** - Confidence, uncertainty, self-limits, or belief updates.
  Source: close match to calibration and metacognitive-monitoring work.
- **Recursive Self-Critique** - Biases in one's own reasoning or correction
  method. Source: derived from cognitive reflection, self-correction, and pilot
  recursive probes.
- **Long-Context Synthesis** - Finds and integrates evidence across long
  transcripts or documents. Source: derived from extended reading tasks; AI
  analogue in LongBench and needle-style evaluations.
- **Instruction Following** - Exact schemas, constraints, word limits, and
  formatting. Source: close match to IFEval-style verifiable instruction
  following.
- **Robustness** - Misleading premises, traps, manipulation, unsafe
  requests, or distribution shift. Source: derived from critical thinking, HELM
  robustness, and safety evaluations.
- **Multilingual And Cultural Reasoning** - Translation, idioms, cultural
  context, and cross-lingual transfer. Source: derived from language assessment
  and multilingual AI benchmarks.
- **Tool Use** - Chooses or uses search, APIs, execution, verification, or
  environment actions. Source: derived from work-sample tasks and AI tool-use or
  agent benchmarks.

## Pilot Coverage So Far

The pilots already exercised several families:

- liar/truth-teller village puzzles: **Logic And Consistency**
- "tower stops the sun from rising" reframing: **Creative Generation**,
  **Transfer Test**, **Logic And Consistency**
- market path/scenario planning: **Planning And Strategy**
- pattern/music analogy probes: **Fluid Reasoning** and **Verbal Abstraction**
- recursive assumption and self-bias questions: **Calibration** and
  **Recursive Self-Critique**
- conceptual truth, autonomy, or meaning questions: **Philosophical Analysis**
- compact JSON/final judgment turns: **Instruction Following**
- the 12-model independent-judge ladder: **Math Reasoning**, **Scientific
  Reasoning**, **Coding**, **Logic And Consistency**, **State Tracking**,
  **Spatial And Visual Reasoning**, **Planning And Strategy**, **Verbal
  Abstraction**, **Fluid Reasoning**, **Creative Generation**, **Recursive
  Self-Critique**, and **Robustness**

## Working Hypotheses

Stronger models may show more adaptive follow-up, evaluator evaluation,
uncertainty calibration, transfer testing, and long-context synthesis. Weaker
models may rely more on familiar task templates, accept others' frames, or
produce generic self-reflection. Strategic settings may increase signaling,
consensus formation, criteria manipulation, and possibly deception. These are
hypotheses for measurement, not assumptions baked into the experiment.

## Reader-Facing Questions

- Do models discover intelligence through testing, negotiation, or status games?
- Who notices that the best evaluator may be the smartest participant?
- Do weaker models recognize when they have been surpassed?
- Do stronger models prove superiority through better answers, better questions,
  or better judgment about evidence?
- Does conversation reveal intelligence differently than benchmarks do?
