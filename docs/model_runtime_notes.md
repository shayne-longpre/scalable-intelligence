# Model Runtime Notes

These notes track provider/runtime behaviors that can distort experiment results
without being part of the intended social-evaluation signal.

## Reasoning Token Starvation

Some OpenRouter models can spend most or all of a completion budget on hidden or
separate reasoning tokens, leaving little or no visible `message.content`.

Observed in pilot replays:

The table records historical low-budget failures and should not be read as a
permanent exclusion list. The 50-model catalog pilot requalified routes on the
exact four probes, raised the completion ceiling to 40,000 tokens subject to
each route's published maximum, and used low-effort sensitivity retries only
for missing cells. It obtained 197 substantive answers out of 200; the three
remaining cells were recorded as unavailable evidence rather than scored as
incorrect. Exact request parameters remain attached to every answer.

| Model | Symptom | Effective adjustment |
| --- | --- | --- |
| `minimax/minimax-m3` | Empty visible answers after more than 6,000 hidden reasoning tokens on a fixed-output probe. | Exclude from fixed-budget comparisons until a specific route and visible-token budget pass the exact probe protocol. |
| `z-ai/glm-5.2` | Structured assessment turns and substantive candidate answers can truncate or return empty when reasoning dominates even a 6,000-token low-effort recovery. | Use `reasoning: {"effort": "none"}` only after qualifying the exact task; otherwise exclude the route from fixed-budget runs. |
| `deepseek/deepseek-v4-pro` and `deepseek/deepseek-v4-flash` | Several routes consumed thousands of reasoning tokens and returned no usable visible answer; other routes rejected reasoning-disabled requests. | Treat as protocol-incompatible for the current fixed-output ladder rather than interpreting missing text as low intelligence. |
| `google/gemini-3-flash-preview` | `effort: "low"` can still consume a 2,500-token completion almost entirely as reasoning, leaving a partial visible answer. | Use an explicit reasoning cap such as `reasoning: {"max_tokens": 512}` and leave separate headroom for visible text. |
| `qwen/qwen3.5-397b-a17b` | A pilot answer and its recovery remained pending beyond one five-minute request window. | Use a shorter provider timeout for pilots and exclude the endpoint from timed comparisons until latency is characterized. |
| `qwen/qwen3-235b-a22b` and `qwen/qwen3-30b-a3b` | Smoke tests succeeded, but substantive probe generation remained pending for many minutes. | Keep smoke and exact-protocol qualification separate; do not admit a model based on a one-line smoke response. |
| `qwen/qwen-2.5-72b-instruct` | The preferred route rate-limited and fallback reached a route that did not complete the chat request reliably. | Pin a qualified provider or replace the model; fallback is not neutral when provider implementations differ. |
| `openai/gpt-oss-120b` | Default routes devoted most of the completion to hidden reasoning despite a low normalized effort setting. | Pin the qualified Groq route and leave enough completion headroom for both reasoning and visible output. |
| `openai/gpt-5.4-mini` | Three probe-writing calls exhausted a 6,000-token medium-reasoning budget without visible text. | Keep bounded visible-output recovery; a 5,000-token low-reasoning retry produced complete probes. |
| `google/gemini-3.5-flash` | Lower completion budgets truncated substantive candidate answers in the close-roster pilot. | Use a qualified low-reasoning profile with 8,000 total output tokens for this protocol. |
| `anthropic/claude-sonnet-4.6` | Generic normalized reasoning settings did not reliably reserve enough visible output on the selected route. | Use an explicit reasoning budget below the total output budget; the qualified pilot profile used 2,000 reasoning tokens within 12,000 total tokens. |

OpenRouter documents reasoning tokens as output tokens and exposes a normalized
`reasoning` request object with `effort`, `max_tokens`, `exclude`, and `enabled`
controls. `exclude: true` hides returned reasoning text but does not stop the
model from using reasoning tokens, so it is not enough for this failure mode.
See the [OpenRouter reasoning-token documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

Default policy for short pilots:

```json
{
  "temperature": 0.75,
  "max_tokens": 1000,
  "reasoning": {
    "effort": "none"
  }
}
```

When only some phases need stricter control, put the override on the phase:

```json
{
  "name": "memory_update_1",
  "kind": "private_judgment",
  "prompt": "interaction_memory_update",
  "require_json": true,
  "model_params": {
    "max_tokens": 1000,
    "reasoning": {
      "effort": "none"
    }
  }
}
```

For intentionally reasoning-heavy conditions, raise `max_tokens` substantially
and analyze reasoning-token usage as part of the treatment. Do not mix
reasoning-heavy and reasoning-disabled participants in the same condition unless
that difference is the experimental manipulation.

## Runtime Guardrails

The rule-based monitor records:

- `empty_visible_output_after_reasoning` when a turn has no visible content but
  reports reasoning tokens.
- `reasoning_dominated_structured_output_failure` when a required JSON turn
  fails parsing while reasoning tokens dominate the completion budget.

Future transcripts also store `finish_reason` and `response_message_keys` in
turn metadata, which is enough to diagnose common provider response-shape issues
without storing full raw API responses.

The shared HTTP transport enforces a true total deadline around each request.
Provider retries are explicit (`request_retries`) and are disabled in the large
ladder config so a single slow route cannot silently multiply its deadline.
Candidate routes use a five-minute provider profile; global judge comparisons
use a separate fifteen-minute profile. Candidate answer and evidence-card
batches remain concurrent, but transcript commits preserve deterministic
protocol order.

Some routes reject a recovery override such as `reasoning: {"effort":"none"}`
even when the primary request is valid. The runner now treats a provider HTTP
400 or 422 on that override as a recoverable compatibility mismatch: it retries
once with the model's primary parameters and records
`rejected_recovery_override`. Other provider errors are not swallowed.
