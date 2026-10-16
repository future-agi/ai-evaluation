---
name: agent-modes
description: The ways an agent can be connected (code repository, uploaded code, hosted platform agent, phone number, provider import) and what each can and cannot be tested on.
---

# How an agent is connected, and what that allows

| Mode | What the harness has | World | How sub-goals are settled |
|---|---|---|---|
| Code repository or uploaded code | The agent's own source, run inside the sandbox | Built from the contract, or the agent's own service provisioned | Mostly checks on world state and tool calls; judged only for words and manner |
| Hosted platform agent (connect-only) | The agent's definition and prompt; calls go to the live agent where it runs | Empty: its data lives on that platform | Every sub-goal judged from the conversation |
| Phone number only (connect-only) | A number to dial and what its owner says it does | Empty | Every sub-goal judged from the conversation |
| Provider import | The provider agent's configuration imported, with its tools served by the environment | Built | Checks where its tools act on the world |

## Connect-only, in practice

- Nothing is seeded, because there is nothing to seed into. A scenario's caller brings their own
  details, and checks on world state would test something the agent never shared, so they are not used.
- The scenarios are still proved to set up correctly; the reference solution is cleared and
  solvable and not-vacuous do not apply, so the proof is marked judged-only.
- Tool calls are visible only where the provider's own record of the call can be read; a phone-only
  agent has none.
- The prompt is the whole specification, so the number of genuinely different situations is bounded.
  Depth comes from perturbation: the same objective met by a different person, accent, language, noise
  and wording, rather than invented situations.

## What the chat can never change in any mode

The agent itself: its prompt, code, provider configuration, voice or model, the number or connection a
run dials, and credentials. A contract amendment changes the harness's reading of the agent, not the agent.
