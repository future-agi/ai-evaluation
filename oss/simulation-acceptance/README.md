# Simulation acceptance harness

This directory runs the voice matrix one cell at a time and provides a separate chat smoke. Direction is from the **target agent's** perspective:

- `inbound`: the target receives the interaction; the simulator speaks first.
- `outbound`: the target initiates the interaction; the target speaks first.

The FutureAGI simulator always runs in the LiveKit runtime configured by `ACCEPTANCE_LIVEKIT_URL`.

## Install and configure

```bash
cd agent-learning-kit
uv sync --extra livekit --group dev
cp oss/simulation-acceptance/.env.example .env.acceptance
# Fill the required values. Never commit this file.
set -a && source .env.acceptance && set +a
```

The scripts use an explicit scenario. The simulator LLM defaults to Gemini; STT/TTS default to Deepgram. Configure each independently with `SIMULATOR_{LLM,STT,TTS}_PROVIDER` and the matching model variables. STT/TTS support Deepgram, Google, OpenAI, ElevenLabs, and Cartesia (`CARTESIA_API_KEY`; defaults `ink-2` and `sonic-3`). Every voice run creates a fresh run ID, a managed invocation-unique room, a manifest, recordings, and a typed report under `artifacts/simulation-acceptance/`.

For `gemini-3.x` on Vertex, the SDK selects the global endpoint unless `GOOGLE_CLOUD_LOCATION` or `VERTEX_LOCATION` is explicitly set. Google-only voice runs need no Deepgram key and receive a 210-second call budget. Google STT accepts comma-separated languages through `SIMULATOR_STT_LANGUAGE=en-US,es-ES`. Use `start`, not `dev`, for a target worker during measured runs; hot reload can replace a registered worker while a dispatch is pending.

For direct Vapi/Retell cases, copy the target's current system prompt into the matching `*_TARGET_SYSTEM_PROMPT` variable. Provider keys remain environment-only.

Use a real company name in test prompts and greetings. Do not pass instructional placeholders such as "identify yourself as the parcel carrier's support line"; a target can speak that wording literally.

Each single-case harness run writes audio directly under `<output-root>/<run-id>/<case-id>/recordings/audio/`; it does not repeat the run and case directories inside `recordings/`. The SDK report paths are authoritative.

## Voice commands

Use `--dry-run` first to validate configuration without placing a call:

```bash
uv run --extra livekit python oss/simulation-acceptance/run_voice_case.py 1.1.1 --dry-run
```

Remove `--dry-run` to execute. A blocked case is still runnable for diagnosis; it exits non-zero with the SDK's typed failure instead of being reported as working.

| Case | Target path | Current status | Command | Additional setup |
| --- | --- | --- | --- | --- |
| 1.1.1 | LiveKit inbound telephony | Proven | `python .../run_voice_case.py 1.1.1` | Outbound trunk, caller number, target LiveKit phone number |
| 1.1.2 | LiveKit inbound WebRTC | Proven | `python .../run_voice_case.py 1.1.2` | Registered target worker name |
| 1.2.1 | LiveKit outbound telephony | Proven | `python .../run_voice_case.py 1.2.1` | Caller-scoped inbound trunk and an outbound-enabled target worker |
| 1.2.2 | LiveKit outbound WebRTC | Proven | `python .../run_voice_case.py 1.2.2` | Registered target worker configured to speak first |
| 2.1.1 | Vapi inbound telephony | Proven | `python .../run_voice_case.py 2.1.1` | Outbound trunk and Vapi target phone number |
| 2.1.2 | Vapi inbound web | Proven | `python .../run_voice_case.py 2.1.2` | Vapi API key and assistant ID |
| 2.2.1 | Vapi outbound telephony | Proven | `python .../run_voice_case.py 2.2.1` | Caller-scoped inbound trunk, working LiveKit SIP ingress, and a Vapi phone number capable of outbound calls |
| 2.2.2 | Vapi outbound web | Proven | `python .../run_voice_case.py 2.2.2` | Vapi assistant initial message |
| 3.1.1 | Retell inbound telephony | Proven | `python .../run_voice_case.py 3.1.1` | Outbound trunk and Retell target phone number |
| 3.1.2 | Retell inbound web | Proven | `python .../run_voice_case.py 3.1.2` | Retell API key and agent ID |

Prefix the commands with `uv run --extra livekit` when the virtual environment is not activated.

`status=completed` means the transport produced a sufficiently balanced conversation. It is not a behavioral grade. The harness also attaches `evaluation_passed` and `evaluation_score`; use those fields to gate outcome adherence.

### Telephony notes

- PSTN runs use 150 seconds because ringing and carrier setup consume part of the call budget.
- Case `1.2.1` dispatches `LIVEKIT_TARGET_AGENT_NAME` into a source room; that worker creates the SIP participant and therefore genuinely initiates the call.
- The reference worker requires `REFERENCE_AGENT_OUTBOUND_SIP_ENABLED=true` and `REFERENCE_AGENT_OUTBOUND_SIP_ALLOWED_NUMBER` equal to `LIVEKIT_INBOUND_DID`; calls to any other destination fail closed.
- `sip_inbound` without `dispatch_rule_name` requires `LIVEKIT_INBOUND_TRUNK_ID`. A pre-existing direct dispatch rule does not.
- Use a dedicated caller-scoped inbound trunk with the DID in `numbers` and originating caller IDs in `allowed_numbers`; this can coexist with the platform pool trunk.
- Vapi provider-managed numbers can originate domestic outbound calls, subject to Vapi's daily limits; imported telephony numbers are still recommended for production scale.
- `endedReason=call-deleted` after SDK cleanup is retained as provider evidence but annotated as SDK teardown.
- Provider recordings and correlation are best effort; SDK-owned LiveKit recordings are authoritative.

The Platform closes simulator calls through the `endCall` tool. The SDK follows that behavior and adds the same kind of safety backstop used by the LiveKit worker: participant disconnect plus a 30-second post-conversation silence timeout. It does not use substring matching on goodbye text.

## Chat

Configure `CHAT_TARGET_URL` and choose either `CHAT_TARGET_PROTOCOL=openai_chat` or `fi.alk`.

```bash
uv run python oss/simulation-acceptance/run_chat.py
```

Verify that one crashing persona no longer destroys healthy results:

```bash
uv run python oss/simulation-acceptance/run_chat.py --failure-isolation-probe
```

The expected probe statuses are `completed`, `failed`, `completed`, with a redacted typed failure on the middle persona.

## Platform Scenario reuse

Platform generation accepts 10–20,000 rows. For a cheap smoke, generate 10 once and run a one-row slice locally. Reusing the same Scenario name for the same Agent Definition downloads the existing processing/completed Scenario instead of creating another one. Use a new name when a genuinely new dataset is wanted. `fetch_scenario(scenario_id)` remains the canonical way to resume or download an existing Scenario.

## External references

- [Vapi List Calls API](https://docs.vapi.ai/api-reference/calls/list)
- [Vapi private call artifact retrieval](https://docs.vapi.ai/security-and-privacy/retrieve-call-artifacts)
- [Vapi Get Phone Number API](https://docs.vapi.ai/api-reference/phone-numbers/get)
- [Retell List Calls API](https://docs.retellai.com/api-references/list-calls)
- [LiveKit SIP API](https://docs.livekit.io/reference/telephony/sip-api/)
