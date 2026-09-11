# Harness implementation and validation status

Status date: 22 August 2026

This document records what is implemented and what has actually been exercised. It complements
`ARCHITECTURE.md` (the target design) and `ENVIRONMENT_CONFORMANCE.md` (the packaging contract).

## Product boundary now implemented

ALK owns the execution plane: repository understanding, environment construction, test-data
setup, scenario execution, evidence collection, grading and artifact production. The platform is
the control and data plane: it creates jobs, supplies job-scoped secret references, receives
events/results and presents history. Local CLI execution and hosted execution use the same
`HarnessJob`, `EnvironmentBundle`, executor and result contracts.

The central invariant is enforced throughout provisioning: ALK provides the environment in which
the submitted agent and its existing tools run. It does not rewrite tools, invent proprietary
service behavior or silently replace an unsupported dependency.

## Completed implementation

### Portable jobs, bundles and runtimes

- Immutable local and hosted job contracts, with local paths rejected for hosted jobs.
- Content-addressed environment bundles with provenance, per-file hashes and sizes.
- Bundle verification before execution; symlinks, path traversal and changed content are rejected.
- A runtime-provider boundary so local Docker/Compose can later be replaced by a hosted sandbox
  provider without changing the harness pipeline.
- Structured lifecycle events, cancellation and terminal result delivery for platform-driven jobs.
- Retry classification that separates transient infrastructure/transport failures from agent
  failures and avoids rerunning deterministic agent outcomes.

### Repository and environment provisioning

- **Compose repository:** adopt the submitted Compose model, isolate project names, ports, networks
  and volumes, start dependency services, inject test endpoints and clean up only that run.
- **Dockerfile-only repository:** keep the submitted runtime unchanged and generate only the
  declared infrastructure around it. Managed adapters currently cover Postgres, ClickHouse,
  Redis, MongoDB and Qdrant, including initialization, protocol-aware readiness and reset
  behavior.
- **Neither Compose nor Dockerfile:** fail admission with an actionable packaging error when an
  external runtime is required. Automatic packaging for this third case remains open work.
- Source fingerprints prevent reuse of stale builds.
- Protocol-aware readiness avoids treating a merely open socket as a ready database.

### Credentials, GitHub input and isolation

- Static credential discovery from source, Compose, Dockerfile and known SDK connectors.
- Preflight reports missing credential names before a sandbox is started.
- Jobs and bundles contain `SecretRef` references, not resolved values; secret values are rejected
  from persisted payloads and redacted from emitted output.
- GitHub repository/revision references are represented in the hosted job contract. The platform
  remains responsible for repository authorization and job-scoped credential resolution.
- Unsafe Compose features such as privileged execution and host escape configuration are rejected.
- Resource/lifecycle ownership is scoped to the run so cleanup cannot target another environment.

### Evidence and artifact integrity

- Setup activity is kept separate from agent activity and cannot earn evaluation credit.
- A missing tool call cannot satisfy a tool-dependent check; dependent checks cannot pass when the
  prerequisite action is absent.
- Results distinguish environment, connectivity, simulator, agent, grading and artifact failures.
- Artifact manifests include hashes and sizes, are verified on ingestion and are delivered with
  an idempotency identity for safe platform retries.
- LiveKit recording selection now ignores ambient/background publications and selects the actual
  conversational speech track.

## Environment conformance tested

The checked-in fixtures are small conformance repositories, not product demo agents:

| Packaging case | Dependencies | What was proven |
|---|---|---|
| Compose | ClickHouse + Redis | Two copies can run simultaneously despite fixed submitted ports; each worker reaches its own seeded services; resetting one does not affect the other. |
| Dockerfile only | ClickHouse + Redis | ALK composes managed dependencies around the unchanged image, injects endpoints and verifies seeded data from the running agent. |
| Dockerfile only | Postgres | ALK creates the database, applies submitted SQL, verifies application readiness and performs isolated cleanup. |
| Dockerfile only | MongoDB + Qdrant | ALK creates both services, the runtime writes/reads both, reset recreates clean state and cleanup removes the project. |

The real Docker conformance matrix exercised build, startup, readiness, state mutation, reset and
teardown. The detailed contract and commands are in `ENVIRONMENT_CONFORMANCE.md`.

## Real voice-agent validation

Three agents from the official `livekit/agents` examples repository were validated at commit
`da6af86ac640a3bc54585764e64321d7048c1c16`:

- `drive_thru` — multi-tool ordering with in-process state;
- `frontdesk` — scheduling with an optional external Cal.com connector and source fallback; and
- `hotel_receptionist` — a larger booking flow with local SQLite state.

For all three, ALK detected the Dockerfile-only runtime, built the upstream image, produced and
verified a sealed bundle, started the worker, checked readiness and source integrity, and cleaned
up the execution environment.

Five real WebRTC calls were then made to each agent (15 total):

| Agent | Calls | Completed | Environment failures | Preserved agent failures |
|---|---:|---:|---:|---:|
| Drive-through | 5 | 5 | 0 | 0 |
| Front desk | 5 | 5 | 0 | 0 |
| Hotel receptionist | 5 | 4 | 0 | 1 |
| **Total** | **15** | **14** | **0** | **1** |

The hotel failure is a useful deterministic finding rather than a harness failure: the agent
repeatedly treated a valid 16-digit card number as 15 digits, entered a validation loop and reached
the seven-minute scenario limit.

Artifact checks passed for every call:

- 15/15 reports and transcripts were created;
- 15/15 combined recordings and 15/15 stereo recordings were present and larger than 100 KB;
- all recordings contained measurable audio; and
- all campaign containers were stopped after execution.

The public example repositories, temporary model-adapted checkouts, recordings and campaign
artifacts are intentionally not committed to ALK.

### Hosted-provider connector validation

Two non-native target connectors were exercised with real calls using the unchanged configured
provider agents:

| Connector | Result | Evidence |
|---|---|---|
| Vapi WebSocket | Completed; evaluation 0.8401 | 14 committed messages, 1,071 transcript characters, SDK and provider recordings; combined audio peak -3.6 dB. |
| Retell Web Call | Completed; evaluation 0.8422 | 11 committed messages, 1,432 transcript characters, SDK and provider recordings; combined audio peak -1.9 dB. |

Both ended through `simulator_end_call`, produced no typed call failure and cleaned up their
FutureAGI LiveKit rooms. Daily and direct OpenAI Realtime remain separate transport gates; model
substitution must not be represented as transport validation.

## Additional public-repository packaging validation

The environment admission path was also exercised against twelve structurally different public
voice-agent repositories on 22 August 2026. These checks intentionally used untouched checkouts;
repository defects remain findings instead of being patched by ALK.

| Repository | Revision | Packaging result | Runtime result |
|---|---|---|---|
| Bolna | `0172347b601ea66dac0414cc1c6b14dc0d85422a` | Nested `local_setup/docker-compose.yml` is now discovered. Host-bound AWS credential mounts are reported explicitly. | Not started: its local Compose requires provider/telephony credentials and a local `.env`; ALK correctly stops before build rather than fabricating them. |
| Pipecat examples (`websocket`) | `696c3541001350262dd11e81ccbf00754a56850c` | Root Dockerfile detected; `env.example` and its placeholder Google key are now recognized. | Rejected before Docker because the published Dockerfile requires an absent `uv.lock`. The same failure previously took about 80 seconds in BuildKit; static admission now reports it in about 1.2 seconds. |
| TEN Framework (`ai_agents`) | `2e56d9659d8599350962374c0dc24725a03d73ce` | Development Compose and production Dockerfile are distinguished; ALK selects the production Dockerfile and preserves the repository's `linux/amd64` hint. | The upstream release reaches `task install` but Bun exits 132 under Docker Desktop emulation on the ARM test host. This is recorded as host/runtime compatibility, not an environment success or agent evaluation. |
| Pipecat Cloud Starter | `9167c21ea76a67a02f330bae0c009d0ef5a6ef95` | Its root Dockerfile and sample credential declarations are accepted without modification. | The submitted image builds, but its real entrypoint fails: the archived repository leaves `pipecat-ai` unpinned and current 1.7.0 removed `LLMMessagesFrame`. Build success alone must not be represented as runtime success. A test-only compatibility checkout produced the real-call result below. |
| Voice Noob | `755552f1b92b3c760799720f2c250e7f0a6bf0bd` | Its Compose-provided Postgres and Redis are selected as infrastructure. Profile-gated pgAdmin and Redis Commander are not mistaken for agent runtimes. | Infrastructure only: the unchanged services started and passed readiness in 6.348 seconds, reset cleanly in 6.238 seconds and were removed. The repository does not package its FastAPI agent runtime in Compose or a Dockerfile; its browser uses direct OpenAI Realtime WebRTC, which the current LiveKit-only call engine cannot drive. This is not an end-to-end agent success. |
| Vocode Core (`telephony_app`) | `e054c33a72787b6a4920f91eb8598ad0bafb4240` | The component's Dockerfile and Compose are both found; the missing Compose `.env` is reported per affected service. | The unchanged Dockerfile reached dependency installation, then failed because it installs current Poetry while invoking the removed `--no-dev` option. This is an upstream reproducibility failure and is not patched by ALK. |
| Open Telephony Stack | `a42e5b6c17c772af72825dd40240807c415b220b` | The independently runnable Asterisk, shim and voice-server components are reported as an ambiguous monorepo rather than guessed. | The Asterisk Compose is rejected for hosted execution because it requests host networking and host TLS/system mounts. Its voice-server Dockerfile also requires the repository root as build context, proving the need for explicit component plus build-context selection. |
| Dograh | `058c540c4d92c55f529d04fabceb17da4901a0cb` | The root Compose runtime is selected; credential discovery is scoped to that runtime instead of every optional provider module. | The unchanged default stack started API, UI, pgvector/Postgres, authenticated Redis and MinIO in 36.391 seconds and cleaned up. This found and fixed authenticated-Redis readiness, empty-body HTTP readiness and dormant-profile port allocation. |
| Voice Asterisk Agent | `6a5e0533b5293d1727847d123bf2ab8a1fc136de` | Asterisk, AudioSocket, API, agent, STT/LLM/TTS and monitoring components are discovered without mistaking `Dockerfile.dockerignore` for a runtime. | Admission remains blocked by missing external env injection, component selection and forbidden Docker-socket access. These are real hosted-environment requirements, not an agent grade. |
| Aeyetech local voice agent | `34df18ab3e023bbf99b103bc4b891556999b6787` | Root Compose and local LiveKit/Whisper/llama.cpp/Kokoro/agent components are found. | Not started: a machine-specific absolute model bind mount and large CPU/GPU model requirements need explicit resource and mount admission first. |
| Salesforce VoiceAgentRAG | `4d653890db18855d564d4ed9ca4d678047799cf4` | Qdrant/FAISS and local/cloud model requirements are visible, but the repository has neither Compose nor a Dockerfile. | Correct case-three packaging finding; ALK does not silently invent a runtime. |
| LiveKit Node starter | `2045cc4917e40382fcf6b39e46ce24316a97370a` | Dockerfile-only Node runtime detected. Multi-source `COPY` validation now checks every input. | Rejected before Docker because the published Dockerfile requires absent `pnpm-lock.yaml`; the earlier BuildKit failure is now a deterministic admission finding. |

The new preflight is deliberately conservative. It detects missing Docker build inputs, ambiguous
monorepo component roots, nested Compose files, development-oriented Compose configuration,
privileged/host namespace requests and host bind mounts before starting a container. Large Docker
errors are bounded before entering job state so one failed build cannot overwhelm the platform UI.

### Pipecat test-only real-call validation

Because no Daily credential was available, a temporary external checkout replaced Daily with
Pipecat's officially supported LiveKit transport. The archived starter's prompt and LLM behavior
were preserved. Deepgram supplied transcription and OpenAI replaced a Cartesia credential that
returned HTTP 401. These compatibility changes were not made to ALK or committed to the public
repository.

One real harness-driven WebRTC call then completed with 10 committed messages, 1,060 transcript
characters and `simulator_end_call`. Combined and stereo recordings were produced at 1,295,634
and 2,591,224 bytes; measured peak audio was -2.9 dB, proving that the files were not silence. The
target received and transcribed the caller audio, generated responses through its LLM pipeline and
returned synthesized speech through LiveKit. Its isolated runtime was removed after the call.

### Test-only compatibility note

The current LiveKit server returned HTTP 401 for the example agents' LiveKit Inference models. To
exercise real calls, temporary external checkouts used direct Deepgram and OpenAI model providers
for STT/LLM/TTS. This was a test setup change only: no agent tools, prompts, database behavior or
harness source was rewritten. The temporary checkouts are not part of this branch.

## Automated release gate

The harness-focused gate completed with:

```text
508 passed, 11 skipped
```

The skipped cases are opt-in Docker/real-agent checks and require their documented runtime flags
and credentials. The WebRTC engine regression suite is included in the 508 passing tests. Ruff and
`git diff --check` also pass for this change set.

## Findings that remain open

These items should not be represented as complete:

1. Implement the production hosted provider (for example Daytona or a microVM fleet) behind the
   existing runtime interface, with quotas, egress policy and hard cleanup guarantees.
2. Add an explicit packaging adapter for repositories with neither Compose nor a Dockerfile.
3. Expand managed dependency conformance beyond Postgres, ClickHouse, Redis, MongoDB and Qdrant.
   RabbitMQ/NATS, MinIO/S3 and browser/code-execution services are the next explicit gates;
   unsupported services must continue to fail explicitly.
4. Add first-class component/build-context selection for monorepositories. Detection and
   actionable ambiguity are implemented, but the platform does not yet let the user select one
   of several independently runnable agents within a repository. Credential discovery now follows
   an automatically selected Compose runtime and its build context, but must also follow an
   explicit user-selected component in ambiguous monorepositories.
   Compose files that require a repository-local `.env` also need a job-scoped external-env
   adapter; ALK currently reports the missing file rather than mutating the submitted checkout.
   Internal environment secrets such as a local stack's JWT signing key should be generated as
   job-scoped secret references rather than requested from the customer.
5. Add generic target-tool event ingestion for uninstrumented third-party LiveKit agents. Audio and
   transcripts work today, but source-owned tool calls cannot always be normalized automatically.
   Add target media adapters for non-LiveKit agents, beginning with direct OpenAI Realtime WebRTC,
   Daily and generic bidirectional audio WebSocket. Voice Noob cannot be called through the current
   engine until one of these adapters exists; switching its model does not change its transport.
6. Productize GitHub authorization (GitHub App), secret storage/resolution and credential UX on the
   platform. The ALK contracts and preflight are present; the production integrations are not.
7. Improve platform run UX with granular live stages, logs, actionable waiting states and artifact
   reconciliation rather than a long generic status.
8. Continue work on diverse scenario/persona generation and controllable simulator behavior.
9. Add production object-storage upload/reconciliation, retention policy and load/chaos testing.
10. Track LiveKit transport instability observed as intermittent signal-resume 502/EOF responses.

## Current conclusion

The local execution architecture and the environment creation path for well-packaged and
Dockerfile-only voice agents are functioning end to end. Real agents could be built, started and
called through the generated environment, and complete transcripts and audio artifacts were
produced. The remaining work is primarily the production hosted sandbox/provider integration,
broader packaging/service coverage, third-party tool telemetry and platform product experience.
