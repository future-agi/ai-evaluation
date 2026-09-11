# Environment packaging conformance

The harness supports repository packaging incrementally while preserving one invariant: it may
package submitted code and provide infrastructure, but it never rewrites or invents agent tools.

## Case 1: repository supplies Compose

ALK adopts the submitted Compose definition and:

- renders it with Compose rather than reimplementing its schema;
- rejects privileged/host escape features;
- assigns an isolated project, host ports and volumes;
- starts only default infrastructure services;
- identifies opt-in agent/worker services;
- discovers typed and generic TCP capabilities;
- injects external/internal endpoints through existing configuration seams;
- waits for Compose healthchecks, protocol-level readiness and a short startup-stability window;
- performs lifecycle reset with only that project's volumes; and
- tears down only that project.

Conformance fixture: `tests/fixtures/harness_agents/voice_analytics_agent`.

The real test starts two simultaneous copies even though the submitted Compose publishes fixed
ClickHouse and Redis ports. It starts both unchanged worker images, proves each reaches its own
seeded ClickHouse and Redis services, resets one project, verifies the other remains healthy, and
cleans up both.

## Case 2: repository supplies Dockerfile but no Compose

ALK uses the submitted Dockerfile unchanged and generates a Compose adapter around it. A
standalone runtime is valid and gets no invented infrastructure. When the contract declares
dependencies, the adapter contains only those supported infrastructure services. Current managed
templates are:

| Engine | Connector | Reset mechanism |
|---|---|---|
| Postgres | `DATABASE_URL`/declared seam | project volume recreation + submitted SQL init |
| ClickHouse | `CLICKHOUSE_URL`/declared seam | project volume recreation + submitted SQL init |
| Redis | `REDIS_URL`/declared seam | project volume recreation |
| MongoDB | `MONGODB_URL`/declared seam | project volume recreation |
| Qdrant | `QDRANT_URL`/declared seam | project volume recreation |

Multiple dependencies are included in one private network and injected into the submitted
runtime. Generated services use no persisted resolved password, so the environment bundle passes
secret scanning and content sealing.

Conformance fixtures:

- `voice_analytics_agent` with Compose removed: managed ClickHouse + Redis;
- `voice_ledger_agent`: managed Postgres.

An additional generated Dockerfile-only conformance agent uses MongoDB and Qdrant together. Its
unchanged runtime writes both services, ALK destroys and recreates the project volumes, and a
second runtime proves both stores begin empty. The real integration gate completes in about 41
seconds on the current Docker Desktop host.

The public `dograh-hq/dograh` root Compose is also exercised unchanged. Its default API, UI,
pgvector/Postgres, password-protected Redis and MinIO services pass readiness and cleanup while
profile-gated TURN/init/tunnel services remain dormant. See `ENVIRONMENT_VALIDATION_MATRIX.md`
for evidence levels and the wider repository batch.

Both real tests build and run the submitted Dockerfile, query submitted seed data from inside the
runtime, emit application readiness evidence and clean up.

### Official LiveKit agent validation

The opt-in real-agent suite also validates three unmodified agents from the official
`livekit/agents` examples repository at commit
`da6af86ac640a3bc54585764e64321d7048c1c16`:

- `drive_thru`: multi-tool ordering agent with in-process business state;
- `frontdesk`: scheduling agent with an optional external Cal.com integration and source-owned
  fallback; and
- `hotel_receptionist`: larger multi-tool booking agent with SQLite-backed local state.

For each repository ALK detects a Dockerfile-only standalone runtime, builds the upstream image,
seals and verifies a portable environment bundle, starts the actual LiveKit worker with referenced
credentials, checks runtime health, verifies the source fingerprint did not change, and tears the
job down. No database or service is synthesized for these agents.

Credential discovery includes requirements consumed internally by a detected connector SDK. For
LiveKit that means `LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` are requested before
worker startup even when submitted code contains no direct environment read. Resolved values are
passed through the child process environment and selected by name; they are never placed in
Docker command arguments or persisted in a bundle.

## Explicit failure behavior

- A missing custom service/tool implementation is not generated.
- An unsupported engine is not replaced with a different database.
- External infrastructure with no Compose and no supported managed adapter fails before calls.
- A required runtime with no Compose and no Dockerfile fails with an actionable packaging error.
- Infrastructure/readiness failures are not agent evaluation results.

## Running the matrix

Unit and contract checks:

```bash
.venv/bin/pytest -q tests/test_harness_service_environments.py
```

Real Docker checks:

```bash
RUN_INTEGRATION=1 .venv/bin/pytest -q tests/test_harness_service_environments.py
```

Official LiveKit example build and bundle checks:

```bash
RUN_INTEGRATION=1 \
LIVEKIT_EXAMPLES_ROOT=/path/to/livekit-agents/examples \
.venv/bin/pytest -q tests/test_harness_livekit_examples.py -k builds
```

Add `LIVEKIT_EXAMPLES_START_WORKERS=1` and provide the three discovered `LIVEKIT_*` values to
include real worker registration/readiness checks.

The real suite pulls/builds container images, starts services, mutates state, resets and removes
all test projects. It should run on CI with a dedicated Docker daemon, not a shared production
host.

Known protocols are checked semantically where possible (`/ping` for ClickHouse and `PING` for
Redis). This prevents a container or briefly-open socket from being reported ready while database
initialization is still restarting the service. Later liveness probes are point-in-time checks and
do not reapply the startup window.
