# ALK Harness Self-Repair: Implementation Plan

## 1. Objective

Make a freshly submitted agent repository reliably progress through:

```text
fresh source checkout
  -> source discovery
  -> contract and environment authoring
  -> deterministic bundle compilation
  -> isolated runtime validation
  -> bounded repair
  -> certification
  -> scenario execution
```

without an engineer diagnosing ordinary schema, seed-data, process-wiring, or generated-scenario
compatibility failures for each new agent.

The production guarantee should be:

> For a declared supported runtime and backend, ALK either produces a fresh, validated,
> reproducible environment or stops before calls with a typed, redacted, actionable
> incompatibility. It never silently weakens the environment, invents missing agent tools, or
> reports harness failures as agent-quality failures.

This work is not intended to make arbitrary unknown infrastructure automatically supported. It
is intended to make variation inside supported surfaces deterministic and self-correcting.

## 2. Required production behavior

Every certification and test run uses the production path unless explicitly marked otherwise:

1. Check out the requested remote commit into a clean source directory.
2. Perform fresh authoring with no reused generated environment or scenarios.
3. Compile a fresh Bundle V2 artifact.
4. Provision it on the target snapshot and production-compatible infrastructure.
5. Validate schema, seed data, processes, capabilities, tools, reset behavior, and scenarios.
6. Repair only generated artifacts, then rebuild and reprovision from scratch.
7. Start calls only after certification passes.
8. Upload the complete certification, repair history, call evidence, recordings, transcripts,
   tool traces, durations, CSAT, and evaluations.

Resuming an earlier authored environment or bundle must be an explicit debug operation and must
not count as production certification.

## 3. Current foundation

The implementation is not starting from zero. These pieces already exist:

- `bundle_v2.py` defines a strict manifest for processes, capabilities, stores, readiness,
  provenance, files, and credential purposes.
- `bundle_author_v2.py` packages source and generated authoring artifacts and adopts source-owned
  schema where possible.
- `process_preflight.py` rejects invalid manifests, unsafe process configurations, unresolved
  capabilities, and invalid secret-purpose wiring before runtime.
- `process_runtime.py` provisions managed services and source processes, seeds stores, waits for
  readiness, resets worlds, and emits typed runtime errors for major lifecycle failures.
- `authoring_runtime_validation.py` builds and provisions a fresh candidate, executes every
  scenario's setup and ready phases, checks invariants, and performs bounded environment and
  scenario repair.
- `source_data_invariants.py` derives source-evidenced business-data invariants and holds them fixed
  across repair attempts.
- `hosted_scheduler.py` separates major agent, simulator, environment, and infrastructure failure
  domains and controls retry/discard behavior.
- Bundle provenance includes the source digest and, when available, repository and commit.
- Credential purposes separate source checkout, target-provider, and simulator-provider material.

The principal limitation is that process/runtime metadata is strongly typed while generated
world data is still transported mainly through SQLite and backend-specific SQL. This loses
semantic information and makes low-level database failures arrive at the repair loop as text.

## 4. Design principles

### 4.1 Source is authoritative

Submitted schemas, tool implementations, API contracts, defaults, constraints, and runtime
configuration are the authority. Generated artifacts may adapt to them but may not rewrite them.

### 4.2 Deterministic transformations before model repair

Representation problems are compiler problems. Booleans, arrays, JSON, defaults, timestamps,
enums, keys, and insertion order must be handled by deterministic code.

The authoring model is used only when a semantic choice is missing, such as which valid records
are needed to make a scenario meaningful.

### 4.3 Preserve intent explicitly

The environment representation must distinguish:

- a field omitted so the source default applies;
- an explicit null value;
- an explicitly supplied value.

It must not infer this distinction after round-tripping through SQLite.

### 4.4 Repair generated artifacts only

Self-repair may change the typed world data, generated process adapter, generated setup/ready code,
or scenario definitions. It must not modify submitted source, disable constraints, weaken checks,
remove scenarios, or fabricate missing tool implementations.

### 4.5 Clean-room validation after every repair

No repair is accepted in-place. A candidate patch must be compiled into a new bundle, provisioned
into a clean world, and validated from the beginning.

### 4.6 Fail before paid or externally visible execution

Environment incompatibility must be discovered during certification, not during the fifth call.

### 4.7 Preserve agent failures

Invalid tool arguments, refusal, incorrect tool choice, or failure to complete a task are agent
behavior. The harness must record these accurately rather than repairing them away.

## 5. Target architecture

```text
                         +-----------------------+
Submitted source ------> | Source discovery     |
                         | schema/tools/runtime  |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         | Canonical source model|
                         | typed + fingerprinted |
                         +-----------+-----------+
                                     |
                  model authors      | validates against
                                     v
                         +-----------------------+
                         | World IR              |
                         | logical typed data    |
                         | process intent        |
                         | scenarios/invariants  |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         | Deterministic compiler|
                         | backend adapters      |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         | Bundle V2 candidate   |
                         +-----------+-----------+
                                     |
                                     v
                         +-----------------------+
                         | Runtime validator     |
                         | schema/seed/tools/etc.|
                         +-----------+-----------+
                                     |
                      typed failure  | pass
                          +----------+----------+
                          v                     v
                +------------------+   +------------------+
                | Repair controller|   | Certificate      |
                | bounded policies |   | execution allowed|
                +--------+---------+   +------------------+
                         |
                         +----> new candidate; never mutate active world
```

## 6. Workstream A: canonical source model

### Goal

Extract one normalized, fingerprinted description of the submitted environment before authoring.

### Proposed modules

- `src/fi/alk/harness/source_model.py`
- `src/fi/alk/harness/source_schema/postgres.py`
- `src/fi/alk/harness/source_schema/sqlite.py`
- Later adapters under `source_schema/` for other supported stores.

### Core models

```python
class LogicalType(str, Enum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    UUID = "uuid"
    TIMESTAMP = "timestamp"
    DATE = "date"
    ENUM = "enum"
    ARRAY = "array"
    JSON = "json"
    BINARY = "binary"


class SourceColumn(BaseModel):
    name: str
    logical_type: LogicalType
    native_type: str
    nullable: bool
    has_default: bool
    default_expression: str | None
    generated: bool
    enum_values: tuple[str, ...] = ()
    element_type: LogicalType | None = None


class SourceTable(BaseModel):
    name: str
    columns: tuple[SourceColumn, ...]
    primary_key: tuple[str, ...]
    unique_keys: tuple[tuple[str, ...], ...]
    foreign_keys: tuple[ForeignKey, ...]
```

The extracted model also records:

- process entrypoints and dependency DAG;
- declared health checks and ports;
- tool names and JSON schemas;
- configuration-variable names without resolved values;
- supported source-owned migrations and seed files;
- source digest and per-evidence-file digests.

### Requirements

- Prefer executable metadata inspection over regex parsing when a disposable source database can
  be started safely.
- Preserve the exact native type alongside its logical type.
- Normalize quoted identifiers without changing their spelling.
- Record unsupported constructs explicitly; do not silently coerce them.
- Hash the canonical serialization and include it in provenance.

### Acceptance criteria

- The Uber, Alderway, Retell import backend, and repository fixtures produce stable source-model
  hashes across repeated clean checkouts of the same commit.
- Boolean, array, JSONB, enum, timestamp, UUID, generated/default, composite-key, and foreign-key
  metadata are represented without loss.
- No credential value is written to the source model.

## 7. Workstream B: typed world IR

### Goal

Replace the generated SQLite database as the semantic interchange format. SQLite may remain a
runtime target, but it must not be the only representation of author intent.

### Proposed modules and artifact

- `src/fi/alk/harness/world_ir.py`
- `authoring/world-ir.json`
- JSON Schema published with the Bundle V2 authoring contract.

### Core value model

```python
class ValueState(str, Enum):
    ABSENT = "absent"
    NULL = "null"
    PRESENT = "present"


class WorldValue(BaseModel):
    state: ValueState
    logical_type: LogicalType | None = None
    value: JsonValue | None = None


class WorldRow(BaseModel):
    identity: str
    values: dict[str, WorldValue]


class WorldTable(BaseModel):
    source_name: str
    rows: list[WorldRow]
```

### IR validation

Before compilation:

- Reject unknown tables and columns.
- Reject explicit null for non-null columns.
- Reject authored values for generated columns.
- Validate enum membership.
- Validate array element types recursively.
- Validate JSON values structurally.
- Validate required columns lacking defaults.
- Validate primary and unique keys.
- Validate foreign-key references and build an insertion DAG.
- Detect cycles and select a source-supported resolution strategy or report unsupported.

### Compatibility migration

For one release window:

1. Read `world-ir.json` when present.
2. Otherwise import `world.sqlite` into the IR using a compatibility adapter.
3. Mark every imported SQLite null as `ABSENT` only when the source column has a default or is
   non-null; otherwise mark the intent as ambiguous in the certificate.
4. Emit a deprecation warning and metric for the legacy path.
5. Stop certifying new hosted bundles from the legacy path after the migration gate passes.

The recently added source-default fix remains as the compatibility behavior, not the final data
model.

### Acceptance criteria

- `ABSENT`, `NULL`, and `PRESENT` survive serialization and compilation distinctly.
- Authoring cannot emit raw backend literals as a substitute for logical values.
- The existing authoring output can still be consumed during the migration window.

## 8. Workstream C: deterministic backend compilers

### Goal

Compile validated World IR into backend-native migrations and seeds without model intervention.

### Proposed modules

- `src/fi/alk/harness/compile/base.py`
- `src/fi/alk/harness/compile/postgres.py`
- `src/fi/alk/harness/compile/sqlite.py`
- `src/fi/alk/harness/compile/redis.py`
- `src/fi/alk/harness/compile/http_fixtures.py`

### Compiler contract

```python
class CompileResult(BaseModel):
    files: list[CompiledFile]
    source_schema_hash: str
    world_ir_hash: str
    compiler_version: str
    decisions: list[CompileDecision]
    warnings: list[CompileWarning]
```

Each decision is safe to persist and contains no row values, for example:

```json
{
  "code": "source_default_applied",
  "table": "call_attempts",
  "column": "updated_at",
  "row_identity": "call-1"
}
```

### PostgreSQL rules

- `ABSENT`: omit the column from the insert.
- `NULL`: emit a bound SQL null only for nullable columns.
- Boolean: accept logical booleans only; bind as boolean.
- Array: accept logical arrays only; bind using the driver or emit a correctly typed array.
- JSON/JSONB: serialize once and bind using the driver adapter.
- Enum: validate membership before SQL execution and cast to the discovered native enum.
- Timestamp: require timezone policy and normalize without discarding offset semantics.
- Generated column: always omit.
- Defaults: never copy/evaluate source default expressions in Python.
- Foreign keys: topologically order rows and tables.
- Identifier quoting: derive exclusively from discovered identifiers.
- Inserts should use driver parameters during validation; any emitted seed SQL must be generated
  from the same typed compiler path.

### Compiler behavior

- Compilation is pure and deterministic for the tuple `(source model, world IR, compiler version)`.
- Recompiling the same inputs produces byte-identical output.
- Every conversion is registered centrally. No agent-name or repository-specific branch is
  permitted.
- Unsupported native types produce a typed compile error rather than falling through to SQL.

### Acceptance criteria

- Existing boolean, PostgreSQL array, explicit-null/default, JSONB, enum, timestamp, and insertion
  ordering regressions pass against a real PostgreSQL service.
- No model call occurs for a representation-only mismatch.
- Compiler output is byte-reproducible.

## 9. Workstream D: structured diagnostics

### Goal

Turn every validation failure into a typed object that identifies ownership and a repair policy.

### Proposed modules

- `src/fi/alk/harness/diagnostics.py`
- Backend-specific exception decoders under `src/fi/alk/harness/diagnostic_adapters/`.

### Diagnostic model

```python
class RepairOwner(str, Enum):
    COMPILER = "compiler"
    AUTHORING = "authoring"
    INFRASTRUCTURE = "infrastructure"
    SOURCE = "source"
    AGENT = "agent"
    UNSUPPORTED = "unsupported"


class HarnessDiagnostic(BaseModel):
    stage: str
    domain: FailureDomain
    component: str
    code: str
    owner: RepairOwner
    retryable: bool
    repair_strategy: str | None
    location: DiagnosticLocation | None
    redacted_message: str
    evidence_refs: list[str]
    fingerprint: str
```

### Initial closed taxonomy

| Code | Owner | Default action |
|---|---|---|
| `schema_type_mismatch` | compiler | recompile with discovered type |
| `required_value_missing` | authoring | request targeted data patch |
| `source_default_suppressed` | compiler | mark value absent |
| `array_shape_mismatch` | compiler/authoring | coerce representation or repair semantics |
| `enum_value_invalid` | authoring | select a source-valid value |
| `foreign_key_missing` | authoring | add source-consistent related record |
| `seed_order_invalid` | compiler | reorder inserts |
| `generated_setup_invalid` | authoring | regenerate only setup artifact |
| `ready_condition_invalid` | authoring | repair without weakening intent |
| `process_dependency_timeout` | infrastructure/environment | bounded retry or reject |
| `credential_missing` | source/platform | stop before provisioning |
| `egress_blocked` | infrastructure | stop with required domain |
| `tool_endpoint_unreachable` | environment | repair wiring or reject |
| `tool_schema_mismatch` | source/authoring | reconcile discovered contract |
| `agent_tool_argument_invalid` | agent | record evaluation failure |
| `unsupported_source_construct` | unsupported | actionable rejection |

### Requirements

- Parse database driver fields such as SQLSTATE, table, column, constraint, and datatype rather
  than matching only human-readable messages.
- Redact credentials and row contents before persistence or outbound reporting.
- Stable diagnostic fingerprints prevent repeated identical repairs.
- The original exception is retained only in restricted runtime logs.

### Acceptance criteria

- Every expected failure in the conformance matrix has a stable code, domain, owner, and strategy.
- No raw credential or sensitive row value appears in the persisted diagnostic.
- Platform and scheduler consume the typed fields without re-deriving ownership from text.

## 10. Workstream E: repair controller

### Goal

Replace the current broad textual retry loop with a bounded policy engine.

### Proposed module

- `src/fi/alk/harness/repair_controller.py`

### Repair order

For each failed certification candidate:

1. Classify and redact all failures.
2. Aggregate the complete failure set for the phase.
3. Stop immediately for agent, source, security, or unsupported failures.
4. Retry transient infrastructure failures with bounded exponential backoff and jitter.
5. Apply all deterministic compiler repairs in one new compilation.
6. Request one targeted authoring patch for remaining semantic failures.
7. Validate the patch against source evidence and repair policy.
8. Rebuild a fresh bundle and reprovision a clean runtime.
9. Reject a repeated diagnostic fingerprint with no material artifact change.
10. Stop after the configured repair budget and emit the full repair history.

### Candidate state machine

```text
DISCOVERED
  -> AUTHORED
  -> COMPILED
  -> VALIDATING
      -> CERTIFIED
      -> REPAIRABLE
          -> PATCHED
          -> COMPILED
      -> RETRYABLE_INFRA
          -> VALIDATING
      -> REJECTED
```

### Repair patch format

The model must return a constrained patch rather than arbitrary rewritten files:

```json
{
  "base_world_ir_hash": "...",
  "operations": [
    {
      "op": "add_row",
      "table": "business_profiles",
      "row": {"...": "..."},
      "reason": "foreign_key_missing",
      "evidence_refs": ["src/models.py:BusinessProfile"]
    }
  ]
}
```

The controller rejects operations that:

- modify source files;
- remove or weaken checks;
- delete requested scenarios;
- disable constraints;
- add undeclared services or tools;
- change credential purposes;
- lack verified source evidence when business semantics are introduced.

### Budgets

Use separate budgets so one domain cannot exhaust another:

- deterministic compiler repairs: no model cost, maximum three materially different candidates;
- semantic environment repairs: maximum two model patches;
- scenario repairs: maximum two model patches;
- infrastructure retries: maximum two per candidate;
- repeated diagnostic fingerprint without changed inputs: stop immediately.

The exact values remain configuration, but the categories and stop conditions are part of the
contract.

### Acceptance criteria

- The same diagnostic cannot produce an unbounded loop.
- Multiple related failures are repaired in one candidate instead of one row at a time.
- Every repair records before/after artifact hashes, diagnostics handled, strategy, and outcome.
- Failed candidates are destroyed and never used for calls.

## 11. Workstream F: semantic pre-call validation

### Goal

Prove the generated environment, not merely process liveness, before simulation.

### Validation levels

#### Level 1: static

- Bundle schema and digest.
- Source and generated-file provenance.
- Process DAG and capability resolution.
- Secret-purpose authorization.
- Tool name and parameter-schema consistency.
- World IR validation against the canonical source model.

#### Level 2: runtime infrastructure

- Build/install succeeds.
- Managed stores start and accept semantic health probes.
- Source processes start and remain stable for the startup window.
- Declared dependencies resolve through generated configuration.
- Egress and ingress requirements are available.

#### Level 3: environment behavior

- Migrations and all seed rows load.
- Source-data invariants hold.
- Every scenario setup executes in an independently reset world.
- Every ready condition holds.
- Reset returns the world to the same baseline hash.
- Two parallel worlds cannot observe each other's sentinel mutations.

#### Level 4: tool contract

- Every declared tool has a reachable implementation.
- Request and response schemas match discovery.
- Safe read-only tools receive generated schema-valid probes.
- Mutating tools run only inside disposable worlds and are followed by reset verification.
- External side-effect tools are validated through declared mocks, provider sandboxes, or marked
  `runtime_only`; ALK must not cause uncontrolled real effects during certification.

### Tool probe specification

Add a probe policy to the contract:

```python
class ToolProbePolicy(BaseModel):
    mode: Literal["read_only", "disposable_world", "provider_sandbox", "runtime_only"]
    arguments: dict[str, WorldValue] | None
    expected_result_schema: dict | None
```

Model-authored probe values must satisfy the discovered JSON schema and source-data invariants.

### Acceptance criteria

- Calls cannot start after any Level 1-3 failure.
- Tool reachability and schema coverage are reported separately from actual agent tool use.
- Certification never claims that an agent selected a tool merely because the harness proved the
  tool callable.

## 12. Workstream G: certification and observability

### Goal

Make production failures diagnosable without direct access to ephemeral sandboxes.

### Certification artifact

Replace the minimal `runtime-validation.json` payload with a versioned certificate:

```json
{
  "schema_version": "futureagi.harness-certification.v1",
  "status": "certified",
  "source": {
    "repository": "...",
    "commit": "...",
    "digest": "...",
    "schema_hash": "..."
  },
  "authoring": {
    "contract_hash": "...",
    "world_ir_hash": "...",
    "scenario_set_hash": "..."
  },
  "compiler": {
    "version": "...",
    "bundle_digest": "..."
  },
  "runtime": {
    "snapshot": "...",
    "validation_attempts": 1
  },
  "checks": {
    "static": "passed",
    "schema_and_seed": "passed",
    "processes": "passed",
    "source_invariants": "passed",
    "scenario_setup_ready": "10/10",
    "tool_contract": "9/9",
    "reset_equivalence": "passed",
    "world_isolation": "passed"
  },
  "repairs": [],
  "limitations": []
}
```

### Platform presentation

The platform should show:

- certification status and exact source commit;
- current candidate and attempt number;
- typed failure stage, component, ownership, and next action;
- repair history with redacted summaries;
- explicit distinction between environment certification, tool availability, and agent behavior;
- a downloadable certificate and diagnostics artifact;
- an operator-only link to restricted logs when available.

Large JSON artifacts should remain collapsed by default.

### Metrics

Track at minimum:

- certification success rate for fresh sources;
- first-attempt certification rate;
- deterministic-repair and model-repair rates;
- failure counts by diagnostic code and source backend;
- repeated-fingerprint aborts;
- time and model cost per stage;
- tool-probe coverage;
- calls prevented by pre-call validation;
- environment failures that escaped certification;
- false attribution of harness failures as agent failures, with a target of zero.

### Acceptance criteria

- A developer can diagnose every rejected candidate from persisted redacted artifacts.
- Source commit, snapshot, compiler version, and bundle digest are visible for every run.
- Production alerts fire when an environment failure escapes certification.

## 13. Workstream H: generated and adversarial testing

### Unit and property-based tests

Generate source schemas and World IR values covering:

- nullable/non-null columns;
- present/null/absent values;
- literal and expression defaults;
- booleans and numeric boundaries;
- scalar and multidimensional arrays;
- JSON and JSONB nesting;
- enums;
- UUID, date, timestamp, and timezone behavior;
- quoted and reserved identifiers;
- primary, composite, unique, and foreign keys;
- generated/identity columns;
- cyclic relationships;
- malformed but plausible model outputs.

For each supported case:

1. Compile twice and assert byte-identical output.
2. Load into the real backend.
3. Read it back and compare logical values.
4. Reset and compare baseline state.
5. Assert diagnostics for invalid cases.

### Conformance fixtures

Maintain fixed remote-commit fixtures representing:

- repository-backed LiveKit voice agents;
- provider-imported Retell agents;
- text/chat agents with database-backed tools;
- multi-store agents;
- data-free agents;
- Dockerfile-only and Compose-owned environments;
- agents with no tool implementation, which must be rejected rather than synthesized.

Future computer-use agents should add browser/desktop capability adapters without changing the
World IR or repair-controller contract.

### Production-path campaign

For every release candidate:

```text
fresh remote checkout
  -> no-cache authoring
  -> fresh bundle
  -> candidate snapshot
  -> certification
  -> real dev-compatible calls
  -> complete artifact audit
```

The campaign must include parallel worlds and repeated clean authoring from the same commit to
detect nondeterministic model output. Both generated worlds must independently compile and certify;
they do not need identical business data, but they must obey the same source model and invariants.

## 14. Security and isolation requirements

- Resolved credentials never enter World IR, generated prompts, bundles, diagnostics, or
  certificates.
- Compiler and diagnostic logs redact values using the same secret material available to hosted
  execution.
- Simulator-provider secrets are visible only to simulator processes.
- Target-provider secrets are visible only to target/provider lifecycle processes.
- Source-checkout credentials are destroyed after acquisition.
- Model repair receives source evidence and redacted diagnostics, never secret values.
- Tool probes run with least privilege and an explicit side-effect policy.
- Repair cannot add egress domains or widen sandbox privileges.
- Every candidate runtime is destroyed after validation, pass or fail.

## 15. Delivery sequence

### Milestone 0: lock the baseline

- Add the current Uber, Alderway, Retell, and chat-agent failures as regression fixtures.
- Record current first-attempt pass rate and failure taxonomy.
- Make the clean production-path command the only certification command used by CI.

Exit criterion: every known environment failure is reproducible without cached authoring.

### Milestone 1: source model and typed diagnostics

- Implement the canonical source model for PostgreSQL and SQLite.
- Introduce `HarnessDiagnostic` and database exception decoding.
- Preserve existing artifact formats and repair behavior.

Exit criterion: current failures have stable codes, locations, ownership, and redacted evidence.

### Milestone 2: World IR and PostgreSQL compiler

- Add World IR and compatibility import from `world.sqlite`.
- Implement strict validation and PostgreSQL compilation.
- Route Bundle V2 seed generation through the compiler.

Exit criterion: known boolean, array, null/default, JSON, enum, timestamp, and key-order cases load
without model repair.

### Milestone 3: policy-driven repair

- Implement the repair controller and constrained authoring patches.
- Detect repeated diagnostics and no-op repairs.
- Persist complete candidate and repair history.

Exit criterion: repair is bounded, transactional, policy-checked, and independently auditable.

### Milestone 4: semantic tool certification

- Add tool probe policies and generated schema-valid probes.
- Validate reset and isolation after mutating probes.
- Mark unprobeable external tools explicitly.

Exit criterion: every declared tool has a certified availability state before calls.

### Milestone 5: platform gate and observability

- Emit the versioned certificate through hosted callbacks.
- Require a valid certificate before run dispatch.
- Expose typed failures, repair history, provenance, and limitations in the platform.

Exit criterion: a rejected run is diagnosable without Daytona shell access, and an uncertified
bundle cannot start calls.

### Milestone 6: generated-schema and production-path release gate

- Add property-based backend tests.
- Run the fixed multi-agent conformance campaign.
- Measure escape rate and first-attempt certification rate.

Exit criterion: zero environment failures escape certification across the release campaign, and
all supported generated-schema cases pass reproducibly.

## 16. Proposed code-change map

| Current file | Change |
|---|---|
| `bundle_author_v2.py` | Consume compiler output instead of directly translating SQLite rows |
| `bundle_v2.py` | Add compiler/certificate references and versioned metadata |
| `authoring_runtime_validation.py` | Delegate decisions to repair controller; aggregate typed diagnostics |
| `source_data_invariants.py` | Bind invariants to source-model and scenario-set hashes |
| `process_preflight.py` | Validate compiler artifacts and certification prerequisites |
| `process_runtime.py` | Emit structured backend/runtime diagnostics without losing SQLSTATE/context |
| `hosted_scheduler.py` | Consume carried ownership/retry policy; do not reclassify from strings |
| `authoring_entrypoint.py` | Produce World IR and invoke clean-room certification |
| `world/snapshot.py` | Treat SQLite as a backend target/compatibility artifact, not semantic authority |

New files:

```text
src/fi/alk/harness/source_model.py
src/fi/alk/harness/source_schema/
src/fi/alk/harness/world_ir.py
src/fi/alk/harness/compile/
src/fi/alk/harness/diagnostics.py
src/fi/alk/harness/diagnostic_adapters/
src/fi/alk/harness/repair_controller.py
src/fi/alk/harness/certification.py
```

## 17. Definition of done

The self-repair project is complete when all of the following hold:

1. Every hosted test begins from a clean remote checkout and uncached authoring by default.
2. The semantic authoring artifact distinguishes absent, null, and present values.
3. Supported backend seeds are produced by deterministic typed compilers.
4. Representation mismatches never consume a model repair attempt.
5. Runtime failures carry a stable code, stage, component, owner, retry policy, and redacted
   evidence.
6. Repairs are constrained patches to generated artifacts and validated in a fresh runtime.
7. Repeated/no-op repairs terminate deterministically.
8. Schema, seed, processes, source invariants, scenario setup/ready, reset, isolation, and tool
   availability pass before calls.
9. An uncertified bundle cannot be dispatched.
10. The platform exposes enough persisted evidence to diagnose failures without ephemeral-shell
    access.
11. Known Uber, Alderway, Retell, chat, and multi-store fixtures certify from fresh source.
12. Generated-schema tests cover all supported logical/native type combinations.
13. No environment or infrastructure failure is reported as an agent-quality failure.
14. Missing source tool implementations are rejected and never invented.
15. Production-path certification has zero harness/environment escapes for the release campaign.

## 18. Immediate next actions

1. Land this design as the implementation contract.
2. Add regression fixtures for every failure already observed, including boolean/smallint,
   PostgreSQL array literals, source defaults suppressed by nulls, and relationship/insertion
   ordering.
3. Implement `SourceModel`, `WorldValue`, and `HarnessDiagnostic` without changing runtime behavior.
4. Route PostgreSQL seed generation through the typed compiler behind a feature flag.
5. Run dual compilation in CI: retain the current output for execution, compare it with the new
   compiler, and report divergences.
6. Promote the new compiler only after the fixed-agent and generated-schema matrices pass.
7. Add the certification gate and then remove the legacy SQLite-authority path for new hosted
   authoring.

This sequence preserves the working Bundle V2 runtime while replacing the fragile authoring-to-
seed boundary incrementally. It targets the recurring root cause instead of accumulating
repository-specific patches.
