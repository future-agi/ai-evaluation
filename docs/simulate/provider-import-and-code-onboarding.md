# Vapi and Retell onboarding: imported targets and repository-created targets

## Status

This is the implementation contract for the hosted harness. It supersedes the earlier assumption
that every environment-backed provider target must be authored entirely by repository code.

## User flows

### 1. Import an existing provider target

The user supplies a Vapi assistant ID or Retell agent ID, a matching run-scoped API key, and the
backend source that implements the target's HTTP tools. The job uses `mode: provider_import`.

ALK:

1. understands and builds the submitted backend source;
2. requires one unambiguous HTTP capability (or `config.public_capability`);
3. fetches the provider definition with the user's key;
4. copies the definition in the user's provider account;
5. rewires custom HTTP tools and provider events to the isolated world's signed HTTPS endpoint;
6. calls only the clone;
7. records every created resource as attempt-owned; and
8. deletes only those recorded resources during cleanup.

The original target is never patched or deleted. Provider-native tools remain provider-native.
Opaque custom implementations and unsupported Retell response engines fail before dialing; ALK
does not synthesize business logic. Retell imports support both Retell LLM and Conversation Flow
response engines. Conversation Flow graphs are copied as a unit, including embedded components,
and every nested custom tool is validated and rewired before any provider resource is created.

### 2. Create from repository code

The user uploads or links a repository containing the complete agent definition, tools and an
`alk.yaml` provider lifecycle. The job uses `mode: environment_backed`.

ALK builds the environment, supplies signed event/tool URLs and the run-scoped provider key, then
executes the repository's provision command. The command creates the provider graph and returns a
validated ownership receipt. Its destroy command removes those resources.

### 3. Connect without cloning

`mode: connect_only` remains a compatibility and diagnostic lane. It calls the supplied ID as-is,
does not rewire tools, and never owns provider resources.

## Trust and credential boundaries

- Vapi/Retell keys are `target_provider` secrets. They are resolved only in the hosted attempt.
- Platform simulator credentials are `simulator_provider` secrets and never enter customer
  processes.
- For Vapi and Retell, the LiveKit room used by the simulated caller is platform infrastructure:
  its URL/key/secret are `simulator_provider` values. For a LiveKit agent-under-test, that
  agent's own LiveKit URL/key/secret remain `target_provider` values. Hosted execution never
  falls back across this boundary.
- Provider definitions, lifecycle receipts and bundles contain no secret values.
- Signed public URLs are capability material and are not persisted in receipts.
- Imported resources carry a deterministic `alk-<job>-w<world>` name/metadata prefix where the
  provider supports it.

## Import support matrix

| Provider | Copied resources | Rewired fields | Deliberate rejection |
| --- | --- | --- | --- |
| Vapi | reusable tools, assistant | assistant server, function/API-request tool URLs | referenced resources that cannot be fetched or copied |
| Retell | Retell LLM or Conversation Flow, voice agent | every nested custom-function URL, agent webhook | custom-LLM and undeclared custom-tool implementations |

## Acceptance gates

Every provider/mode fixture must prove: source build and readiness, clone/provision receipt,
original target unchanged, five multi-turn calls, transcripts and recordings, provider call IDs,
tool evidence when the fixture has tools, evaluation coverage, and cleanup of every owned provider
resource. A green call with missing required evidence is not a pass.

## Reproducible code-upload fixtures

`examples/harness/provider_voice_environment/` contains self-contained Vapi and Retell
repository-created targets plus an import backend. Both targets expose a real
`record_preference` implementation, return ownership receipts, and delete resources in reverse
dependency order. They are certification inputs, not hidden harness implementations.
