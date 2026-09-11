# Vapi and Retell hosted provider certification — 2026-09-02

This report is maintained from the live certification campaign in
`provider-vapi-retell-matrix-20260902/REPORT.md` at the workspace root. The final verified result is:

- Retell provider import: 5/5 calls passed end to end on snapshot
  `provider-import-vapi-retell-20260902-r11`.
- Retell code-source onboarding: 5/5 calls passed end to end on snapshot
  `provider-import-vapi-retell-20260902-r12`.
- Retell Conversation Flow import: 5/5 calls passed end to end on snapshot
  `provider-import-retell-conversation-flow-20260902-r14`.
- Vapi import and code-source flows reached Vapi successfully, but the provider rejected calls
  because the test wallet balance was negative. Vapi live-call success is intentionally outside
  this release gate until the account is funded.
- Provider credentials remained `target_provider` references. Future AGI-owned simulator
  credentials were platform-managed and did not travel through the job API.

## Retell code-source evidence

- Hosted job: `90bc6533-23ca-4622-a10e-8a0dcc529373`
- Harness run: `20ceb155-fd82-474a-91ee-0543991c7608`
- Platform test: `9249dbf2-6126-4796-a614-b12e77f60203`
- Platform execution: `c2d8377a-40c3-4870-9c4a-ce478de59bda`
- Calls: 5 passed, 0 failed, on infrastructure attempt 1
- Messages: 28 total; two calls correctly completed at five messages
- Audio: 20 recording artifacts, four per call
- Transcripts: 5, one per call
- Evals: 18/18 generated sub-goals held; platform rows contain 3–4 eval outputs per call
- Tools: 10 raw provider tool/event trace records persisted
- Cleanup: Daytona cleanup verified and the temporary Retell agent was deleted
- Secrets: no plaintext provider-key pattern appeared in persisted job JSON
- Regression suite: 941 complete harness tests passed, 1 skipped; all 45 targeted
  call-runner tests passed

## Completion-rule fix

Retell and Vapi own their end-call action. A complete agent-first clarification call can therefore
end after five alternating messages: agent greeting, caller request, agent clarification, caller
answer, then agent confirmation plus the provider end-call tool. The former shared six-message
minimum rejected that valid shape as `insufficient_conversation` before grading its tool evidence.

Provider-hosted connectors now accept five messages. Native LiveKit remains at six because the ALK
simulator owns that hang-up path. Regression tests assert both boundaries.

## Retell import evidence

- Hosted job: `65be5bf3-5704-47ee-8330-cf2245d46267`
- Harness run: `4a5dcb66-3a37-4fb5-844c-a891579e7bd1`
- Platform test: `8f18e2d3-60dc-4c8b-86ec-ea29bfb6a92e`
- Platform execution: `af6717a0-b647-4846-bf63-cb78ce76c332`
- Calls: 5 passed, 0 failed
- Audio/transcripts: 20 recording artifacts and 5 transcript artifacts
- Tools: 14 raw provider tool/event records
- Source immutability: the original Retell agent retained its original LLM, tool, and webhook
  configuration after the run

## Retell Conversation Flow import evidence

- Source agent: `agent_c5322ea1bb2d7f5cf661bbd4c4` (`conversation-flow` response engine)
- Hosted job: `fa087f27-f572-4e4b-9552-6e50958001a9`
- Harness run: `16944eca-9026-42d6-8827-abaf6fd2936f`
- Platform test: `085a0186-9e71-48d9-9d06-06270c568bf1`
- Platform execution: `226888cd-9bc9-408b-a419-b2fc4858521c`
- Calls: 5 passed, 0 failed, on infrastructure attempt 1
- Conversations: 61 turns total; durations populated for all five calls (92–203 seconds)
- Audio/transcripts: all five calls have four call-linked recordings and one transcript; all 24
  job recording artifacts, including validation evidence, were non-empty RIFF/WAVE files
- Tools: all four source Conversation Flow custom tools were cloned, recursively rewired to the
  isolated environment, invoked successfully, and persisted with arguments and results:
  `fetch_appointment_details`, `update_appointment`, `cancel_appointment`, and `create_booking`
- Evals: 13/13 generated sub-goals held and every platform call contains its eval outputs
- CSAT: completed for all five platform call rows
- Cleanup: no ALK-owned temporary Retell agents or Conversation Flows remained after the run
- Secrets: the Retell key remained a `target_provider` secret reference and no plaintext match
  appeared in persisted job output or source-controlled files
- Source immutability: the importer never updates the source agent or source Conversation Flow;
  a live clone/rewire/delete check also verified the source agent document was byte-canonically
  unchanged
- Regression suite: 1,239 harness and harness-integration tests passed, 1 skipped

The preceding r13 campaign completed the provider plumbing but exposed an authoring defect: one
scenario combined a correct out-of-scope transfer branch with a requirement to finish a booking
after the transfer. The call and artifacts completed, but that contradictory transactional check
failed. The scenario-writing contract now requires one coherent terminal outcome and forbids
combining escalation/refusal/end-call branches with downstream work they correctly prevent. The
fresh r14 campaign then passed 5/5.

## Vapi disposition

The fixed Vapi import job `b663402f-c77f-4dcd-a880-bb066088d6fb` produced an imported three-tool
contract and entered environment/scenario authoring through the unified hosted path. Earlier
five-call import and code-source campaigns reached Vapi, where every call was rejected because the
test account wallet balance was negative. The run was then canceled at the user's direction. No
additional ALK change is indicated by this evidence; fund the provider account before repeating
live-call certification.
