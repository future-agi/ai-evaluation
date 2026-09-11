---
kind: agent-learning.docs-page.v1
track: redteam
objective: safety
stage: simulate
backing:
  - examples/sdk_voice_redteam_campaign.py
artifact_kinds:
  - agent-learning.redteam.v1
commands:
  - agent-learn redteam-corpus --corpus examples/redteam_corpus.json --output artifacts/voice-corpus.json
postcondition: python -c "import json; p=json.load(open('artifacts/voice-corpus.json')); assert p['kind']=='agent-learning.redteam.v1', p['kind']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: false
---

# Voice attack corpus: the voice channel joins the gated matrix

> **Twin:** [`examples/sdk_voice_redteam_campaign.py`](../../examples/sdk_voice_redteam_campaign.py)
> · emits `agent-learning.redteam.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

> **Authorization boundary.** Voice red-team campaigns run only against agents
> the user owns or is explicitly authorized to test — the same boundary as
> every red-team surface in the kit. Voice attacks never target third parties;
> telephony rungs are owner-keyed; all examples and fixtures run against
> kit-owned local agents.

## 1. What you are testing

The gate-enforced red-team corpus used to execute on `chat` only. Phase 12 adds
`voice` as a second corpus channel, so
`V1_REDTEAM_CORPUS_EXECUTION_CHANNELS` is `["chat", "voice"]` and the corpus
gates assert voice coverage automatically. Each voice row carries BOTH a
semantic `surface` (one of the frozen six — `instruction`, `tool`, `memory`,
`retrieval`, `environment`, `long_context`) AND an orthogonal `voice_surface`
(one of `asr_front_end`, `diarization`, `vad_boundary`, `silence_region`,
`homophone_divergence`, `stored_voice`). The six semantic surfaces stay frozen;
the voice surface is a refinement axis on top of the existing matrix.

Every voice row references an attack family from
`V1_VOICE_ATTACK_FAMILY_MATRIX`, the honest family table that records each
family's maturity, the structured `phone_survival` object, its defended-by
notes, and whether it is expressible at rung-1.

## 2. Run it

CLI:

```bash
agent-learn redteam-corpus --corpus examples/redteam_corpus.json \
  --output artifacts/voice-corpus.json
```

SDK, same operation:

```python
import json

from fi.alk import redteam

rows = json.load(open("examples/redteam_corpus.json"))["rows"]
campaign = redteam.build_redteam_corpus_campaign(
    name="redteam-corpus-campaign",
    corpus_rows=rows,
)
voice_rows = [r for r in rows if r["channel"] == "voice"]
assert len(voice_rows) == 12
```

## 3. What you built

Postcondition (machine-checkable — the same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/voice-corpus.json')); assert p['kind']=='agent-learning.redteam.v1', p['kind']; print('ok')"
```

The artifact reports the grown channel set `["chat", "voice"]`, 24 corpus rows
(12 chat + 12 voice), and one covered, executed, mitigated cell per row.

The phone-survival honesty discipline is load-bearing. Ultrasonic carrier
families are pinned `phone_survival.status: "dies"` with
`scope_label: "smart_speaker_only"` — a real attack against a smart speaker,
but never counted as phone coverage for a SIP agent. At rung-1 every attack
instance carries the pin `phone_survival: {"status": "untested", "tier":
"research_pinned"}`; an instance never claims channel survival without a codec
round-trip record, which arrives only with the Phase-9A loopback rung.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `missing_channels: ["voice"]` in a corpus gate | real gap | a voice row is missing or the channel constant was not grown |
| `corpus_errors` on `voice_surface` | config fault | a voice row's `voice_surface` is not one of the six |
| hook returns no rows | config fault | `agent-learn doctor` → `summary.public_boundary_passed` |

## 5. Prove it / keep it

A voice corpus that passes today is a baseline. Wire the same command into CI so
new voice rows must arrive with their dual-field shape, a valid family token,
and the rung-1 phone-survival pin. The composed persona × signal search over
these rows is documented in
[voice-composed-campaigns](voice-composed-campaigns.md).
