---
kind: agent-learning.docs-page.v1
track: simulate
objective: capability
stage: simulate
backing:
  - examples/sdk_voice_loopback.py
artifact_kinds:
  - agent-learning.run.v1
commands:
  - python examples/sdk_voice_loopback.py artifacts/voice-loopback.json
postcondition: python -c "import json; p=json.load(open('artifacts/voice-loopback.json')); assert p['kind']=='agent-learning.voice-loopback.v1', p['kind']; print('ok')"
claims:
  - phrase: codec-survival
    gate_id: voice_loopback_readiness
  - phrase: audio-loopback
    gate_id: voice_loopback_readiness
doctor_checks:
  - missing_engine_modules
  - public_boundary_passed
opt_in_lane: true
---

# Voice loopback: the deterministic audio channel, credential-free

> **Twin:** [`examples/sdk_voice_loopback.py`](../../examples/sdk_voice_loopback.py)
> · emits `agent-learning.run.v1` · offline, no credentials, deterministic.
> A coding agent can complete this page from the frontmatter alone.

This is the rung-2 audio-loopback transport. It runs **entirely in-process** on
committed WAV fixtures — no sockets, no rooms, no OS audio devices, no keys —
and produces the two PCM streams (`user_pcm` + `agent_pcm`) that feed the
already-built dual-channel metrics engine. Same seed in, byte-identical PCM out.

**Honesty disclaimer (load-bearing).** A deterministic in-process loopback is
**NOT a live lane**. Every rung-2 artifact carries
`fidelity_tier: "deterministic_loopback"` and an evidence class of
`live_stressed` (or `captured_fixture` when stored) — **never `live_lane`**.
`live_lane` is reserved for the rung-3 keyed real-provider transport. The gate
fails any rung-2 artifact that claims `live_lane` (the
`loopback_fidelity_overclaim` tripwire).

## 1. What you are testing

Voice agents fail on the telephony channel, not just on words: a value that
survived clean audio but died through the codec, a barge-in the agent ignored,
a recovery turn that never landed. The rung-2 loopback exercises exactly those
signals deterministically. The default rung-2 run applies the codec round-trip
(G.711 μ-law @ 8 kHz + Gilbert-Elliott 2 %/100 ms) so a `channels` block AND a
computed `phone_survival` (`tier: channel_simulated`) always appear with zero
configuration. Opting out (`codec_profile: "none"`) is the explicit action — it
yields a clean-PCM loopback with a `channels` block but no `phone_survival`.

## 2. Run it

```bash
python examples/sdk_voice_loopback.py artifacts/voice-loopback.json
```

SDK (the operation the twin performs):

```python
from fi.alk.live import _loopback, _codec, _stats

loop = _loopback.run_loopback_roundtrip(turns, user_wav=user_wav, seed=1142)
u, a, rec = _codec.apply_codec_profile(
    loop["user_pcm"], loop["agent_pcm"], profile="g711_ulaw_8k_ge",
    seed=1142, sample_rate=24000,
)
channels = _stats.derive_channel_evidence(u, a, sample_rate=8000)
```

## 3. What you built

Postcondition (machine-checkable — same check the docs gate enforces):

```bash
python -c "import json; p=json.load(open('artifacts/voice-loopback.json')); assert p['kind']=='agent-learning.voice-loopback.v1', p['kind']; print('ok')"
```

The artifact holds the loopback determinism proof (byte-identical PCM + an
identical `channels` block under the pinned seed), the codec-survival
round-trip record, the rung-2 `channels` + computed `phone_survival`, and the
constructed overclaim negatives the gate must catch. An un-validated acoustic
claim carries the rung-1 pin `phone_survival: {"status": "untested", "tier":
"research_pinned"}` — no channel survival is claimed without a codec round-trip
record.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `LoopbackFixtureMissing` | config fault | a committed WAV fixture path is missing/unreadable |
| `loopback_fidelity_overclaim` in the gate | overclaim | a rung-2 artifact stamped `evidence_class: live_lane` |
| public boundary error | config fault | `summary.public_boundary_passed` |

## 5. Prove it / keep it

The twin runs fresh on every `agent-learn release-check` via the
`voice_loopback_readiness` gate (eight evidence arrays, all credential-free).
The codec-survival number is honest by computation: it is licensed only while
that gate is green. For the keyed real-provider rung-3 transport (the only
honest `live_lane`), see the roadmap — it is an owner-keyed opt-in lane, never a
release prerequisite. To tune the whole voice agent against these signals, see
[voice-improvement](../optimize/voice-improvement.md).
