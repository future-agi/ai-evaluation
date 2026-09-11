---
kind: agent-learning.docs-page.v1
track: prove
objective: safety
stage: promote
backing:
  - examples/sdk_redteam_readiness_certification_optimization.py
  - examples/sdk_workspace_import_certification_optimization.py
  - examples/framework_certification_optimization.json
artifact_kinds:
  - agent-learning.suite.v1
  - agent-learning.optimization.v1
commands:
  - for key in $(python -c "import json; print(' '.join(json.load(open('examples/agent_learning_suite.json'))['required_env']))"); do export "$key=local-offline"; done
  - agent-learn suite examples/agent_learning_suite.json --output artifacts/suite.json
  - agent-learn trust artifacts/suite.json --output trust-verification.json --quiet
  - AGENT_LEARNING_FRAMEWORK_CERT_OPT_EXAMPLE_KEY=local-offline agent-learn optimize examples/framework_certification_optimization.json --output artifacts/framework-certification.json
postcondition: python -c "import json; p=json.load(open('trust-verification.json')); assert p['kind']=='agent-learning.suite.trust-verification.v1', p['kind']; assert p['status']=='passed', p['status']; print('ok')"
claims: []
doctor_checks:
  - missing_engine_modules
opt_in_lane: false
---

# Trust Certificates: promotion verdicts you can verify later

> **Twin:** [`examples/sdk_redteam_readiness_certification_optimization.py`](../../examples/sdk_redteam_readiness_certification_optimization.py)
> · emits `agent-learning.optimization.v1` · offline, no credentials.
> A coding agent can complete this page from the frontmatter alone.

## 1. What you are testing

A green pipeline run is a fact about the past; promotion is a decision about
the future. The gap between the two is where teams get hurt: the artifact that
passed is not the one being promoted, or the pass happened under conditions
nobody recorded. The kit's answer is to make the verdict part of the artifact.
Every suite run embeds a `trust_certificate` block
(`agent-learning.suite.trust-certificate.v1`) with a `verdict` of `approved`,
`conditional`, or `rejected`, a `promotion_ready` flag, and an
`assurance_level`. `agent-learn trust` then verifies a saved certificate
without re-running anything — a deploy job can gate on the verdict in
milliseconds, long after the suite ran.

Certification optimizations produce the deeper, domain-specific proof blocks.
The three backing twins each certify a different surface:
[`framework_certification_optimization.json`](../../examples/framework_certification_optimization.json)
scores lifecycle, capability, probe, and portability evidence for a framework
adapter and attaches a `framework_certification_proof` block with per-check
counts; the red-team readiness twin certifies that a workspace's framework
targets can be imported, invoked, and attacked across declared surfaces before
anyone trusts a red-team pass; the workspace import twin certifies repository
provenance (`repository_url`, `commit_sha`) and import evidence for every
declared target. All three run offline against scripted agents.

## 2. Run it

Produce a suite artifact, then verify its certificate without re-running:

```bash
for key in $(python -c "import json; print(' '.join(json.load(open('examples/agent_learning_suite.json'))['required_env']))"); do export "$key=local-offline"; done

agent-learn suite examples/agent_learning_suite.json --output artifacts/suite.json

agent-learn trust artifacts/suite.json \
  --output trust-verification.json --quiet

AGENT_LEARNING_FRAMEWORK_CERT_OPT_EXAMPLE_KEY=local-offline \
agent-learn optimize examples/framework_certification_optimization.json \
  --output artifacts/framework-certification.json
```

Relative outputs resolve against your current working directory, so
`artifacts/suite.json` and `trust-verification.json` land where you run the
commands. By default `trust` requires verdict
`approved` and `promotion_ready: true`; relax with `--allow-conditional` or
`--no-require-promotion-ready` where your policy permits.

The same verification from the SDK:

```python
from fi.alk import suite

verdict = suite.verify_trust_certificate_file("artifacts/suite.json")
assert verdict["status"] == "passed", verdict["findings"]
```

## 3. What you built

Postcondition (machine-checkable — same shape the docs gate enforces):

```bash
python -c "import json; p=json.load(open('trust-verification.json')); assert p['kind']=='agent-learning.suite.trust-verification.v1', p['kind']; assert p['status']=='passed', p['status']; print('ok')"
```

The verification artifact records `observed_verdict`, `promotion_ready`,
`assurance_level`, the full certificate copy, and a `findings[]` array that is
empty on pass and names the exact failure type
(`suite_trust_certificate_verdict_too_low`,
`suite_trust_certificate_not_promotion_ready`) otherwise. The certification
optimization artifact carries its proof block plus summary counts
(`framework_certification_proof_passed`, `..._check_count`,
`..._failed_check_count`) your CI can assert on directly.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `vendored import failed` | infra | `agent-learn doctor` → `summary.missing_engine_modules` |
| `suite_trust_certificate_missing` finding | wrong input artifact | point `trust` at a suite artifact, not a child result |
| `verdict_too_low` (`conditional` observed) | evidence gap in the suite run | read `trust_certificate.conditions` in the suite artifact, fix the named job |
| `missing required environment variable(s)` | config fault | export the manifest's `required_env` keys with placeholder values |

## 5. Prove it / keep it

Make `agent-learn trust <suite artifact>` the last step before any promotion:
it is cheap enough to run on every deploy and strict by default. Archive the
verification artifact next to the build it licensed — six months later the
question "why did we ship this" has a machine-readable answer. The suite that
produces certificates is documented in [trinity-suite](trinity-suite.md); the
kit applies the same discipline to itself in
[release-check-in-your-ci](release-check-in-your-ci.md).
