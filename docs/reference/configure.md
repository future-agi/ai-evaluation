---
kind: agent-learning.docs-page.v1
track: reference
backing: []
artifact_kinds: []
commands:
  - agent-learn doctor
postcondition: python -c "from fi.alk.config import API_KEY_ENV_NAMES; assert API_KEY_ENV_NAMES[0] == 'AGENT_LEARNING_API_KEY'; print('ok')"
claims: []
doctor_checks:
  - api_key_configured
opt_in_lane: false
---

# Configuration Reference

> **Twin:** none — reference page (`backing: []`). All semantics below are
> read directly from `src/fi/alk/config.py`.

## 1. What you are testing

The kit runs in two modes. **Offline** is the default: every cookbook backing
example, every golden path, and the whole `docs_executability` gate run with
no environment variables set and no credentials. **Platform** mode adds a
Future AGI API key so platform-backed evaluation and reporting can attach;
nothing about the local artifact contract changes.

Configuration is one frozen dataclass (`AgentLearningConfig`) read from the
environment at import time and adjustable at runtime with `configure()`. One
key is the intended setup: `AGENT_LEARNING_API_KEY`. The legacy aliases exist
for compatibility, and precedence is positional — the first non-empty name in
each tuple wins.

## 2. Run it

```bash
export AGENT_LEARNING_API_KEY="..."   # optional — omit for offline mode
agent-learn doctor
```

```python
from fi.alk import configure
from fi.alk.config import current_config, get_api_key

configure(api_key="...")          # optional override of AGENT_LEARNING_API_KEY
print(current_config().api_url)   # https://api.futureagi.com by default
print(bool(get_api_key()))        # False in offline mode — and that is fine
```

`doctor` reports the result without printing the key:
`config.api_key_configured` and `summary.api_key_configured` are booleans.

## 3. What you built

```bash
python -c "from fi.alk.config import API_KEY_ENV_NAMES; assert API_KEY_ENV_NAMES[0] == 'AGENT_LEARNING_API_KEY'; print('ok')"
```

Alias precedence, exactly as coded (first non-empty value wins):

| Setting | Precedence order | Default |
| --- | --- | --- |
| API key | `AGENT_LEARNING_API_KEY` → `FUTURE_AGI_API_KEY` → `FI_API_KEY` | unset (offline) |
| Secret key | `AGENT_LEARNING_SECRET_KEY` → `FUTURE_AGI_SECRET_KEY` → `FI_SECRET_KEY` | falls back to the API key |
| API URL | `AGENT_LEARNING_API_URL` → `FUTURE_AGI_API_URL` | `https://api.futureagi.com` |
| Project id | `AGENT_LEARNING_PROJECT_ID` → `FUTURE_AGI_PROJECT_ID` | unset |
| Workspace id | `AGENT_LEARNING_WORKSPACE_ID` → `FUTURE_AGI_WORKSPACE_ID` | unset |

Behavior worth knowing before you wire CI:

- `configure(api_key=...)` also sets the secret key to the same value unless
  you pass `secret_key` explicitly.
- After `configure()` (and at import), `_sync_env` writes the resolved values
  back to **all** alias names in `os.environ`, so vendored engine code reading
  `FI_API_KEY` and new code reading `AGENT_LEARNING_API_KEY` see one value.
- `get_api_key(required=True)` raises
  `RuntimeError: Missing Future AGI API key. Set one of:
  AGENT_LEARNING_API_KEY, FUTURE_AGI_API_KEY, FI_API_KEY.` — commands that
  need the platform fail with that named-variable message rather than a stack
  of HTTP errors.
- The environment is read once at import; export variables before launching
  Python or the CLI, or call `configure()` afterwards.

## 4. When it fails

| Symptom | First-mile class | Doctor check |
| --- | --- | --- |
| `RuntimeError: Missing Future AGI API key. Set one of: ...` | keys — a platform-backed step ran in offline mode | `agent-learn doctor` → `summary.api_key_configured` |
| key exported but `api_key_configured` is `false` | config fault — set after import, or empty string (empty values are skipped) | rerun `doctor` in the shell that exported the key |
| two different keys behave inconsistently | config fault — a higher-precedence alias is shadowing; check the table order | `config.api_key_configured` plus `env \| grep -E 'AGENT_LEARNING\|FUTURE_AGI\|FI_'` |

## 5. Prove it / keep it

Offline mode is not a degraded mode — it is the release contract: the
verification ladder runs every golden path in a clean temp directory with no
environment variables set, and the docs gate executes fresh-lane backing
examples with environment save/restore. Keep your CI job key-free unless a
page explicitly requires platform mode, and start with the
[run golden path](../quickstart/golden-path-run.md). The command surface that
consumes this configuration is cataloged in [reference/cli.md](cli.md).
