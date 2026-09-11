#!/usr/bin/env bash
#
# Publish the hosted-harness Daytona snapshot from Dockerfile.hosted.
#
# The snapshot is the immutable BASE runtime, reused for every target agent:
# OS + interpreters (py 3.11/3.12, node 20/22) + the engine catalog
# (postgres/redis/rabbitmq) + the ALK guest baked at /opt/alk + the
# /run/futureagi and /work layout. A job uploads the *customer's* agent source
# into a fresh sandbox created FROM this snapshot; you do NOT cut a snapshot per
# agent. Cut a new snapshot only when the base runtime changes (interpreters,
# engines, Dockerfile/layout, or the ALK guest code) -- i.e. once per runtime
# release, never per customer.
#
# Usage:
#   DAYTONA_API_KEY=dtn_xxx scripts/daytona-snapshot.sh [SNAPSHOT_NAME]
#   SNAP_DRY_RUN=1 scripts/daytona-snapshot.sh v7      # validate only (no key needed)
#
# Environment:
#   DAYTONA_API_KEY      Daytona API key (required unless SNAP_DRY_RUN).
#   SNAPSHOT_NAME        Published name; also positional $1. Default: alk-hosted-v1.
#   DAYTONA_API_URL      Default: https://app.daytona.io/api
#   DAYTONA_TARGET       Optional Daytona target/region.
#   HOSTED_DOCKERFILE    Default: <repo>/Dockerfile.hosted
#   SNAP_CPU/SNAP_MEM/SNAP_DISK   vCPU / GiB RAM / GiB disk. Default: 4 / 8 / 10.
#   DAYTONA_SDK_VERSION  Default: 0.207.0
#   DAYTONA_VENV         Cached SDK venv. Default: <repo>/.venv-daytona
#   SNAP_DRY_RUN         If set, validate config + Dockerfile + SDK, then exit
#                        WITHOUT publishing (used by CI on PRs).
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/.." && pwd)"

SNAPSHOT_NAME="${1:-${SNAPSHOT_NAME:-alk-hosted-v1}}"
DAYTONA_API_URL="${DAYTONA_API_URL:-https://app.daytona.io/api}"
DAYTONA_TARGET="${DAYTONA_TARGET:-}"
HOSTED_DOCKERFILE="${HOSTED_DOCKERFILE:-$repo/Dockerfile.hosted}"
SNAP_CPU="${SNAP_CPU:-4}"
SNAP_MEM="${SNAP_MEM:-8}"
SNAP_DISK="${SNAP_DISK:-10}"
DAYTONA_SDK_VERSION="${DAYTONA_SDK_VERSION:-0.207.0}"
DAYTONA_VENV="${DAYTONA_VENV:-$repo/.venv-daytona}"

# A real publish needs the API key; a dry run (Dockerfile/SDK validation) does not.
if [ -z "${SNAP_DRY_RUN:-}" ]; then
  : "${DAYTONA_API_KEY:?DAYTONA_API_KEY is required (or set SNAP_DRY_RUN=1 to validate only)}"
fi

if [ ! -f "$HOSTED_DOCKERFILE" ]; then
  echo "error: Dockerfile not found: $HOSTED_DOCKERFILE" >&2
  exit 1
fi

# Cache a Python venv with the Daytona SDK (the SDK is an ops dependency, not an
# ALK runtime dependency, so it is not in pyproject).
# Prefer python3.12 or python3.11 over bare python3, which may be 3.9 on some
# systems and the Daytona SDK (>=0.207) requires >=3.10.
_find_python() {
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      local ver
      ver="$("$candidate" -c 'import sys; print(sys.version_info[:2]>=(3,10))' 2>/dev/null)"
      if [ "$ver" = "True" ]; then
        echo "$candidate"
        return
      fi
    fi
  done
  echo "error: no python >=3.10 found (daytona SDK requires it)" >&2
  exit 1
}

py="$DAYTONA_VENV/bin/python"
if [ ! -x "$py" ]; then
  _base_python="$(_find_python)"
  echo ">> creating Daytona SDK venv at $DAYTONA_VENV (using $_base_python)"
  "$_base_python" -m venv "$DAYTONA_VENV"
  "$py" -m pip install -q --upgrade pip
fi
if ! "$py" -c "import daytona" 2>/dev/null; then
  echo ">> installing daytona==$DAYTONA_SDK_VERSION"
  "$py" -m pip install -q "daytona==$DAYTONA_SDK_VERSION"
fi

echo ">> snapshot='$SNAPSHOT_NAME' dockerfile='$HOSTED_DOCKERFILE' cpu=$SNAP_CPU mem=${SNAP_MEM}Gi disk=${SNAP_DISK}Gi${SNAP_DRY_RUN:+ (dry run)}"

DAYTONA_API_KEY="${DAYTONA_API_KEY:-}" \
SNAPSHOT_NAME="$SNAPSHOT_NAME" \
DAYTONA_API_URL="$DAYTONA_API_URL" \
DAYTONA_TARGET="$DAYTONA_TARGET" \
HOSTED_DOCKERFILE="$HOSTED_DOCKERFILE" \
SNAP_CPU="$SNAP_CPU" SNAP_MEM="$SNAP_MEM" SNAP_DISK="$SNAP_DISK" \
SNAP_DRY_RUN="${SNAP_DRY_RUN:-}" \
"$py" - <<'PY'
import os
import sys

from daytona import CreateSnapshotParams, Daytona, DaytonaConfig, Image, Resources

name = os.environ["SNAPSHOT_NAME"]
dockerfile = os.environ["HOSTED_DOCKERFILE"]
cpu = int(os.environ["SNAP_CPU"])
mem = int(os.environ["SNAP_MEM"])
disk = int(os.environ["SNAP_DISK"])

# Parse the Dockerfile + build context (validates the path/COPY graph early).
img = Image.from_dockerfile(dockerfile)

if os.environ.get("SNAP_DRY_RUN"):
    print(f"DRY_RUN ok: would publish '{name}' (cpu={cpu} mem={mem}Gi disk={disk}Gi)")
    sys.exit(0)

d = Daytona(
    DaytonaConfig(
        api_key=os.environ["DAYTONA_API_KEY"],
        api_url=os.environ.get("DAYTONA_API_URL") or None,
        target=os.environ.get("DAYTONA_TARGET") or None,
    )
)
try:
    snap = d.snapshot.create(
        CreateSnapshotParams(
            name=name,
            image=img,
            resources=Resources(cpu=cpu, memory=mem, disk=disk),
        ),
        on_logs=lambda chunk: print(chunk, end="", flush=True),
    )
except Exception as exc:  # noqa: BLE001
    print(f"\nSNAPSHOT_ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    sys.exit(1)

digest = ""
for attr in ("image_digest", "digest", "image_ref", "sha256", "image"):
    value = getattr(snap, attr, None)
    if value:
        digest = str(value)
        break

summary = os.environ.get("GITHUB_STEP_SUMMARY")
lines = [
    "---",
    f"SNAPSHOT_NAME={snap.name}",
    f"SNAPSHOT_STATE={getattr(snap, 'state', None)}",
    f"SNAPSHOT_DIGEST={digest}",
    "---",
]
print("\n" + "\n".join(lines))
print("Pin in the platform .env, then recreate the simulation-runner worker:")
print(f"  ALK_DAYTONA_SNAPSHOT={snap.name}")
if digest:
    print(f"  ALK_DAYTONA_SNAPSHOT_DIGEST={digest}")
else:
    print("  ALK_DAYTONA_SNAPSHOT_DIGEST=  # SDK exposed no digest field")
    print(f"  # snapshot object: {snap!r}")

if summary:
    with open(summary, "a", encoding="utf-8") as handle:
        handle.write(f"### Daytona snapshot published\n\n- **name**: `{snap.name}`\n")
        handle.write(f"- **state**: `{getattr(snap, 'state', None)}`\n")
        handle.write(f"- **digest**: `{digest or '(none exposed)'}`\n")
PY
