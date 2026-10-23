#!/bin/sh
set -eu
umask 077
/bin/chown -R --no-dereference root:root /usr/local
/bin/chmod -R go-w /usr/local
install -d -m 0700 /run/alk-docker /var/lib/alk-docker /var/lib/alk-compose
install -d -m 0700 /run/alk-docker/tmp
export HOME=/root TMPDIR=/run/alk-docker/tmp XDG_RUNTIME_DIR=/run/alk-docker
install -d -m 0750 -o root -g 2000 /run/alk-compose
dockerd --host unix:///run/alk-docker/docker.sock --group root \
  --data-root /var/lib/alk-docker --exec-root /run/alk-docker/exec \
  --pidfile /run/alk-docker/docker.pid > /run/alk-docker/daemon.log 2>&1 &
daemon=$!
trap 'kill "$daemon" 2>/dev/null || true' EXIT INT TERM
ready=false
for attempt in $(seq 1 60); do
  if docker -H unix:///run/alk-docker/docker.sock info >/dev/null 2>&1; then
    chmod 0600 /run/alk-docker/docker.sock
    ready=true
    break
  fi
  if ! kill -0 "$daemon" 2>/dev/null; then
    cat /run/alk-docker/daemon.log >&2
    exit 1
  fi
  sleep 1
done
"$ready"
timeout --signal=TERM --kill-after=60s 14400s \
  /opt/alk-venv/bin/python -m fi.alk.harness.hosted_compose_bootstrap
