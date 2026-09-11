#!/bin/sh
set -eu
city="$(psql "$DATABASE_URL" -tAc 'SELECT city FROM calls ORDER BY id LIMIT 1')"
echo "AGENT_READY postgres=${city}"
while true; do sleep 60; done
