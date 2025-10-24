#!/usr/bin/env bash
# Stop the Ask PanDA container started via run_askpanda.sh.

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-ask-panda}"

if docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  docker stop "${CONTAINER_NAME}"
else
  echo "Container '${CONTAINER_NAME}' is not running."
fi
