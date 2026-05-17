#!/usr/bin/env bash
set -euo pipefail

PROFILE_FLAG="--profile hybrid"
COMPOSE_DIR="$(cd "$(dirname "$0")/../.." && pwd)/docker"
QDRANT_PORT="${QDRANT_PORT:-6333}"
KIWI_PORT="${KIWI_PORT:-8765}"

cmd="${1:-up}"

case "$cmd" in
  up)
    if ! command -v docker >/dev/null 2>&1; then
      echo "docker not found; cannot bring up hybrid stack" >&2
      exit 2
    fi
    docker run -d --rm --name qdrant-e2e -p ${QDRANT_PORT}:6333 qdrant/qdrant:v1.12.4
    (cd "$COMPOSE_DIR" && docker compose ${PROFILE_FLAG} up -d kiwi-service)
    for i in $(seq 1 30); do
      if curl -fsS "http://localhost:${KIWI_PORT}/healthz" >/dev/null 2>&1 \
         && curl -fsS "http://localhost:${QDRANT_PORT}/readyz" >/dev/null 2>&1; then
        echo "stack ready"
        exit 0
      fi
      sleep 1
    done
    echo "stack failed to become ready" >&2
    exit 1
    ;;
  down)
    docker stop qdrant-e2e >/dev/null 2>&1 || true
    if command -v docker >/dev/null 2>&1; then
      (cd "$COMPOSE_DIR" && docker compose ${PROFILE_FLAG} down) || true
    fi
    ;;
  *)
    echo "usage: $0 {up|down}" >&2
    exit 2
    ;;
esac
