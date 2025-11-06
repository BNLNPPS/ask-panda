#!/usr/bin/env bash
# Helper script to start the Ask PanDA container with common mounts and secrets.
#
# Fill in the environment variables below (or export them before calling)
# to point at your PanDA token files and API keys. The script defaults to
# the locally built image tag "ask-panda".

set -euo pipefail

# ---- User-configurable settings ----
IMAGE_TAG="${IMAGE_TAG:-ask-panda}"
CONTAINER_NAME="${CONTAINER_NAME:-ask-panda}"

# Location on the host where you want cached PanDA artefacts to persist
CACHE_DIR="${CACHE_DIR:-/tmp/ask-panda-cache}"

# Paths to the local token files (set to actual token locations)
OIDC_AUTH_TOKEN_PATH="${OIDC_AUTH_TOKEN_PATH:-/secure/path/panda_primary.token}"
PANDA_AUTH_TOKEN_KEY_PATH="${PANDA_AUTH_TOKEN_KEY_PATH:-/secure/path/panda.key}"
OIDC_AUTH_ORIGIN="${OIDC_AUTH_ORIGIN:-atlas.pilot}"

# API keys for LLM providers (set whichever you plan to use)
MISTRAL_API_KEY="${MISTRAL_API_KEY:-}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
GEMINI_API_KEY="${GEMINI_API_KEY:-}"

# ---- Derived paths ----
# Script is in docker/ subdirectory, so go up one level to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHROMA_DIR="${CHROMA_DIR:-${PROJECT_ROOT}/chromadb}"
RESOURCES_DIR="${RESOURCES_DIR:-${PROJECT_ROOT}/resources}"

# Ensure cache directory exists on the host
mkdir -p "${CACHE_DIR}"

# Assemble Docker run command
docker run --rm \
  --name "${CONTAINER_NAME}" \
  -p 8000:8000 \
  -v "${CACHE_DIR}:/app/cache" \
  -v "${CHROMA_DIR}:/app/chromadb" \
  -v "${RESOURCES_DIR}:/app/resources:ro" \
  -v "${OIDC_AUTH_TOKEN_PATH}:/tokens/panda_primary.token:ro" \
  -v "${PANDA_AUTH_TOKEN_KEY_PATH}:/tokens/panda.key:ro" \
  -e OIDC_AUTH_TOKEN="/tokens/panda_primary.token" \
  -e PANDA_AUTH_TOKEN_KEY="/tokens/panda.key" \
  -e OIDC_AUTH_ORIGIN="${OIDC_AUTH_ORIGIN}" \
  -e MISTRAL_API_KEY="${MISTRAL_API_KEY}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
  "${IMAGE_TAG}"
