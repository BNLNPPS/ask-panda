#!/bin/bash
# Test script for Docker setup

set -e

echo "=== Testing Docker Setup ==="
echo

echo "1. Checking if containers are running..."
docker compose ps
echo

echo "2. Testing Ask-PanDA server (port 8001)..."
curl -f http://localhost:8001/docs > /dev/null 2>&1 && \
    echo "✓ Ask-PanDA server is accessible" || \
    echo "✗ Ask-PanDA server is not accessible"
echo

echo "3. Testing Ollama shim (port 11435)..."
curl -f http://localhost:11435/api/version 2>/dev/null && \
    echo "✓ Ollama shim is accessible" || \
    echo "✗ Ollama shim is not accessible"
echo

echo "4. Testing Ollama shim /api/tags endpoint..."
curl -s http://localhost:11435/api/tags | python3 -m json.tool 2>/dev/null && \
    echo "✓ Ollama shim /api/tags works" || \
    echo "✗ Ollama shim /api/tags failed"
echo

echo "5. Checking Docker network..."
docker network inspect ask-panda-network > /dev/null 2>&1 && \
    echo "✓ Docker network 'ask-panda-network' exists" || \
    echo "✗ Docker network not found"
echo

echo "=== Testing Complete ==="
echo
echo "To use with Open WebUI, set:"
echo "  OLLAMA_BASE_URL=http://localhost:11435"
