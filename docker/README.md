# Docker Deployment Guide

This directory contains Docker-related files for containerized deployment of the Ask PanDA FastAPI service.

## Files

- **`../Dockerfile`**: Container image definition (located in project root)
- **`run_askpanda.sh`**: Helper script to start the container with proper configuration
- **`stop_askpanda.sh`**: Helper script to stop the running container

## Prerequisites

- Docker installed and running
- API keys for at least one LLM provider (Anthropic, OpenAI, Gemini, or Mistral)
- PanDA authentication tokens (optional, for PanDA-specific features)

## Quick Start

### 1. Build the Docker Image

From the project root directory:

```bash
docker build -t ask-panda .
```

### 2. Configure Environment Variables

The `run_askpanda.sh` script uses environment variables for configuration. You can either:

**Option A**: Export them before running the script:

```bash
export ANTHROPIC_API_KEY='your_anthropic_api_key'
export OPENAI_API_KEY='your_openai_api_key'
export GEMINI_API_KEY='your_gemini_api_key'
export MISTRAL_API_KEY='your_mistral_api_key'
```

**Option B**: Edit `run_askpanda.sh` directly and set the default values (lines 22-26).

### 3. Start the Container

```bash
./docker/run_askpanda.sh
```

The service will be available at `http://localhost:8000`

### 4. Stop the Container

```bash
./docker/stop_askpanda.sh
```

## Configuration Options

The `run_askpanda.sh` script supports several environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_TAG` | `ask-panda` | Docker image tag to use |
| `CONTAINER_NAME` | `ask-panda` | Name for the running container |
| `CACHE_DIR` | `/tmp/ask-panda-cache` | Host directory for cache persistence |
| `CHROMA_DIR` | `${PROJECT_ROOT}/chromadb` | Host directory for ChromaDB data |
| `RESOURCES_DIR` | `${PROJECT_ROOT}/resources` | Host directory for resource files |
| `OIDC_AUTH_TOKEN_PATH` | `/secure/path/panda_primary.token` | Path to PanDA OIDC token |
| `PANDA_AUTH_TOKEN_KEY_PATH` | `/secure/path/panda.key` | Path to PanDA auth key |
| `OIDC_AUTH_ORIGIN` | `atlas.pilot` | OIDC authentication origin |
| `ANTHROPIC_API_KEY` | - | Anthropic (Claude) API key |
| `OPENAI_API_KEY` | - | OpenAI (GPT) API key |
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `MISTRAL_API_KEY` | - | Mistral AI API key |

### Example with Custom Configuration

```bash
export IMAGE_TAG="ask-panda:latest"
export CONTAINER_NAME="my-ask-panda"
export CACHE_DIR="/var/cache/ask-panda"
export ANTHROPIC_API_KEY="sk-ant-..."
./docker/run_askpanda.sh
```

## Volume Mounts

The container uses the following volume mounts:

- **`/app/cache`**: Stores cached data and downloaded files
- **`/app/chromadb`**: Persistent storage for the ChromaDB vector database
- **`/app/resources`**: Read-only mount for resource documents (optional)
- **`/tokens/`**: Read-only mount for PanDA authentication tokens (optional)

## Manual Docker Run

If you prefer not to use the helper scripts, you can run the container manually:

```bash
docker run --rm \
  --name ask-panda \
  -p 8000:8000 \
  -v /tmp/ask-panda-cache:/app/cache \
  -v ./chromadb:/app/chromadb \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
  -e MISTRAL_API_KEY="${MISTRAL_API_KEY}" \
  ask-panda
```

## Accessing the Service

Once the container is running, you can access the FastAPI service:

- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/

## Troubleshooting

### Container won't start

Check that:
- Port 8000 is not already in use: `lsof -i :8000`
- At least one API key is set
- Docker daemon is running

### Volume mount errors

Ensure the host directories exist and have proper permissions:

```bash
mkdir -p /tmp/ask-panda-cache
mkdir -p ./chromadb
mkdir -p ./resources
```

### View container logs

```bash
docker logs ask-panda
```

Or for continuous log streaming:

```bash
docker logs -f ask-panda
```

## Building for Production

For production deployments, consider:

1. **Using a specific tag**: `docker build -t ask-panda:1.0.0 .`
2. **Multi-stage builds**: Optimize image size if needed
3. **Health checks**: Add Docker health check configuration
4. **Resource limits**: Set memory and CPU limits
5. **Restart policy**: Use `--restart unless-stopped` for automatic restarts

Example production run:

```bash
docker run -d \
  --name ask-panda \
  --restart unless-stopped \
  -p 8000:8000 \
  -v /var/lib/ask-panda/cache:/app/cache \
  -v /var/lib/ask-panda/chromadb:/app/chromadb \
  --memory="2g" \
  --cpus="1.0" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  ask-panda:1.0.0
```

## Notes

- The Dockerfile is located in the project root directory (one level up from this docker/ directory)
- The container runs as a non-privileged user for security
- All Python dependencies are installed from `requirements.txt`
- The vector store is automatically created from documents in the `resources/` directory when the server starts
