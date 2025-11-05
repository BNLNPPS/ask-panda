#!/usr/bin/env python3
"""
ollama_shim.py
──────────────────────────────────────────────────────────────────────
Minimal Ollama-compatible façade for Ask-PanDA.

This FastAPI application exposes the handful of endpoints that Open WebUI
expects from an Ollama server. Instead of talking to a local Ollama daemon,
each request is forwarded to the Ask-PanDA HTTP API, which in turn uses the
Mistral API (or any other LLM provider configured inside Ask-PanDA).

Current Configuration:
    - Backend: Ask-PanDA RAG endpoint (`/rag_ask`)
    - LLM: Mistral API (open-mistral-7b model)
    - Model exposed: mistral-proxy:latest
    - Default port: 11434 (11435 in Docker to avoid conflicts)

Endpoints implemented:
    GET  /api/tags      → model listing with :latest tag support
    GET  /api/ps        → running models list
    GET  /api/version   → version information
    POST /api/chat      → translate chat requests to Ask-PanDA RAG
    POST /api/generate  → translate generate requests to Ask-PanDA RAG

Environment Variables:
    ASK_PANDA_BASE_URL       - Ask-PanDA server URL (default: http://localhost:8000)
    OLLAMA_SHIM_MODEL        - Backend model name (default: mistral)
    OLLAMA_SHIM_MODEL_DISPLAY - Display name for Open WebUI (default: mistral-proxy)
    OLLAMA_SHIM_PORT         - Port to listen on (default: 11434)
    OLLAMA_SHIM_VERSION      - Version string (default: v0.0-shim)
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Ask-PanDA Ollama Shim", version="0.1.0")

ASK_PANDA_BASE_URL = os.getenv("ASK_PANDA_BASE_URL", "http://localhost:8000")
# Use rag_ask for both endpoints - agent_ask has timeout issues with complex processing
ASK_PANDA_AGENT_ENDPOINT = f"{ASK_PANDA_BASE_URL.rstrip('/')}/rag_ask"
ASK_PANDA_RAG_ENDPOINT = f"{ASK_PANDA_BASE_URL.rstrip('/')}/rag_ask"

OLLAMA_SHIM_MODEL = os.getenv("OLLAMA_SHIM_MODEL", "mistral")
OLLAMA_SHIM_MODEL_DISPLAY = os.getenv(
    "OLLAMA_SHIM_MODEL_DISPLAY", f"{OLLAMA_SHIM_MODEL}-proxy"
)
OLLAMA_VERSION = os.getenv("OLLAMA_SHIM_VERSION", "v0.0-shim")


def _utcnow_iso() -> str:
    return _dt.datetime.utcnow().isoformat() + "Z"


def _normalize_model_name(model_name: Optional[str]) -> str:
    """
    Strip the :latest or other tags from model names for Ask-PanDA compatibility.
    Converts 'mistral-proxy:latest' -> 'mistral' using the configured base model.
    """
    if not model_name:
        return OLLAMA_SHIM_MODEL

    # Strip any tag (e.g., :latest)
    base_name = model_name.split(":")[0]

    # If it matches our display name, use the configured backend model
    if base_name == OLLAMA_SHIM_MODEL_DISPLAY.split(":")[0]:
        return OLLAMA_SHIM_MODEL

    return base_name


def _extract_user_prompt(messages: List[Dict[str, Any]]) -> str:
    """
    Collapse an Ollama `messages` list into a single text prompt.

    The simplest workable strategy (sufficient for the shim) is to concatenate
    the conversation using `role:` prefixes. The last user message is the one
    we send to Ask-PanDA, prefixed with short context for continuity.
    """
    if not messages:
        return ""

    history_lines = [f"{msg.get('role', 'unknown')}: {msg.get('content', '')}".strip()
                     for msg in messages[:-1]]
    last_msg = messages[-1]
    last_user = last_msg.get("content", "")

    history_prefix = "\n".join(history_lines).strip()
    if history_prefix:
        combined = f"{history_prefix}\nuser: {last_user}"
    else:
        combined = last_user
    return combined.strip()


async def _call_ask_panda(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            logger.debug("Posting to %s with payload %s", endpoint, payload)
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        msg = f"Ask-PanDA returned {exc.response.status_code}: {exc.response.text}"
        logger.error(msg)
        raise HTTPException(status_code=exc.response.status_code, detail=msg)
    except httpx.RequestError as exc:
        msg = f"Failed to reach Ask-PanDA: {exc}"
        logger.error(msg)
        raise HTTPException(status_code=502, detail=msg)

    try:
        return response.json()
    except ValueError:
        msg = "Ask-PanDA response was not valid JSON"
        logger.error(msg)
        raise HTTPException(status_code=502, detail=msg)


@app.get("/api/tags")
async def list_models() -> Dict[str, Any]:
    """
    Mirror Ollama's /api/tags to satisfy Open WebUI model discovery.
    """
    # Ensure model name includes :latest tag for compatibility
    model_name = OLLAMA_SHIM_MODEL_DISPLAY
    if ":" not in model_name:
        model_name = f"{model_name}:latest"

    payload = {
        "models": [
            {
                "name": model_name,
                "model": model_name,
                "modified_at": _utcnow_iso(),
            }
        ]
    }
    return payload


@app.get("/api/ps")
async def list_running_models() -> Dict[str, Any]:
    """
    Report a fake running model list. This is primarily for UI niceties.
    """
    # Ensure model name includes :latest tag for compatibility
    model_name = OLLAMA_SHIM_MODEL_DISPLAY
    if ":" not in model_name:
        model_name = f"{model_name}:latest"

    payload = {
        "models": [
            {
                "name": model_name,
                "model": model_name,
                "digest": "shim",
                "size": 0,
                "details": {},
                "expires_at": None,
            }
        ]
    }
    return payload


@app.get("/api/version")
async def version() -> Dict[str, Any]:
    """
    Provide a synthetic version string.
    """
    return {"version": OLLAMA_VERSION}


@app.post("/api/chat")
async def chat(request: Request) -> JSONResponse:
    """
    Emulate Ollama's /api/chat endpoint by forwarding the last user prompt to
    Ask-PanDA's agent endpoint.
    """
    data: Dict[str, Any] = await request.json()

    messages: Optional[List[Dict[str, Any]]] = data.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    prompt = _extract_user_prompt(messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="Could not extract user prompt")

    # Normalize model name (strip :latest tag, map to backend model)
    requested_model = data.get("model")
    model_id = _normalize_model_name(requested_model)

    ask_payload = {"question": prompt, "model": model_id}
    ask_response = await _call_ask_panda(ASK_PANDA_AGENT_ENDPOINT, ask_payload)

    answer = ask_response.get("answer", "")
    response_body = {
        "model": requested_model or OLLAMA_SHIM_MODEL_DISPLAY,
        "created_at": _utcnow_iso(),
        "message": {"role": "assistant", "content": answer},
    }

    return JSONResponse(response_body)


@app.post("/api/generate")
async def generate(request: Request) -> JSONResponse:
    """
    Minimal support for Ollama's /api/generate endpoint (prompt-based).
    """
    data: Dict[str, Any] = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # Normalize model name (strip :latest tag, map to backend model)
    requested_model = data.get("model")
    model_id = _normalize_model_name(requested_model)

    ask_payload = {"question": prompt, "model": model_id}
    ask_response = await _call_ask_panda(ASK_PANDA_RAG_ENDPOINT, ask_payload)

    answer = ask_response.get("answer", "")
    response_body = {
        "model": requested_model or OLLAMA_SHIM_MODEL_DISPLAY,
        "created_at": _utcnow_iso(),
        "response": answer,
        "done": True,
    }
    return JSONResponse(response_body)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("OLLAMA_SHIM_PORT", "11434"))
    uvicorn.run("ollama_shim:app", host="0.0.0.0", port=port, reload=False)
