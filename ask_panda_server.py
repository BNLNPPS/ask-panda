# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Authors:
# - Paul Nilsson, paul.nilsson@cern.ch, 2025

"""
FastAPI server for RAG using multiple LLMs.

This script sets up and runs a FastAPI application that provides an endpoint
(`/rag_ask`) for performing Retrieval-Augmented Generation (RAG) queries.
It supports multiple Large Language Models (LLMs) such as Anthropic's Claude,
OpenAI's GPT, a local LLaMA instance, and Google's Gemini.

The server initializes API keys from environment variables, loads a FAISS
vector store for context retrieval, and defines a PandaMCP class to handle
the RAG logic and interaction with the different LLMs.
"""

# Set up basic logging configuration at the top of your file
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("ask_panda_server.log"),
        logging.StreamHandler()  # Optional: keeps logs visible in console too
    ]
)
logger = logging.getLogger(__name__)

import anthropic
import asyncio
import google.generativeai as genai
import httpx  # Import httpx
import openai
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastmcp import FastMCP
from mistralai import Mistral
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from pydantic import BaseModel
from typing import Optional  # For type hinting

from tools.tools import get_vectorstore_manager, Timer

_MISTRAL_CONCURRENCY = asyncio.Semaphore(4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event handler to initialize resources.

    Args:
        app: FastAPI instance to attach the lifespan handler to.
    """
    global vectorstore_manager, mcp

    resources_dir = Path("resources")
    chroma_dir = Path("chromadb")

    vectorstore_manager = get_vectorstore_manager(resources_dir, chroma_dir)
    if not vectorstore_manager:
        logger.error("Failed to initialize VectorStoreManager.")
        raise RuntimeError("VectorStoreManager initialization failed.")

    vectorstore_manager.start_periodic_updates()

    mcp = PandaMCP("panda", resources_dir, chroma_dir, vectorstore_manager)

    logger.info("FastAPI startup complete. Vector store and MCP initialized successfully.")
    yield

# FastAPI instance
app = FastAPI(lifespan=lifespan)

# Declare global references (initialized later)
vectorstore_manager = None
mcp = None

# Set up API keys from environment variables
ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
LLAMA_API_URL: Optional[str] = os.getenv(
    "LLAMA_API_URL", "http://localhost:11434/api/generate"
)
MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")

# Configure SDKs - these operations might implicitly use the API keys above
# or might be for libraries that don't require explicit key passing at every call.
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY  # Still useful for some openai direct calls if any
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# vectorstore: FAISS = FAISS.load_local(
#    "vectorstore",
#    embeddings,  # type: ignore # FAISS.load_local expects specific embedding types, ignore for flexibility.
#    allow_dangerous_deserialization=True,  # Safe if it's your own data
# )


class PandaMCP(FastMCP):
    """
    Panda Multi-Cloud Platform (MCP) for RAG queries.

    This class extends FastMCP to provide Retrieval-Augmented Generation (RAG)
    capabilities by querying various Large Language Models (LLMs) based on
    context retrieved from a vector store. It initializes with a name
    passed to the FastMCP base class.
    """

    def __init__(
            self,
            namespace: str,
            resources_dir: Path,
            vectorstore_dir: Path,
            vectorstore_manager: vectorstore_manager
    ):
        """
        Initializes the PandaMCP class.

        Args:
            namespace (str): Namespace identifier for MCP.
            resources_dir (Path): Directory containing resource documents.
            vectorstore_dir (Path): Directory for ChromaDB storage.
            vectorstore_manager: Instance managing vectorstore operations.
        """
        super().__init__(namespace)

        self.resources_dir = resources_dir
        self.vectorstore_dir = vectorstore_dir
        self.vectorstore_manager = vectorstore_manager

        self._mistral_client = None
        self._mistral_lock = asyncio.Lock()

    async def _get_mistral_client(self) -> Mistral:
        """
        Lazily initialize and return the Mistral client.

        Returns:
            Mistral: An instance of the Mistral client.
        """
        if self._mistral_client is not None:
            return self._mistral_client

        async with self._mistral_lock:
            if self._mistral_client is None:
                self._mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
                await self._mistral_client.__aenter__()

        return self._mistral_client

    async def _call_mistral_old(self, prompt: str) -> str:
        """
        Use the official Mistral SDK for cleaner, more reliable API calls.
        """
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return "Error: MISTRAL_API_KEY not set in environment."

        t = Timer("mistral")
        try:
            async with Mistral(api_key=api_key) as client:
                res = await client.chat.complete_async(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )
                t.done()
                return res.choices[0].message.content.strip()
        except Exception as e:
            return f"Mistral SDK error: {str(e)}"

    async def _call_mistral(self, prompt: str) -> str:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            return "Error: MISTRAL_API_KEY not set in environment."

        system_msg = (
            "You are a helpful assistant. "
        )

        async with _MISTRAL_CONCURRENCY:
            tries, backoff = 3, 1.0
            last_err = None
            for _ in range(tries):
                try:
                    client = await self._get_mistral_client()
                    # If your mistralai version exposes client-level httpx settings,
                    # they’ll be used here; otherwise it uses sane defaults.
                    res = await client.chat.complete_async(
                        model="mistral-large-latest",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt},
                        ],
                        stream=False,
                    )
                    return res.choices[0].message.content.strip()
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(backoff)
                    backoff *= 2
            return f"Mistral SDK error after retries: {last_err!s}"

    async def _call_anthropic(self, prompt: str) -> str:
        """
        Call the Anthropic API to get a response for the given prompt.

        Args:
            prompt (str): The prompt to send to the Anthropic model.

        Returns:
            str: The text response from the Anthropic model, or an error message
                 if the API call fails or the API key is missing.

        Raises:
            ValueError: If the ANTHROPIC_API_KEY environment variable is not set.
        """
        if not ANTHROPIC_API_KEY:
            raise ValueError(  # noqa: E501
                "Anthropic API key is not set. "
                "Please set the ANTHROPIC_API_KEY environment variable."
            )
        try:
            client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            completion = await client.messages.create(  # await the call
                model="claude-3-haiku-20240307",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.content[0].text.strip()
        except anthropic.APIConnectionError as e:
            # More specific connection error
            return f"Error interacting with Anthropic API: Connection error - {e}"
        except anthropic.RateLimitError as e:
            # Specific rate limit error
            return f"Error interacting with Anthropic API: Rate limit exceeded - {e}"
        except anthropic.APIStatusError as e:
            # Specific API status error (e.g. 400, 500)
            return f"Error interacting with Anthropic API: API status error ({e.status_code}) - {e.message}"
        except anthropic.APIError as e:
            # General Anthropic API error
            return f"Error interacting with Anthropic API: {e}"
        except Exception as e:
            # Catch any other unexpected errors
            return f"An unexpected error occurred with Anthropic API: {e}"

    async def _call_openai(self, prompt: str) -> str:
        """
        Call the OpenAI API to get a response for the given prompt.

        Args:
            prompt (str): The prompt to send to the OpenAI model.

        Returns:
            str: The text response from the OpenAI model, or an error message
                 if the API call fails or the API key is missing.

        Raises:
            ValueError: If the OPENAI_API_KEY environment variable is not set.
        """
        if not OPENAI_API_KEY:
            raise ValueError(  # noqa: E501
                "OpenAI API key is not set. "
                "Please set the OPENAI_API_KEY environment variable."
            )
        try:
            # Instantiate AsyncOpenAI client
            client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
            # Use new syntax for chat completions and await
            completion = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return completion.choices[0].message.content.strip()
        # Note: OpenAI v1.x error types might differ, ensure these are correct or update as needed.
        # OpenAI v1.x uses exceptions directly from the openai module
        except openai.AuthenticationError as e:
            return f"Error interacting with OpenAI API: Authentication failed - {e}"
        except openai.RateLimitError as e:
            return f"Error interacting with OpenAI API: Rate limit exceeded - {e}"
        except openai.APIConnectionError as e:
            return f"Error interacting with OpenAI API: Connection error - {e}"
        except (
            openai.BadRequestError
        ) as e:  # Covers what used to be InvalidRequestError
            return f"Error interacting with OpenAI API: Invalid request - {e}"
        except openai.APIError as e:  # Base error for other API related issues
            return f"Error interacting with OpenAI API: {e}"
        except Exception as e:
            return f"An unexpected error occurred with OpenAI API: {e}"

    async def _call_llama(self, prompt: str) -> str:
        """
        Call the LLaMA API (via a local server) to get a response for the given prompt.

        This method uses the LLAMA_API_URL environment variable to connect to the
        LLaMA model server. No direct API key is typically required for this setup,
        but the server must be accessible.

        Args:
            prompt (str): The prompt to send to the LLaMA model.

        Returns:
            str: The text response from the LLaMA model, or an error message
                 if the API call fails.
        """
        # No API key check needed for LLaMA as per requirements
        try:
            llama_payload = {"model": "llama3", "prompt": prompt, "stream": False}
            async with httpx.AsyncClient() as client:  # Use httpx.AsyncClient
                llama_response = await client.post(
                    LLAMA_API_URL, json=llama_payload, timeout=120.0
                )  # await post, added timeout
            llama_response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            return llama_response.json().get("response", "").strip()
        except httpx.HTTPStatusError as e:  # Specific for HTTP errors like 4xx, 5xx
            error_message = (  # noqa: E501
                f"Error interacting with LLaMA API: HTTP error "
                f"({e.response.status_code}) - {e.response.text}"
            )
            return error_message
        except httpx.TimeoutException as e:  # Specific for timeouts
            return f"Error interacting with LLaMA API: Timeout - {e}"
        except (
            httpx.RequestError
        ) as e:  # Base for other request related errors (connection, etc.)
            return f"Error interacting with LLaMA API: Request error - {e}"
        # Catching standard json.JSONDecodeError if llama_response.json() fails
        except (
            Exception
        ) as e:  # Catch-all for other errors, e.g. JSONDecodeError if response.json() fails
            return f"An unexpected error occurred with LLaMA API: {e}"

    async def _call_gemini(self, prompt: str) -> str:
        """
        Call the Google Gemini API to get a response for the given prompt.

        Args:
            prompt (str): The prompt to send to the Gemini model.

        Returns:
            str: The text response from the Gemini model, or an error message
                 if the API call fails or the API key is missing.

        Raises:
            ValueError: If the GEMINI_API_KEY environment variable is not set.
        """
        if not GEMINI_API_KEY:
            raise ValueError(  # noqa: E501
                "Gemini API key is not set. "
                "Please set the GEMINI_API_KEY environment variable."
            )
        try:
            gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
            response = await gemini_model.generate_content_async(
                prompt
            )  # Use generate_content_async
            return response.text.strip()
        except genai.types.BlockedPromptException as e:
            return f"Error interacting with Gemini API: Prompt was blocked - {e}"
        except genai.types.StopCandidateException as e:
            return f"Error interacting with Gemini API: Content generation stopped unexpectedly - {e}"  # noqa: E501
        except (
            genai.types.generation_types.BrokenResponseError
        ) as e:  # More specific error for broken responses
            return f"Error interacting with Gemini API: Broken response - {e}"
        except (
            genai.core.exceptions.GoogleAPIError
        ) as e:  # Catching google.api_core.exceptions.GoogleAPIError
            return f"Error interacting with Gemini API: Google API error - {e}"
        except Exception as e:
            return f"An unexpected error occurred with Gemini API: {e}"

    async def rag_query(self, question: str, model: str) -> str:
        """
        Perform a RAG query: retrieve context, then call the specified LLM.

        This method first performs a similarity search on the vector store
        using the provided question to retrieve relevant document snippets.
        These snippets are then combined with the original question to form
        a detailed prompt, which is dispatched to the appropriate LLM
        helper method based on the 'model' parameter.

        Args:
            question (str): The input question to query the vector store and
                            subsequently the LLM.
            model (str): The identifier of the LLM to use for generating the
                         answer (e.g., "anthropic", "openai", "llama", "gemini").

        Returns:
            str: The answer generated by the selected LLM based on the
                 retrieved context and question, or an error message if issues
                 occur during the process.

        Raises:
            ValueError: If an unsupported model identifier is provided.
        """
        context_docs = vectorstore_manager.query(question, k=5)
        context = "\n\n".join(context_docs)

        # Construct prompt explicitly
        prompt = f"Answer based on the following context:\n{context}\n\nQuestion: {question}"

        # Call appropriate LLM based on provided model
        if model == "anthropic":
            return await self._call_anthropic(prompt)
        if model == "openai":
            return await self._call_openai(prompt)
        if model == "llama":
            return await self._call_llama(prompt)
        if model == "gemini":
            return await self._call_gemini(prompt)
        if model == "mistral":
            return await self._call_mistral(prompt)

        return f"Invalid model specified: '{model}'."


class QuestionRequest(BaseModel):
    """
    Models the request body for the `/rag_ask` endpoint.

    Attributes:
        question (str): The question to be answered by the RAG system.
        model (str): The identifier of the LLM to use for generating the answer.
    """

    question: str
    model: str


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health-check endpoint."""
    logger.debug("Health check complete.")
    return {"status": "ok"}


@app.post("/rag_ask")
async def rag_ask(request: QuestionRequest) -> dict[str, str]:
    """
    Handle POST requests to the `/rag_ask` endpoint.

    This endpoint receives a question and a model name, processes the query
    using the PandaMCP's RAG (Retrieval-Augmented Generation) system,
    and returns the generated answer.

    Args:
        request (QuestionRequest): The request body, conforming to the
                                   QuestionRequest model, containing the user's
                                   question and the desired LLM model.

    Returns:
        Dict[str, str]: A dictionary containing the generated answer,
                        structured as `{"answer": "..."}`.
    """
    logger.info(f"Received query: '{request.question}' using model: '{request.model}'")

    response_text = await mcp.rag_query(request.question, request.model.lower())

    logger.info(f"Query processed using model '{request.model.lower()}'.")
    return {"answer": response_text}
