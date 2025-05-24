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

import os

# import asyncio  # No longer explicitly needed after httpx integration
from fastapi import FastAPI
from pydantic import BaseModel
from fastmcp import FastMCP
import anthropic
from anthropic import AsyncAnthropic  # Import AsyncAnthropic
import openai
from openai import AsyncOpenAI  # Import AsyncOpenAI
import google.generativeai as genai

# import requests # No longer needed for LLaMA
import httpx  # Import httpx
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, Optional, Union  # For type hinting

app: FastAPI = FastAPI()

# Set up API keys from environment variables
ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
LLAMA_API_URL: Optional[str] = os.getenv(
    "LLAMA_API_URL", "http://localhost:11434/api/generate"
)

# Configure SDKs - these operations might implicitly use the API keys above
# or might be for libraries that don't require explicit key passing at every call.
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY  # Still useful for some openai direct calls if any
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Define type for embeddings before assignment
embeddings: Union[OpenAIEmbeddings, HuggingFaceEmbeddings]

# Load vector store once at startup (same model used during vectorstore creation)
if False:  # This block is currently not executed, kept for potential future use
    embeddings = OpenAIEmbeddings()
else:
    model_name: str = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

vectorstore: FAISS = FAISS.load_local(
    "vectorstore",
    embeddings,  # type: ignore # FAISS.load_local expects specific embedding types, ignore for flexibility.
    allow_dangerous_deserialization=True,  # Safe if it's your own data
)


class PandaMCP(FastMCP):
    """
    Panda Multi-Cloud Platform (MCP) for RAG queries.

    This class extends FastMCP to provide Retrieval-Augmented Generation (RAG)
    capabilities by querying various Large Language Models (LLMs) based on
    context retrieved from a vector store. It initializes with a name
    passed to the FastMCP base class.
    """

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
            client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)  # Use AsyncAnthropic
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
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
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
                    LLAMA_API_URL, json=llama_payload, timeout=30.0
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
            gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
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
        docs = vectorstore.similarity_search(question, k=5)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = (  # noqa: E501
            "Answer based on the following context:\n"
            f"{context}\n\n"
            f"Question: {question}"
        )

        if model == "anthropic":
            return await self._call_anthropic(prompt)
        elif model == "openai":
            return await self._call_openai(prompt)
        elif model == "llama":
            return await self._call_llama(prompt)
        elif model == "gemini":
            return await self._call_gemini(prompt)
        else:
            raise ValueError(  # noqa: E501
                f"Unsupported model '{model}'. Ensure this is the final else."
            )


# Initialize the PandaMCP instance
mcp: PandaMCP = PandaMCP("panda")


class QuestionRequest(BaseModel):
    """
    Models the request body for the `/rag_ask` endpoint.

    Attributes:
        question (str): The question to be answered by the RAG system.
        model (str): The identifier of the LLM to use for generating the answer.
    """

    question: str
    model: str


@app.post("/rag_ask")
async def rag_ask(request: QuestionRequest) -> Dict[str, str]:
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
    response = await mcp.rag_query(request.question, request.model)  # await the call
    return {"answer": response}


if __name__ == "__main__":
    # Run the FastAPI server
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
