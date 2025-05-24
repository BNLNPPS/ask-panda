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

"""MCP server for RAG (Retrieval-Augmented Generation) using multiple LLMs."""

import os
import asyncio # httpx might need it, or other async operations
from fastapi import FastAPI
from pydantic import BaseModel
from fastmcp import FastMCP
import anthropic
from anthropic import AsyncAnthropic # Import AsyncAnthropic
import openai
from openai import AsyncOpenAI # Import AsyncOpenAI
import google.generativeai as genai
# import requests # No longer needed for LLaMA
import httpx # Import httpx
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

app = FastAPI()

# Set up API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:11434/api/generate")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Load vector store once at startup (same model used during vectorstore creation)
if False:
    embeddings = OpenAIEmbeddings()
else:
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True  # Safe if it's your own data
)

class PandaMCP(FastMCP):
    """PandaMCP class for handling RAG queries."""
    def rag_query(self, question: str, model: str) -> str:
        """
        Perform a similarity search on the vector store and retrieve relevant documents.

        Args:
            question (str): The input question to query the vector store.
            model (str): The model to use for generating the answer.

        Returns:
            str: The retrieved documents as a concatenated string.
        """
        docs = vectorstore.similarity_search(question, k=5)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"Answer based on the following context:\n{context}\n\nQuestion: {question}"

        if model == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API key is not set. Please set the ANTHROPIC_API_KEY environment variable.")
            try:
                client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY) # Use AsyncAnthropic
                completion = await client.messages.create( # await the call
                    model="claude-3-haiku-20240307",
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}]
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

        elif model == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
            try:
                # Instantiate AsyncOpenAI client
                client = AsyncOpenAI(api_key=OPENAI_API_KEY)
                # Use new syntax for chat completions and await
                completion = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512
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
            except openai.BadRequestError as e: # Covers what used to be InvalidRequestError
                return f"Error interacting with OpenAI API: Invalid request - {e}"
            except openai.APIError as e: # Base error for other API related issues
                return f"Error interacting with OpenAI API: {e}"
            except Exception as e:
                return f"An unexpected error occurred with OpenAI API: {e}"

        elif model == "llama":
            # No API key check needed for LLaMA as per requirements
            try:
                llama_payload = {"model": "llama3", "prompt": prompt, "stream": False}
                async with httpx.AsyncClient() as client: # Use httpx.AsyncClient
                llama_response = await client.post(LLAMA_API_URL, json=llama_payload, timeout=30.0) # await post, added timeout
                llama_response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
                return llama_response.json().get("response", "").strip()
            except httpx.HTTPStatusError as e: # Specific for HTTP errors like 4xx, 5xx
                return f"Error interacting with LLaMA API: HTTP error ({e.response.status_code}) - {e.response.text}"
            except httpx.TimeoutException as e: # Specific for timeouts
                return f"Error interacting with LLaMA API: Timeout - {e}"
            except httpx.RequestError as e: # Base for other request related errors (connection, etc.)
                return f"Error interacting with LLaMA API: Request error - {e}"
            # Catching standard json.JSONDecodeError if llama_response.json() fails
            except Exception as e: # Catch-all for other errors, e.g. JSONDecodeError if response.json() fails
                return f"An unexpected error occurred with LLaMA API: {e}"

        elif model == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError("Gemini API key is not set. Please set the GEMINI_API_KEY environment variable.")
            try:
                gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')
                response = await gemini_model.generate_content_async(prompt) # Use generate_content_async
                return response.text.strip()
            except genai.types.BlockedPromptException as e:
                return f"Error interacting with Gemini API: Prompt was blocked - {e}"
            except genai.types.StopCandidateException as e:
                return f"Error interacting with Gemini API: Content generation stopped unexpectedly - {e}"
            except genai.types.generation_types.BrokenResponseError as e: # More specific error for broken responses
                return f"Error interacting with Gemini API: Broken response - {e}"
            except genai.core.exceptions.GoogleAPIError as e: # Catching google.api_core.exceptions.GoogleAPIError
                return f"Error interacting with Gemini API: Google API error - {e}"
            except Exception as e:
                return f"An unexpected error occurred with Gemini API: {e}"

        raise ValueError(f"Unsupported model '{model}'.")

# Initialize the PandaMCP instance
mcp = PandaMCP("panda")

class QuestionRequest(BaseModel):
    """Pydantic model for the request body of the /rag_ask endpoint."""
    question: str
    model: str

@app.post("/rag_ask")
async def rag_ask(request: QuestionRequest) -> dict:
    """
    Handle a POST request to the /rag_ask endpoint.

    This endpoint receives a question and a model name, processes the query
    using the RAG (Retrieval-Augmented Generation) system, and returns the
    generated answer.

    Args:
        request (QuestionRequest): A Pydantic model containing the question
            and the model name.

    Returns:
        dict: A dictionary containing the generated answer.
    """
    response = await mcp.rag_query(request.question, request.model) # await the call
    return {"answer": response}

if __name__ == "__main__":
    # Run the FastAPI server
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
