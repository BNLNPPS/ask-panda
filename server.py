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
from fastapi import FastAPI
from pydantic import BaseModel
from fastmcp import FastMCP
import anthropic
import openai
import google.generativeai as genai
import requests
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()

# Set up API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLAMA_API_URL = os.getenv("LLAMA_API_URL", "http://localhost:11434/api/generate")

openai.api_key = OPENAI_API_KEY
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
    def rag_query(self, question: str, model: str) -> str:
        docs = vectorstore.similarity_search(question, k=5)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"Answer based on the following context:\n{context}\n\nQuestion: {question}"

        if model == "anthropic":
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            completion = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.content[0].text.strip()

        elif model == "openai":
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512
            )
            return completion.choices[0].message.content.strip()

        elif model == "llama":
            llama_payload = {"model": "llama3", "prompt": prompt, "stream": False}
            llama_response = requests.post(LLAMA_API_URL, json=llama_payload)
            llama_response.raise_for_status()
            return llama_response.json().get("response", "").strip()

        elif model == "gemini":
            gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')
            response = gemini_model.generate_content(prompt)
            return response.text.strip()

        raise ValueError(f"Unsupported model '{model}'.")

mcp = PandaMCP("panda")

class QuestionRequest(BaseModel):
    question: str
    model: str

@app.post("/rag_ask")
async def rag_ask(request: QuestionRequest):
    response = mcp.rag_query(request.question, request.model)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
