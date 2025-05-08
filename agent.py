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

"""This script is a simple command-line agent that interacts with a RAG (Retrieval-Augmented Generation) server."""

import requests
import sys
from fastmcp import FastMCP

mcp = FastMCP("panda")

def ask(question: str, model: str) -> str:
    """
    Send a question to the RAG server and retrieve the answer.

    Args:
        question (str): The question to ask the RAG server.
        model (str): The model to use for generating the answer (e.g., 'openai', 'anthropic').

    Returns:
        str: The answer returned by the RAG server.
    """
    server_url = "http://localhost:8000/rag_ask"
    response = requests.post(server_url, json={"question": question, "model": model})
    if response.ok:
        return response.json()["answer"]
    return f"Error: {response.text}"

def main():
    """
    Check if the correct number of command-line arguments is provided.

    This ensures that the script is executed with exactly two arguments:
    a question and a model.

    Raises:
        SystemExit: If the number of arguments is not equal to 3.
    """
    if len(sys.argv) != 3:
        print("Usage: python agent.py \"<question>\" <model>")
        sys.exit(1)

    question, model = sys.argv[1], sys.argv[2]
    answer = ask(question, model)
    print(f"Answer from {model.capitalize()} (via RAG):\n{answer}")

if __name__ == "__main__":
    main()
