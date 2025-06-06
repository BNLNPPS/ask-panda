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

import logging
import os  # Added for environment variable access
import requests
import sys
from json import JSONDecodeError  # Added for specific exception handling
# from fastmcp import FastMCP # Removed unused import

import errorcodes
from server import MCP_SERVER_URL, check_server_health

# mcp = FastMCP("panda") # Removed unused instance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def ask(question: str, model: str) -> str:
    """
    Send a question to the RAG server and retrieve the answer.

    The server URL is determined by the `RAG_SERVER_URL` environment
    variable, defaulting to `"http://localhost:8000/rag_ask"` if not set.
    The request to the server includes a 30-second timeout.

    Args:
        question (str): The question to ask the RAG server.
        model (str): The model to use for generating the answer
                     (e.g., 'openai', 'anthropic').

    Returns:
        str: The answer from the RAG server. If an error occurs during the
             request, or if the server responds with an error, a string
             prefixed with "Error:" is returned detailing the issue.
    """
    server_url = os.getenv("MCP_SERVER_URL", f"{MCP_SERVER_URL}/rag_ask")
    try:
        response = requests.post(server_url, json={"question": question, "model": model}, timeout=30)
        if response.ok:
            try:
                return response.json()["answer"]
            except JSONDecodeError:  # Changed to use imported JSONDecodeError
                return "Error: Could not decode JSON response from server."
            except KeyError:
                return "Error: 'answer' key missing in server response."
        else:
            try:
                # Attempt to parse JSON for detailed error message
                error_data = response.json()
                if isinstance(error_data, dict) and "detail" in error_data:
                    return f"Error from server: {error_data['detail']}"
                # Fallback if "detail" key is not found or JSON is not a dict
                return f"Error: Server returned status {response.status_code} - {response.text}"
            except JSONDecodeError:  # Changed to use imported JSONDecodeError
                # Fall through to the generic error message if JSON parsing fails
                pass
            # Fallback if JSON parsing fails or "detail" is not in a dict
            return f"Error: Server returned status {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Error: Network issue or server unreachable - {e}"

def main() -> None:
    """
    Parse command-line arguments, call the RAG server, and print the response.

    This function serves as the main entry point for the command-line agent.
    It expects two arguments: the question to ask and the model to use.
    It calls the `ask` function to get a response from the RAG server.
    If the `ask` function returns an error (a string prefixed with "Error:"),
    this error is printed to `sys.stderr` and the script exits with status 1.
    Otherwise, the successful answer is printed to `sys.stdout`.

    Raises:
        SystemExit: If the number of command-line arguments is incorrect, or
                    if an error occurs during the RAG server request.
    """
    # Check server health before proceeding
    ec = check_server_health()
    if ec == errorcodes.EC_TIMEOUT:
        logger.warning(f"Timeout while trying to connect to {MCP_SERVER_URL}.")
        os.sleep(5)  # Wait for a while before retrying
        ec = check_server_health()
        if ec:
            logger.error("MCP server is not healthy after retry. Exiting.")
            sys.exit(1)
    elif ec:
        logger.error("MCP server is not healthy. Exiting.")
        sys.exit(1)

    if len(sys.argv) != 3:
        logger.info("Usage: python agent.py \"<question>\" <model>")
        sys.exit(1)

    question, model = sys.argv[1], sys.argv[2]
    answer = ask(question, model)
    if answer.startswith("Error:"):
        logger.info(answer, file=sys.stderr)
        sys.exit(1)
    else:
        logger.info(f"Answer from {model.capitalize()} (via RAG):\n{answer}")

if __name__ == "__main__":
    main()
