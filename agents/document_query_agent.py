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

import argparse
import logging
import os
import requests
import sys
from json import JSONDecodeError
from time import sleep

from ask_panda_server import MCP_SERVER_URL, check_server_health
from tools.context_memory import ContextMemory
from tools.errorcodes import EC_TIMEOUT

# mcp = FastMCP("panda") # Removed unused instance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("document_query_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
memory = ContextMemory()


def ask(question: str, model: str, session_id: str) -> str:
    """
    Send a question to the RAG server and retrieve the answer.

    The server URL is determined by the `RAG_SERVER_URL` environment
    variable, defaulting to `"http://localhost:8000/rag_ask"` if not set.
    The request to the server includes a 30-second timeout.

    Args:
        question (str): The question to ask the RAG server.
        model (str): The model to use for generating the answer
                     (e.g., 'openai', 'anthropic').
        session_id (str): The session ID for tracking the conversation.

    Returns:
        str: The answer from the RAG server. If an error occurs during the
             request, or if the server responds with an error, a string
             prefixed with "Error:" is returned detailing the issue.
    """
    server_url = os.getenv("MCP_SERVER_URL", f"{MCP_SERVER_URL}/rag_ask")

    # Retrieve context
    history = memory.get_history(session_id)

    # Construct prompt
    prompt = ""
    for user_msg, agent_msg in history:
        prompt += f"User: {user_msg}\nAssistant: {agent_msg}\n"
    prompt += f"User: {question}\nAssistant:"

    try:
        response = requests.post(server_url, json={"question": prompt, "model": model}, timeout=30)
        if response.ok:
            try:
                # Store interaction
                s = response.json()["answer"]
                memory.store_turn(session_id, question, s)
                return s
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
    if ec == EC_TIMEOUT:
        logger.warning(f"Timeout while trying to connect to {MCP_SERVER_URL}.")
        sleep(10)  # Wait for a while before retrying
        ec = check_server_health()
        if ec:
            logger.error("MCP server is not healthy after retry. Exiting.")
            sys.exit(1)
    elif ec:
        logger.error("MCP server is not healthy. Exiting.")
        sys.exit(1)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument('--session-id', type=str, required=True,
                        help='Session ID for the context memory')
    parser.add_argument('--question', type=str,
                        help='The question to ask the RAG server')
    parser.add_argument('--model', type=str,
                        help='The model to use for generating the answer')
    args = parser.parse_args()

    if len(sys.argv) != 4:
        logger.info("Usage: python document_query_agent.py \"<question>\" <model> <session-id>")
        sys.exit(1)

    answer = ask(args.question, args.model, args.session_id)
    if answer.startswith("Error:"):
        logger.info(answer, file=sys.stderr)
        sys.exit(1)
    else:
        answer_dict = {
            "session_id": args.session_id,
            "question": args.question,
            "model": args.model,
            "answer": answer
        }
        logger.info(f"Answer:\n{answer_dict}")


if __name__ == "__main__":
    main()
