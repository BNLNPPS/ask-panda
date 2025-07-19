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

import argparse
import logging
import os
import requests
import sys
from json import JSONDecodeError
from time import sleep

from ask_panda_server import MCP_SERVER_URL, check_server_health
from tools.errorcodes import EC_TIMEOUT

# mcp = FastMCP("panda") # Removed unused instance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("selection_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SelectionAgent:
    def __init__(self, agents: dict, model):
        self.agents = agents  # dict like {"document": ..., "queue": ...}
        self.model = model        # e.g., OpenAI or Anthropic wrapper

    def classify_question(self, question: str) -> str:
        prompt = f"""
You are a routing assistant for a question-answering system. Your job is to classify a question into one of the following categories, based on its topic:

- document: Questions about general usage, concepts, how-to guides, or explanation of systems (e.g. PanDA, prun, pathena, containers, error codes).
- queue: Questions about site or queue data stored in a JSON file (e.g. corepower, copytool, status of a queue, which queues use rucio).
- task: Questions about a specific task's status or job counts (e.g. status of task NNN, number of failed jobs).
- log_analyzer: Questions about why a specific job failed (e.g. log or failure analysis of job NNN).
- pilot_activity: Questions about pilot activity, failures, or statistics, possibly involving Grafana (e.g. pilots running on queue X, pilots failing, links to
Grafana).

Classify the following question:

"{question}"

Output only one of the categories: document, queue, task, log, or pilot.
"""
        result = self.ask(prompt).strip().lower()
        return result if result in self.agents else "document"

    def answer(self, question: str) -> str:
        return self.classify_question(question)

    def ask(self, question: str) -> str:
        """
        Send a question to the LLM via the MCP server and retrieve the answer.

        Args:
            question (str): The question to ask the LLM.

        Returns:
            str: The answer from theLLM. If an error occurs during the
            request, or if the server responds with an error, a string
            prefixed with "Error:" is returned detailing the issue.
        """
        server_url = os.getenv("MCP_SERVER_URL", f"{MCP_SERVER_URL}/rag_ask")

        # Construct prompt
        prompt = question

        try:
            response = requests.post(server_url, json={"question": prompt, "model": self.model}, timeout=30)
            if response.ok:
                try:
                    # Store interaction
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


def get_agents() -> dict:
    """
    Create and return a dictionary of agents for different categories.

    Returns:
        dict:
    """
    return {
        "document": "document_query_agent.py --question=QUESTION --model=MODEL --session-id=SESSION_ID",
        "queue": "queue_query_agent.py --question=QUESTION --model=MODEL --session-id=SESSION_ID",
        "task": "task_query_agent.py --question=QUESTION --model=MODEL --session-id=SESSION_ID",
        "log_analyzer": "log_analysis_agent.py --log-files LOG_FILES --pandaid PANDAID --model MODEL --mode MODE",
        "pilot_activity": "pilot_monitor_agent.py --question=QUESTION --model=MODEL --session-id=SESSION_ID"
    }


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
    parser.add_argument('--question', type=str, required=True,
                        help='The question to ask the RAG server')
    parser.add_argument('--model', type=str, required=True,
                        help='The model to use for generating the answer')
    parser.add_argument('--pandaid', type=str, required=False,
                        help='PanDA ID for the job or task, if applicable')
    args = parser.parse_args()

    agents = get_agents()
    selection_agent = SelectionAgent(agents, args.model)

    response = selection_agent.answer(args.question)
    logger.info(response)


if __name__ == "__main__":
    main()
