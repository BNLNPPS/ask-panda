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
import re
import requests
import sys
from json import JSONDecodeError
from time import sleep

from agents.document_query_agent import DocumentQueryAgent
from agents.log_analysis_agent import LogAnalysisAgent
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


def get_agents(model: str, session_id: str or None, pandaid: str or None, taskid: str or None, cache: str) -> dict:
    """
    Create and return a dictionary of agents for different categories.

    Args:
        model (str): The model to use for generating answers.
        session_id (str or None): The session ID for the context memory.
        pandaid (str or None): The PanDA ID for the job or task, if applicable.
        taskid (str or None): The task ID for the job or task, if applicable.
        cache (str): The location of the cache directory.

    Returns:
        dict: A dictionary mapping agent categories to their respective agent classes.
    """
    return {
        "document": DocumentQueryAgent(model, session_id) if session_id else None,
        "queue": None,
        "task": None,
        "log_analyzer": LogAnalysisAgent(model, pandaid, cache) if pandaid else None,
        "pilot_activity": None
    }


def extract_job_id(text: str) -> int or None:
    """
    Extract a job ID from the given text using a regular expression.

    Args:
        text: The text from which to extract the job ID.

    Returns:
        int or None: The extracted job ID as an integer, or None if no job ID is found.
    """
    pattern = r'\b(?:job|panda[\s_]?id)\s+(\d+)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def extract_task_id(text: str) -> int or None:
    """
    Extract a task ID from the given text using a regular expression.

    Args:
        text: The text from which to extract the task ID.

    Returns:
        int or None: The extracted task ID as an integer, or None if no task ID is found.
    """
    pattern = r'\b(?:task[\s_]?id|task)\s+(\d+)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


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
    parser.add_argument('--cache', type=str, default="cache",
                        help='Location of cache directory (default: cache)')

    args = parser.parse_args()

    # does the question contain a job or task id?
    # use a regex to extract "job NNNNN" from args.question
    pandaid = extract_job_id(args.question)
    if pandaid is not None:
        logger.info(f"Extracted PanDA ID: {pandaid}")
    else:
        logger.info("No PanDA ID found in the question.")
    taskid = extract_task_id(args.question)
    if taskid is not None:
        logger.info(f"Extracted Task ID: {taskid}")
    else:
        logger.info("No Task ID found in the question.")

    agents = get_agents(args.model, args.session_id, pandaid, taskid, args.cache)
    selection_agent = SelectionAgent(agents, args.model)

    response = selection_agent.answer(args.question)
    logger.info(response)


if __name__ == "__main__":
    main()
