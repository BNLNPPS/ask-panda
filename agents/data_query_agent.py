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

"""This agent can download task metadata from PanDA and ask an LLM to analyze the relevant parts."""

import argparse
import ast
import asyncio
import logging
import os
import re
import requests
import sys
from collections import deque

from fastmcp import FastMCP
from time import sleep

from tools.errorcodes import EC_NOTFOUND, EC_OK, EC_UNKNOWN_ERROR, EC_TIMEOUT
from ask_panda_server import MCP_SERVER_URL, check_server_health
from tools.tools import fetch_data, read_json_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("data_query_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

mcp = FastMCP("panda")


class TaskStatusAgent:
    """
    A simple agent that can give information about task status.
    This agent fetches metadata from PanDA, extracts relevant parts, and asks an LLM for analysis.
    """
    def __init__(self, model: str, taskid: str, cache: str, session_id: str) -> None:
        """
        Initialize the TaskStatusAgent with a model.

        Args:
            model (str): The model to use for generating the answer (e.g., 'openai', 'anthropic').
            taskid (str): The PanDA job ID to analyze.
            cache (str): The location of the cache directory for storing downloaded files.
            session_id (str): The session ID for tracking the conversation.
        """
        self.model = model  # e.g., OpenAI or Anthropic wrapper
        try:
            self.taskid = int(taskid)  # PanDA task ID for the analysis
        except ValueError:
            logger.error(f"Invalid task ID: {taskid}. It should be an integer.")
            sys.exit(1)
        self.session_id = session_id
        self.cache = cache
        if not os.path.exists(self.cache):
            logger.info(f"Cache directory {self.cache} does not exist. Creating it.")
            try:
                os.makedirs(self.cache)
            except OSError as e:
                logger.error(f"Failed to create cache directory {self.cache}: {e}")
                sys.exit(1)

        path = os.path.join(os.path.join(self.cache, "tasks"), str(taskid))
        if not os.path.exists(path):
            logger.info(f"Creating directory for task {taskid} in cache.")
            try:
                os.makedirs(path)
            except OSError as e:
                logger.error(f"Failed to create directory {path}: {e}")
                sys.exit(1)

    def ask(self, question: str) -> str:
        """
        Send a question to the LLM and retrieve the answer.

        Args:
            question (str): The question to ask the LLM.

        Returns:
            str: The answer returned by the MCP server.
        """
        server_url = f"{MCP_SERVER_URL}/rag_ask"
        response = requests.post(server_url, json={"question": question, "model": self.model})
        if response.ok:
            answer = response.json()["answer"]
            if not answer:
                err = "No answer returned from the LLM."
                logger.error(f"{err}")
                return {'error': f"{err}"}

            # Strip code block formatting
            try:
                clean_code = re.sub(r"^```(?:python)?\n|\n```$", "", answer.strip())
            except re.error as e:
                logger.error(f"Regex error while cleaning code: {e}")
                return {'error': f"Regex error: {e}"}

            # convert the answer to a Python dictionary
            try:
                answer_dict = ast.literal_eval(clean_code)
            except (SyntaxError, ValueError) as e:
                err = f"Error converting answer to dictionary: {e}"
                logger.error(f"{err}")
                return {'error': f"{err}"}

            if not answer_dict:
                err = "Failed to store the answer as a Python dictionary."
                logger.error(f"{err}")
                return {'error': f"{err}"}

            return answer_dict

        return {'error': f"requests.post() error: {response.text}"}

    async def fetch_all_data(self) -> tuple[int, dict or None, dict or None]:
        """
        Fetch metadata from PanDA for a given task ID.

        Returns:
            Exit code (int): The exit code indicating the status of the operation.
            File dictionary (dict): A dictionary containing the file names and their corresponding paths.
            Metadata dictionary (dict): A dictionary containing the relevant metadata for the task.
        """
        _metadata_dictionary = {}
        _file_dictionary = {}

        # Download metadata and pilot log concurrently
        workdir = os.path.join(self.cache, "tasks")
        url = f"https://bigpanda.cern.ch/jobs/?jeditaskid={self.taskid}&json&mode=nodrop"
        metadata_task = asyncio.create_task(fetch_data(self.taskid, filename="metadata.json", jsondata=True, workdir=workdir, url=url))

        # Wait for download to complete
        metadata_success, metadata_message = await metadata_task
        if metadata_success != 0:
            logger.warning(f"Failed to fetch metadata for task {self.taskid} - will not be able to analyze the task status")
            return EC_NOTFOUND, _file_dictionary, _metadata_dictionary

        logger.info(f"Downloaded JSON file: {metadata_message}")
        _file_dictionary["json"] = metadata_message

        task_data = read_json_file(metadata_message)
        if not task_data:
            logger.warning(f"Error: Failed to read the JSON data from {metadata_message}.")
            return EC_UNKNOWN_ERROR, None, None

        # Extract relevant metadata from the JSON data
        try:
            _metadata_dictionary["jobs"] = {}
            for job in task_data['jobs']:
                jobstatus = job.get('jobstatus', 'unknown')
                if jobstatus not in _metadata_dictionary["jobs"]:
                    _metadata_dictionary["jobs"][jobstatus] = 0
                _metadata_dictionary["jobs"][jobstatus] += 1

                for key in job:
                    if "errordiag" in key:
                        if "errordiags" not in _metadata_dictionary:
                            _metadata_dictionary["errordiags"] = deque()
                        if job[key] != "":
                            _metadata_dictionary["errordiags"].append(job[key])
                    if "errorcode" in key:
                        if "errorcodes" not in _metadata_dictionary:
                            _metadata_dictionary["errorcodes"] = deque()
                        if job[key] > 0:
                            _metadata_dictionary["errorcodes"].append(job[key])
        except KeyError as e:
            logger.warning(f"Error: Missing key in JSON data: {e}")
            return EC_UNKNOWN_ERROR, None, None

        return EC_OK, _file_dictionary, _metadata_dictionary

    def formulate_question(self, metadata_dictionary: dict) -> str:
        """
        Construct a question to ask the LLM based on the extracted lines and metadata.

        Args:
            metadata_dictionary:

        Returns:
            str: A formatted question string to be sent to the LLM.
        """
        jobs = metadata_dictionary.get("jobs", None)
        if not jobs:
            logger.warning("Error: No jobs information found in the metadata dictionary.")
            return ""

        question = "You are an expert on distributed analysis. A PanDA task is either not start, in progress or has completed (finished or failed).\n\n"
        question += """
Please provide a summary of the task status based on the metadata provided. The task is identified by its PanDA TaskID.
The dictionary should have the task id as the key (an integer), and its value should include the following fields:

"description": A short summary of the task status in plain English. If there are no jobs, state that the task has not started yet.

"problems": A plain-language explanation of any issues (job failures as listed in the dictionary).

Return only a valid Python dictionary. Here's the metadata dictionary:
        """
        question = question.replace("TaskID", f"task ID ({self.taskid})")
        description = str(metadata_dictionary)
        question += f"\n\n{description}\n\n"

        return question

    def generate_question(self) -> str:
        """
        Generate a question to ask the LLM based on the task metadata.

        Returns:
            str: A formatted question string to be sent to the LLM.
        """
        # Fetch the files from PanDA
        exit_code, file_dictionary, metadata_dictionary = asyncio.run(self.fetch_all_data())
        logger.info(f"metadata_dictionary: {metadata_dictionary}")

        if exit_code == EC_NOTFOUND:
            logger.warning(
                f"No metadata found for task {self.taskid}")
        elif not file_dictionary:
            logger.warning(f"Error: Failed to metadata files for PandaID {self.taskid}.")
            sys.exit(1)

        # Formulate the question based on the extracted lines and metadata
        question = self.formulate_question(metadata_dictionary)
        if not question:
            logger.warning("No question could be generated.")
            sys.exit(1)

        return question


def main():
    """
    Check if the correct number of command-line arguments is provided.

    This ensures that the script is executed with exactly two arguments:
    a question and a model.

    Raises:
        SystemExit: If the number of arguments is not equal to 4.
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

    parser.add_argument('--taskid', type=int, required=True,
                        help='PanDA TaskID (integer)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use (e.g., openai, anthropic, etc.)')
    parser.add_argument('--cache', type=str, default="cache",
                        help='Location of cache directory (default: cache)')
    parser.add_argument('--session-id', type=str, required=True,
                        help='Session ID for the context memory')
    args = parser.parse_args()

    agent = TaskStatusAgent(args.model, args.taskid, args.cache)

    # Generate a proper question to ask the LLM based on the metadata and log files
    question = agent.generate_question()
    logger.info(f"Asking question: \n\n{question}")

    # Ask the question to the LLM
    answer = agent.ask(question)

    logger.info(f"Answer from {args.model.capitalize()}:\n{answer}")

    # store the answer in the session memory
    # ..

    sys.exit(0)


if __name__ == "__main__":
    main()
