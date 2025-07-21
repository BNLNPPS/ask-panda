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

"""This agent can download a log file from PanDA and ask an LLM to analyze the relevant parts."""

import argparse
import ast
import asyncio
import logging
import os
import re
import requests
import sys
from collections import deque

# from docutils.nodes import description
from fastmcp import FastMCP
from time import sleep

from tools.errorcodes import EC_NOTFOUND, EC_OK, EC_UNKNOWN_ERROR, EC_TIMEOUT
from ask_panda_server import MCP_SERVER_URL, check_server_health
from tools.tools import fetch_data, read_json_file, read_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("log_analysis_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

mcp = FastMCP("panda")


class LogAnalysisAgent:
    """
    A simple command-line agent that interacts with a RAG server to analyze log files.
    This agent fetches log files from PanDA, extracts relevant parts, and asks an LLM for analysis.
    """
    def __init__(self, model: str, pandaid: str, cache: str) -> None:
        """
        Initialize the LogAnalysisAgent with a model.

        Args:
            model (str): The model to use for generating the answer (e.g., 'openai', 'anthropic').
            pandaid (str): The PanDA job ID to analyze.
            cache (str): The location of the cache directory for storing downloaded files.
        """
        self.model = model  # e.g., OpenAI or Anthropic wrapper
        try:
            self.pandaid = int(pandaid)  # PanDA job ID for the analysis
        except ValueError:
            logger.error(f"Invalid PanDA ID: {pandaid}. It should be an integer.")
            sys.exit(1)

        self.cache = cache
        if not os.path.exists(self.cache):
            logger.info(f"Cache directory {self.cache} does not exist. Creating it.")
            try:
                os.makedirs(self.cache)
            except OSError as e:
                logger.error(f"Failed to create cache directory {self.cache}: {e}")
                sys.exit(1)

        path = os.path.join(os.path.join(self.cache, "jobs"), str(self.pandaid))
        if not os.path.exists(path):
            logger.info(f"Creating directory for PandaID {self.pandaid} in cache.")
            try:
                os.makedirs(path)
            except OSError as e:
                logger.error(f"Failed to create directory {path}: {e}")
                sys.exit(1)

    def ask(self, question: str) -> str:
        """
        Send a question to the LLMr and retrieve the answer.

        Args:
            question (str): The question to ask the RAG server.

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

    async def fetch_all_data(self, log_file: str) -> tuple[int, dict or None, dict or None]:
        """
        Fetches all files and metadata from PanDA for a given job ID.

        Args:
            log_file (str): The name of the log file to fetch.

        Returns:
            Exit code (int): The exit code indicating the status of the operation.
            File dictionary (dict): A dictionary containing the file names and their corresponding paths.
            Metadata dictionary (dict): A dictionary containing the relevant metadata for the job.
        """
        _metadata_dictionary = {}
        _file_dictionary = {}

        # Download metadata and pilot log concurrently
        workdir = os.path.join(self.cache, "jobs")
        metadata_task = asyncio.create_task(fetch_data(self.pandaid, filename="metadata.json", jsondata=True, workdir=workdir))
        pilot_log_task = asyncio.create_task(fetch_data(self.pandaid, filename=log_file, jsondata=False, workdir=workdir))

        # Wait for both downloads to complete
        metadata_success, metadata_message = await metadata_task
        pilot_log_success, pilot_log_message = await pilot_log_task

        if pilot_log_success != 0:
            logger.warning(f"Failed to fetch the pilot log file for PandaID {self.pandaid} - will only use metadata for error analysis.")
        else:
            _file_dictionary[log_file] = pilot_log_message
            logger.info(f"Downloaded file: {log_file}, stored as {pilot_log_message}")

        if metadata_success != 0:
            logger.warning(f"Failed to fetch metadata for PandaID {self.pandaid} - will not be able to analyze the job failure.")
            return EC_NOTFOUND, _file_dictionary, _metadata_dictionary

        logger.info(f"Downloaded JSON file: {metadata_message}")
        _file_dictionary["json"] = metadata_message

        # Verify that the current job is actually a failed job (otherwise, we don't want to download the log files)
        job_data = read_json_file(metadata_message)
        if not job_data:
            logger.warning(f"Error: Failed to read the JSON data from {metadata_message}.")
            return EC_UNKNOWN_ERROR, None, None
        if not job_data['job']['jobstatus'] == 'failed':
            logger.warning(f"Error: The job with PandaID {self.pandaid} is not in a failed state - nothing to explain.")
            return EC_UNKNOWN_ERROR, None, None

        # Fetch pilot error descriptions
        path = os.path.join(self.cache, "pilot_error_codes_and_descriptions.json")
        pilot_error_descriptions = read_json_file(path)
        if not pilot_error_descriptions:
            logger.warning("Error: Failed to read the pilot error descriptions.")
            return EC_UNKNOWN_ERROR, None, None

        # Fetch transform error descriptions
        path = os.path.join(self.cache, "trf_error_codes_and_descriptions.json")
        transform_error_descriptions = read_json_file(path)
        if not transform_error_descriptions:
            logger.warning("Error: Failed to read the transform error descriptions.")
            return EC_UNKNOWN_ERROR, None, None

        # Extract relevant metadata from the JSON data
        try:
            _metadata_dictionary["piloterrorcode"] = job_data['job']['piloterrorcode']
            _metadata_dictionary["piloterrordiag"] = job_data['job']['piloterrordiag']
            _metadata_dictionary["exeerrorcode"] = job_data['job']['exeerrorcode']
            _metadata_dictionary["exeerrordiag"] = job_data['job']['exeerrordiag']
            _metadata_dictionary["piloterrordescription"] = pilot_error_descriptions.get(str(_metadata_dictionary.get("piloterrorcode")))
            _metadata_dictionary["trferrordescription"] = transform_error_descriptions.get(str(_metadata_dictionary.get("exeerrorcode")))
        except KeyError as e:
            logger.warning(f"Error: Missing key in JSON data: {e}")
            return EC_UNKNOWN_ERROR, None, None

        return EC_OK, _file_dictionary, _metadata_dictionary

    def extract_preceding_lines_streaming(self, log_file: str, error_pattern: str, num_lines: int = 20, output_file: str = None):
        """
        Extracts the preceding lines from a log file when a specific error pattern is found.

        Note: Can handle very large files efficiently by using a sliding window approach.

        Args:
            log_file (str): The path to the log file to be analyzed.
            error_pattern (str): The regular expression pattern to search for in the log file.
            num_lines (int): The number of preceding lines to extract (default is 20).
            output_file (str, optional): If provided, the extracted lines will be saved to this file.
        """
        logger.info(f"Searching for error pattern '{error_pattern}' in log file '{log_file}'.")
        buffer = deque(maxlen=num_lines)
        pattern = re.compile(error_pattern)

        with open(log_file, 'r', encoding='utf-8') as file:
            for line in file:
                buffer.append(line)
                if pattern.search(line):
                    # Match found; output the preceding lines
                    if output_file:
                        with open(output_file, 'w') as out_file:
                            out_file.writelines(buffer)
                        logger.info(f"Extracted lines saved to: {output_file}")
                    else:
                        logger.warning("".join(buffer))
                    return

    def get_relevant_error_string(self, metadata_dictionary: dict) -> str:
        """
        Construct a relevant error string based on the metadata dictionary.

        This function will select a proper error string to use when extracting the relevant context from the log file.

        Args:
            metadata_dictionary (dict): A dictionary containing metadata about the job.

        Returns:
            str: A formatted error string that includes pilot and transform error codes and descriptions.
        """
        depth = 50  # Number of characters to use from the error description

        pilot_error_code = metadata_dictionary.get("piloterrorcode", 1008)  # Default to 1008 if not found
        pilot_error_diag = metadata_dictionary.get("piloterrordiag", "CRITICIAL")
        # exe_error_code = metadata_dictionary.get("exeerrorcode", "Unknown")
        # exe_error_diag = metadata_dictionary.get("exeerrordiag", "No description available.")

        # This dictionary can be used to find relevant error strings that might appear in the log based on the error codes.
        error_string_dictionary = {
            1099: "Failed to stage-in file",
            1104: r"work directory \(.*?\) is too large",  # the regular expression will be ignored
            1150: "pilot has decided to kill looping job",  # i.e. this string will appear in the log when the pilot has decided that the job is looping
            1201: "caught signal: SIGTERM",  # need to add all other kill signals here
            1235: "job has exceeded the memory limit",
            1324: "Service not available at the moment",
        }

        # If the current error code is not in the error string dictionary, then we will use a part of the pilot error description as the error string.
        if pilot_error_code not in error_string_dictionary:
            error_string_dictionary[pilot_error_code] = pilot_error_diag[:depth]  # Use the first 50 characters of the description

        return error_string_dictionary.get(pilot_error_code, "No relevant error string found.")

    def formulate_question(self, output_file: str, metadata_dictionary: dict) -> str:
        """
        Construct a question to ask the LLM based on the extracted lines and metadata.

        Args:
            output_file:
            metadata_dictionary:

        Returns:
            str: A formatted question string to be sent to the LLM.
        """
        # Check if the output file exists and read its contents, otherwise the prompt will only use the known pilot error descriptions.
        if output_file:
            log_extracts = read_file(output_file)
            if not log_extracts:
                logger.warning(f"Error: Failed to read the extracted log file {output_file}.")
                return ""
        else:
            log_extracts = None

        #    errorcode = metadata_dictionary.get("piloterrorcode", None)
        errordiag = metadata_dictionary.get("piloterrordiag", None)
        if not errordiag:
            logger.warning("Error: No pilot error diagnosis found in the metadata dictionary.")
            return ""

        question = ("You are an expert on distributed analysis. A PanDA job has failed. The job was run on a linux worker node, "
                    "and the pilot has detected an error.\n\n")

        description = ""
        if log_extracts:
            description += f"Error diagnostics: \"{errordiag}\".\n\n"
            description += f"The log extracts are as follows:\n\n\"{log_extracts}\""
        else:
            description += f"Error diagnostics: \"{errordiag}\".\n\n"

        preliminary_diagnosis = metadata_dictionary.get("piloterrordiag", None)
        if preliminary_diagnosis:
            description += f"\nA preliminary diagnosis exists: \"{metadata_dictionary.get('piloterrordescription', 'No description available.')}\"\n\n"

        question += """
    Please convert the following explanation for PanDA job error code 1221 into a Python dictionary.
    Do not wrap the dictionary in Markdown (no triple backticks, no "```python").
The dictionary should have the error code as the key (an integer), and its value should include the following fields:

    "description": A short summary of the error in plain English.

    "non_expert_guidance": A dictionary containing:

        "problem": A plain-language explanation of the issue.

        "possible_causes": A list of plausible reasons for this error.

        "recommendations": A list of actionable steps a scientist or user should take.

    "expert_guidance": A dictionary containing:

        "analysis": A technical explanation of the root cause.

        "investigation_steps": A list of diagnostic actions a system admin or expert should take.

        "possible_scenarios": A dictionary with known edge cases or failure patterns, each with a short explanation.

        "preventative_measures": A list of best practices to prevent this issue in the future.

Return only a valid Python dictionary. Here's the error description:
        """
        question += f"\n\n{description}\n\n"

        return question

    def generate_question(self, log_file: str) -> str:
        """
        Generate a question to ask the LLM based on the log file and metadata.

        Args:
            log_file (str): The path to the log file to be analyzed.

        Returns:
            str: A formatted question string to be sent to the LLM.
        """
        # Fetch the files from PanDA
        exit_code, file_dictionary, metadata_dictionary = asyncio.run(self.fetch_all_data(log_file))
        if exit_code == EC_NOTFOUND:
            logger.warning(
                f"No log files found for PandaID {self.pandaid} - will proceed with only superficial knowledge of failure.")
        elif not file_dictionary:
            logger.warning(f"Error: Failed to fetch files for PandaID {self.pandaid}.")
            sys.exit(1)

        # Extract the relevant parts for error analysis
        if file_dictionary and log_file not in file_dictionary and exit_code != EC_NOTFOUND:
            logger.warning(f"Error: Log file {log_file} not found in the fetched files.")
            sys.exit(1)
        output_file = f"{self.pandaid}-{log_file}_extracted.txt"
        log_file_path = file_dictionary.get(log_file) if file_dictionary else None
        if log_file_path:
            # Create an output file for the log extracts
            error_string = self.get_relevant_error_string(metadata_dictionary)
            self.extract_preceding_lines_streaming(log_file_path, error_string[:40], output_file=output_file)
        if not os.path.exists(output_file):
            logger.info("The error string was not found in the log file, so no output file was created.")
            output_file = None

        # Formulate the question based on the extracted lines and metadata
        question = self.formulate_question(output_file, metadata_dictionary)
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

    parser.add_argument('--log-file', type=str, default='pilotlog.txt',
                        help='Optional log file (default is pilotlog.txt)')
    parser.add_argument('--pandaid', type=int, required=True,
                        help='PandaID (integer)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use (e.g., openai, anthropic, etc.)')
    parser.add_argument('--cache', type=str, default="cache",
                        help='Location of cache directory (default: cache)')
    # parser.add_argument('--session-id', type=str, required=True,
    #                     help='Session ID for the context memory')
    args = parser.parse_args()

    agent = LogAnalysisAgent(args.model, args.pandaid, args.cache)

    # Generate a proper question to ask the LLM based on the metadata and log files
    question = agent.generate_question(args.log_file)
    logger.info(f"Asking question: \n\n{question}")

    # Ask the question to the LLM
    answer = agent.ask(question)

    logger.info(f"Answer from {args.model.capitalize()}:\n{answer}")

    # store the answer in the session memory
    # ..

    sys.exit(0)


if __name__ == "__main__":
    main()
