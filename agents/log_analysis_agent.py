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
# import asyncio
import logging
import os
import re
import requests
import sys
from collections import deque
# from fastmcp import FastMCP
from time import sleep

from tools.context_memory import ContextMemory
from tools.errorcodes import EC_NOTFOUND, EC_OK, EC_UNKNOWN_ERROR, EC_TIMEOUT
from tools.server_utils import MCP_SERVER_URL, check_server_health
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
memory = ContextMemory()
# mcp = FastMCP("panda")


class LogAnalysisAgent:
    """
    A simple command-line agent that interacts with a RAG server to analyze log files.
    This agent fetches log files from PanDA, extracts relevant parts, and asks an LLM for analysis.
    """
    def __init__(self, model: str, pandaid: str, cache: str, session_id: str) -> None:
        """
        Initialize the LogAnalysisAgent with a model.

        Args:
            model (str): The model to use for generating the answer (e.g., 'openai', 'anthropic').
            pandaid (str): The PanDA job ID to analyze.
            cache (str): The location of the cache directory for storing downloaded files.
            session_id (str): The session ID for tracking the conversation.
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

        self.session_id = session_id  # Session ID for tracking conversation

    def ask(self, question: str) -> str:
        """
        Send a question to the LLM and retrieve the answer.

        Args:
            question (str): The question to ask the RAG server.

        Returns:
            str: The answer returned by the MCP server.
        """

        def extract_python_literal(s: str) -> str:
            # strip fences if present
            m = re.search(r"```(?:python)?\s*(\{.*\})\s*```", s, flags=re.S)
            return m.group(1) if m else s.strip()

        def parse_answer_to_dict(raw: str) -> dict:
            payload = extract_python_literal(raw)
            return ast.literal_eval(payload)  # works with triple-quoted strings

        server_url = f"{MCP_SERVER_URL}/rag_ask"
        import time
        logger.info(f"ask() was called at {time.time()}")

        response = requests.post(server_url, json={"question": question, "model": self.model})
        if response.ok:
            answer = response.json()["answer"]
            if not answer:
                err = "No answer returned from the LLM."
                logger.error(f"{err}")
                return f"{err}"

            # Strip code block formatting
            try:
                # clean_code = re.sub(r"^```(?:python|json)?\n|\n```$", "", answer.strip())
                clean_code = re.sub(r"^```(?:python|json)?\n|\n```$", "", answer.strip())
            except re.error as e:
                logger.error(f"Regex error while cleaning code: {e}")
                return f"{e}"

            # convert the answer to a Python dictionary
            try:
                answer_dict = parse_answer_to_dict(clean_code)
            except (SyntaxError, ValueError) as e:
                err = f"Error converting answer to dictionary: {e}\n\nanswer={answer}\n\nclean_code={clean_code}"
                logger.error(f"{err}")
                return f"{err}"

            if not answer_dict:
                err = "Failed to store the answer as a Python dictionary."
                logger.error(f"{err}")
                return f"{err}"

            # format the answer for better readability
            formatted_answer = format_answer(answer_dict)
            if not formatted_answer:
                logger.error(f"Failed to format the answer from the LLM using: {answer_dict}")
                sys.exit(1)

            logger.info(f"Answer from {self.model.capitalize()}:\n{formatted_answer}")

            # store the answer in the session memory
            if self.session_id != "None":
                memory.store_turn(self.session_id, "Investigate the job failure", formatted_answer)
                logger.info(f"Answer stored in session ID {self.session_id}:\n\nquestion={question}\n\nanswer={formatted_answer}")

            return formatted_answer

        return f"requests.post() error: {response.text}"

    # async def fetch_all_data(self, log_file: str) -> tuple[int, dict or None, dict or None]:

    def fetch_all_data(self, log_file: str) -> tuple[int, dict or None, dict or None, str]:
        """
        Fetches all files and metadata from PanDA for a given job ID.

        The function might update the log_file name to fetch based on the job's error code,
        so it must also return the actual log file name used since that will be used later on.

        Args:
            log_file (str): The name of the log file to fetch.

        Returns:
            Exit code (int): The exit code indicating the status of the operation.
            File dictionary (dict): A dictionary containing the file names and their corresponding paths.
            Metadata dictionary (dict): A dictionary containing the relevant metadata for the job.
            Log file (str): The actual log file name used for fetching.
        """
        _metadata_dictionary = {}
        _file_dictionary = {}

        # Download metadata and pilot log concurrently
        workdir = os.path.join(self.cache, "jobs")
        # metadata_task = asyncio.create_task(fetch_data(self.pandaid, filename="metadata.json", jsondata=True, workdir=workdir))
        # pilot_log_task = asyncio.create_task(fetch_data(self.pandaid, filename=log_file, jsondata=False, workdir=workdir))

        # Wait for both downloads to complete
        # metadata_success, metadata_message = await metadata_task
        # pilot_log_success, pilot_log_message = await pilot_log_task
        metadata_success, metadata_message = fetch_data(self.pandaid, filename="metadata.json", jsondata=True, workdir=workdir)
        if metadata_success != 0:
            logger.warning(f"Failed to fetch metadata for PandaID {self.pandaid} - will not be able to analyze the job failure.")
            return EC_NOTFOUND, _file_dictionary, _metadata_dictionary, log_file
        logger.info(f"Downloaded JSON file: {metadata_message}")
        _file_dictionary["json"] = metadata_message

        # Verify that the current job is actually a failed job (otherwise, we don't want to download the log files)
        job_data = read_json_file(metadata_message)
        if not job_data:
            logger.warning(f"Error: Failed to read the JSON data from {metadata_message}.")
            return EC_UNKNOWN_ERROR, None, None, log_file

        # what happened to the job - check the status
        jobstatus = job_data['job'].get('jobstatus', 'unknown').lower()
        logger.info(f"Job status: {jobstatus}")

        try:
            _metadata_dictionary["piloterrorcode"] = job_data['job']['piloterrorcode']
            _metadata_dictionary["piloterrordiag"] = job_data['job']['piloterrordiag']
            _metadata_dictionary["exeerrorcode"] = job_data['job']['exeerrorcode']
            _metadata_dictionary["exeerrordiag"] = job_data['job']['exeerrordiag']
            _metadata_dictionary['jobstatus'] = jobstatus
        except KeyError as e:
            logger.warning(f"Error: Missing key in JSON data: {e}")
            return EC_UNKNOWN_ERROR, None, None, log_file

        # update the log file for user payload failures
        if _metadata_dictionary.get("piloterrorcode", 0) == 1305:
            log_file = "payload.stdout"
            logger.info(f"Updated log file to fetch for user payload failure: {log_file}")

        # only download the log file for failed jobs (otherwise all relevant info is in the metadata file)
        if jobstatus == 'failed':
            pilot_log_success, pilot_log_message = fetch_data(self.pandaid, filename=log_file, jsondata=False, workdir=workdir)
            if pilot_log_success != 0:
                logger.warning(f"Failed to fetch the pilot log file for PandaID {self.pandaid} - will only use metadata for error analysis.")
            else:
                _file_dictionary[log_file] = pilot_log_message
                logger.info(f"Downloaded file: {log_file}, stored as {pilot_log_message}")

        # if not job_data['job']['jobstatus'] == 'failed':
        #     logger.warning(f"Error: The job with PandaID {self.pandaid} is not in a failed state - nothing to explain.")
        #     return EC_UNKNOWN_ERROR, None, None, log_file

        # less info is needed for jobs that did not fail
        if jobstatus not in ['failed', 'holding', 'cancelled']:
            _metadata_dictionary['metadata'] = job_data['job'].copy()
            return EC_OK, _file_dictionary, _metadata_dictionary, log_file

        # Fetch pilot error descriptions
        path = os.path.join(self.cache, "pilot_error_codes_and_descriptions.json")
        pilot_error_descriptions = read_json_file(path)
        if not pilot_error_descriptions:
            logger.warning("Error: Failed to read the pilot error descriptions.")
            return EC_UNKNOWN_ERROR, None, None, log_file

        # Fetch transform error descriptions
        path = os.path.join(self.cache, "trf_error_codes_and_descriptions.json")
        transform_error_descriptions = read_json_file(path)
        if not transform_error_descriptions:
            logger.warning("Error: Failed to read the transform error descriptions.")
            return EC_UNKNOWN_ERROR, None, None, log_file

        # Extract relevant metadata from the JSON data
        try:
            _metadata_dictionary["piloterrordescription"] = pilot_error_descriptions.get(str(_metadata_dictionary.get("piloterrorcode")))
            _metadata_dictionary["trferrordescription"] = transform_error_descriptions.get(str(_metadata_dictionary.get("exeerrorcode")))
        except KeyError as e:
            logger.warning(f"Error: Missing key in JSON data: {e}")
            return EC_UNKNOWN_ERROR, None, None, log_file

        return EC_OK, _file_dictionary, _metadata_dictionary, log_file

    def extract_preceding_lines_streaming(
            self,
            log_file: str,
            error_pattern: str,
            num_lines: int = 30,
            output_file: str = None
    ):
        """
        Extracts the preceding lines from a log file when a specific error pattern is found.

        Special case:
            If error_pattern == "" and "payload" in log_file, simply return the last `num_lines` lines.

        Args:
            log_file (str): The path to the log file to be analyzed.
            error_pattern (str): The regular expression pattern to search for in the log file.
            num_lines (int): The number of preceding lines to extract (default is 30).
            output_file (str, optional): If provided, the extracted lines will be saved to this file.
        """
        logger.info(f"Searching for error pattern '{error_pattern}' in log file '{log_file}'.")

        # Special case: just return last num_lines from payload logs
        if error_pattern == "" and "payload" in log_file:
            num_lines = 100
            with open(log_file, 'r', encoding='utf-8') as file:
                lines = deque(file, maxlen=num_lines)  # keeps last num_lines lines
            if output_file:
                with open(output_file, 'w') as out_file:
                    out_file.writelines(lines)
                logger.info(f"Last {num_lines} lines saved to: {output_file}")
            else:
                logger.warning("".join(lines))
            return

        # Normal case: search for error pattern
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

            #    errorcode = metadata_dictionary.get("piloterrorcode", None)
            errordiag = metadata_dictionary.get("piloterrordiag", None)
            if not errordiag:
                logger.warning("Error: No pilot error diagnosis found in the metadata dictionary.")
                return ""
        else:
            log_extracts = None
            errordiag = None

        jobstatus = metadata_dictionary.get("jobstatus", "unknown").lower()
        question = f"You are an expert on distributed analysis. A PanDA job has job status \'{jobstatus}\'. The job was run on a linux worker node."
        if jobstatus in ['failed', 'holding', 'cancelled']:
            question += "and the pilot has detected a possible error.\n\n"
        question += "\n\n"

        description = ""
        if jobstatus == 'finished':
            question += "The job has finished successfully. Please analyze the metadata and summarize it:\n\n"
            question += f"{metadata_dictionary.get('metadata')}\n"
            question += """Do not wrap the dictionary in Markdown (no triple backticks, no "```python"). Return only a valid Python dictionary."""
            return question
        elif jobstatus not in ['failed', 'holding', 'cancelled']:
            question += f"The job is in state {jobstatus}. Please analyze the metadata and summarize it:\n\n"
            question += f"{metadata_dictionary.get('metadata')}\n"
            question += """Do not wrap the dictionary in Markdown (no triple backticks, no "```python"). Return only a valid Python dictionary."""
            return question
        elif jobstatus in ['failed', 'holding', 'cancelled']:
            if log_extracts:
                description += f"Error diagnostics: \'{errordiag}\'.\n\n"
                description += f"The log extracts are as follows:\n\n\'{log_extracts}\'"
            elif errordiag:
                description += f"Error diagnostics: \"{errordiag}\".\n\n"
            else:
                description += f"Error diagnostics: \"{errordiag}\".\n\n"
            preliminary_diagnosis = metadata_dictionary.get("piloterrordiag", None)
            if preliminary_diagnosis:
                description += f"\nA preliminary diagnosis exists: \"{metadata_dictionary.get('piloterrordescription', 'No description available.')}\"\n\n"

        piloterrorcode = metadata_dictionary.get("piloterrorcode", "")
        question += f"The pilot error code is {piloterrorcode}.\n\n"
        question += """
Instructions:

Do not wrap the dictionary in Markdown (no triple backticks, no "```python").

Return the dictionary with field names and values using HTML tags for bold, e.g., <b>description</b>.

Make sure to wrap any multi-paragraph strings in triple quotes.

No code fences.

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

Return only a valid Python dictionary.

Here's the error description:
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
        # exit_code, file_dictionary, metadata_dictionary, log_file = asyncio.run(self.fetch_all_data(log_file))
        exit_code, file_dictionary, metadata_dictionary, log_file = self.fetch_all_data(log_file)
        if exit_code == EC_NOTFOUND:
            logger.warning(
                f"No log files found for PandaID {self.pandaid} - will proceed with only superficial knowledge of failure.")
        elif not file_dictionary:
            logger.warning(f"Error: Failed to fetch files for PandaID {self.pandaid}.")
            sys.exit(1)

        # Extract the relevant parts for error analysis
        if len(file_dictionary) == 1 and 'json' in file_dictionary:
            logger.info(f"Only metadata found for PandaID {self.pandaid} - no log files to analyze.")
            output_file = None
        else:
            if file_dictionary and log_file not in file_dictionary and exit_code != EC_NOTFOUND:
                logger.warning(f"Error: Log file {log_file} not found in the fetched files.")
                sys.exit(1)
            output_file = f"{self.pandaid}-{log_file}_extracted.txt"
            log_file_path = file_dictionary.get(log_file) if file_dictionary else None
            if log_file_path:
                # Create an output file for the log extracts
                if "pilotlog" in log_file:
                    error_string = self.get_relevant_error_string(metadata_dictionary)
                    if error_string:
                        error_string = error_string[:40]  # Limit to first 40 characters
                else:
                    error_string = ""  # not needed for payload logs
                self.extract_preceding_lines_streaming(log_file_path, error_string, output_file=output_file)
            if not os.path.exists(output_file):
                logger.info("The error string was not found in the log file, so no output file was created.")
                output_file = None

        # Formulate the question based on the extracted lines and metadata
        question = self.formulate_question(output_file, metadata_dictionary)
        if not question:
            logger.warning("No question could be generated.")
            sys.exit(1)

        return question


def format_answer(answer: dict) -> str:
    """
    Format the answer dictionary into a human-readable string.

    Args:
        answer (dict): The answer dictionary returned by the LLM.

    Returns:
        str: A formatted string containing the description, non-expert guidance, and expert guidance.
    """
    # the dictionary will only ever contain a single key-value pair
    logger.info(f"answer dictionary to format: {answer} type={type(answer)}")
    error_code, value = next(iter(answer.items()))

    to_store = ""

    # for metadata, the value is a string - otherwise a dictionary
    if isinstance(value, str):
        to_store += value
        return to_store

    description = value.get('description')
    if description:
        to_store += f"**Description:**\n{description}\n\n"

    non_expert_guidance = value.get('non_expert_guidance')  # dict
    if non_expert_guidance:
        problem = non_expert_guidance.get('problem')
        if problem:
            to_store += f"**Non-expert guidance - problem:**\n{problem}\n\n"

        possible_causes = non_expert_guidance.get('possible_causes')
        if possible_causes:
            bullet_list = '\n'.join(f'* {item}' for item in possible_causes) if isinstance(possible_causes, list) else possible_causes
            to_store += f"**Non-expert guidance - possible causes:**\n{bullet_list}\n\n"

        recommendations = non_expert_guidance.get('recommendations')
        if recommendations:
            bullet_list = '\n'.join(f'* {item}' for item in recommendations) if isinstance(recommendations, list) else recommendations
            to_store += f"**Non-expert guidance - recommendations:**\n{bullet_list}\n\n"

    expert_guidance = value.get('expert_guidance')  # dict
    if expert_guidance:
        analysis = expert_guidance.get('analysis')
        if analysis:
            to_store += f"**Expert guidance - analysis:**\n{analysis}\n\n"

        investigation_steps = expert_guidance.get('investigation_steps')
        if investigation_steps:
            bullet_list = '\n'.join(f'* {item}' for item in investigation_steps) if isinstance(investigation_steps, list) else investigation_steps
            to_store += f"**Expert guidance - investigation steps:**\n{bullet_list}\n\n"

        possible_scenarios = expert_guidance.get('possible_scenarios')
        if possible_scenarios:
            if isinstance(possible_scenarios, dict):
                bullet_list = '\n'.join(f'* {k}: {v}' for k, v in possible_scenarios.items())
                to_store += f"**Expert guidance - possible scenarios:**\n{bullet_list}\n\n"
            elif isinstance(possible_scenarios, list):
                bullet_list = '\n'.join(f'* {item}' for item in possible_scenarios) if isinstance(possible_scenarios, list) else possible_scenarios
                to_store += f"**Expert guidance - possible scenarios:**\n{bullet_list}\n\n"
            else:
                to_store += f"**Expert guidance - possible scenarios:**\n{possible_scenarios}\n\n"

    return to_store


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
    parser.add_argument('--session-id', type=str, default="None",
                        help='Session ID for the context memory')
    args = parser.parse_args()

    agent = LogAnalysisAgent(args.model, args.pandaid, args.cache, args.session_id)

    # Generate a proper question to ask the LLM based on the metadata and log files
    question = agent.generate_question(args.log_file)
    logger.info(f"Asking question: \n\n{question}")

    # Ask the question to the LLM
    answer = agent.ask(question)
    if not answer:
        logger.error("No answer returned from the LLM.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
