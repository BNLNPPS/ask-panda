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
from tools.tools import fetch_data, read_json_file, read_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("error_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    server_url = f"{MCP_SERVER_URL}/rag_ask"
    response = requests.post(server_url, json={"question": question, "model": model})
    if response.ok:
        return response.json()["answer"]

    return f"Error: {response.text}"


async def fetch_all_data(pandaid: int, log_files: list) -> tuple[int, dict or None, dict or None]:
    """
    Fetches all files and metadata from PanDA for a given job ID.

    Args:
        pandaid (int): The panda job ID.
        log_files (list): A list of log files to fetch.

    Returns:
        Exit code (int): The exit code indicating the status of the operation.
        File dictionary (dict): A dictionary containing the file names and their corresponding paths.
        Metadata dictionary (dict): A dictionary containing the relevant metadata for the job.
    """
    _metadata_dictionary = {}
    _file_dictionary = {}

    # Can only download a single log file
    log_file = log_files[0] if log_files else 'pilotlog.txt'

    # Download metadata and pilot log concurrently
    metadata_task = asyncio.create_task(fetch_data(pandaid, jsondata=True))
    pilot_log_task = asyncio.create_task(fetch_data(pandaid, filename=log_file))

    # Wait for both downloads to complete
    metadata_success, metadata_message = await metadata_task
    pilot_log_success, pilot_log_message = await pilot_log_task

    if pilot_log_success != 0:
        logger.warning(f"Failed to fetch the pilot log file for PandaID {pandaid} - will only use metadata for error analysis.")
    else:
        _file_dictionary[log_file] = pilot_log_message
        logger.info(f"Downloaded file: {log_file}, stored as {pilot_log_message}")

    if metadata_success != 0:
        logger.warning(f"Failed to fetch metadata for PandaID {pandaid} - will not be able to analyze the job failure.")
        return EC_NOTFOUND, _file_dictionary, _metadata_dictionary

    logger.info(f"Downloaded JSON file: {metadata_message}")
    _file_dictionary["json"] = metadata_message

    # Verify that the current job is actually a failed job (otherwise, we don't want to download the log files)
    job_data = read_json_file(metadata_message)
    if not job_data:
        logger.warning(f"Error: Failed to read the JSON data from {metadata_message}.")
        return EC_UNKNOWN_ERROR, None, None
    if not job_data['job']['jobstatus'] == 'failed':
        logger.warning(f"Error: The job with PandaID {pandaid} is not in a failed state - nothing to explain.")
        return EC_UNKNOWN_ERROR, None, None

    # Fetch pilot error descriptions
    pilot_error_descriptions = read_json_file("cache/pilot_error_codes_and_descriptions.json")
    if not pilot_error_descriptions:
        logger.warning(f"Error: Failed to read the pilot error descriptions.")
        return EC_UNKNOWN_ERROR, None, None

    # Fetch transform error descriptions
    transform_error_descriptions = read_json_file("cache/trf_error_codes_and_descriptions.json")
    if not transform_error_descriptions:
        logger.warning(f"Error: Failed to read the transform error descriptions.")
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


def extract_preceding_lines_streaming(log_file: str, error_pattern: str, num_lines: int = 20, output_file: str = None):
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


def get_relevant_error_string(metadata_dictionary: dict) -> str:
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


def formulate_question(output_file: str, metadata_dictionary: dict) -> str:
    """
    Construct a question to ask the LLM based on the extracted lines and metadata.

    Args:
        output_file:
        metadata_dictionary:

    Returns:

    """
    # Check if the output file exists and read its contents, otherwise the prompt will only use the known pilot error descriptions.
    if output_file:
        log_extracts = read_file(output_file)
        if not log_extracts:
            logger.warning(f"Error: Failed to read the extracted log file {output_file}.")
            return ""
    else:
        log_extracts = None

    piloterrordiag = metadata_dictionary.get("piloterrordiag", None)
    if not piloterrordiag:
        logger.warning("Error: No pilot error diagnosis found in the metadata dictionary.")
        return ""

    question = "You are an expert on distributed analysis. A PanDA job has failed. The job was run on a linux worker node, and the pilot has detected an error.\n\n"
    if log_extracts:
        question += f"Analyze the given log extracts for the error: \"{piloterrordiag}\".\n\n"
        question += f"The log extracts are as follows:\n\n\"{log_extracts}\""
    else:
        question += f"Analyze the error: \"{piloterrordiag}\".\n\n"

    preliminary_diagnosis = metadata_dictionary.get("piloterrordiag", None)
    if preliminary_diagnosis:
        question += f"\nA preliminary diagnosis exists: \"{metadata_dictionary.get('piloterrordescription', 'No description available.')}\"\n\n"

    question += (
        "\n\nPlease provide a detailed analysis of the error and suggest possible solutions or next steps if possible. Separate your answer into the following sections: "
        "1) Explanations and suggestions for non-expert users (for scientists and not for complete beginners, so don't oversimplify explanations), and only show information that is relevant for users, 2) Explanations and suggestions for experts and/or system admins\n")

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

    parser.add_argument('--log-files', type=str, required=True,
                        help='Comma-separated list of log files')
    parser.add_argument('--pandaid', type=int, required=True,
                        help='PandaID (integer)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use (e.g., openai, anthropic, etc.)')
    parser.add_argument('--mode', type=str, required=True,
                        help='Mode to use (ML or contextual)')
    args = parser.parse_args()

    # Split the log files into a list
    log_files = args.log_files.split(',')

    # Fetch the files from PanDA
    exit_code, file_dictionary, metadata_dictionary = asyncio.run(fetch_all_data(args.pandaid, log_files))
    if exit_code == EC_NOTFOUND:
        logger.warning(f"No log files found for PandaID {args.pandaid} - will proceed with only superficial knowledge of failure.")
    elif not file_dictionary:
        logger.warning(f"Error: Failed to fetch files for PandaID {args.pandaid}.")
        sys.exit(1)
    logger.info(metadata_dictionary)

    # Extract the relevant parts for error analysis
    if args.mode.lower() == 'contextual':
        # Use contextual mode to analyze the log files
        # Only analyze the pilot log for now
        log_file = 'pilotlog.txt'
        if file_dictionary and log_file not in file_dictionary and exit_code != EC_NOTFOUND:
            logger.warning(f"Error: Log file {log_file} not found in the fetched files.")
            sys.exit(1)
        output_file = f"{args.pandaid}-{log_file}_extracted.txt"
        log_file_path = file_dictionary.get(log_file) if file_dictionary else None
        if log_file_path:
            # Create an output file for the log extracts
            error_string = get_relevant_error_string(metadata_dictionary)
            extract_preceding_lines_streaming(log_file_path, error_string[:40], output_file=output_file)
        if not os.path.exists(output_file):
            logger.info("The error string was not found in the log file, so no output file was created.")
            output_file = None

        # Formulate the question based on the extracted lines and metadata
        question = formulate_question(output_file, metadata_dictionary)
        if not question:
            logger.warning("No question could be generated.")
            sys.exit(1)
        logger.info(f"Asking question: \n\n{question}")

        # Ask the question to the LLM
        answer = ask(question, args.model)
        print(f"Answer from {args.model.capitalize()} (via RAG):\n{answer}")

    elif args.mode.lower() == 'ml':
        # Use ML mode to analyze the log files
        raise NotImplementedError("ML mode is not implemented yet.")
    else:
        logger.error(f"Invalid mode specified: {args.mode}. Use 'ML' or 'contextual'.")
        sys.exit(1)

    #answer = ask(question, model)
    #print(f"Answer from {model.capitalize()} (via RAG):\n{answer}")

if __name__ == "__main__":
    main()