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
import json
import logging
import os
import re
import requests
import sys

from collections import deque
from fastmcp import FastMCP
from time import sleep
from typing import Optional
from https import download_data

import errorcodes
from server import MCP_SERVER_URL, check_server_health

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


def fetch_data(panda_id: int, filename: str = None, jsondata: bool = False) -> tuple[int, Optional[str]]:
    """
    Fetches a given file from PanDA.

    Args:
        panda_id (int): The job or task ID.
        filename (str): The name of the file to fetch.
        jsondata (bool): If True, return a JSON string for the job.

    Returns:
        str or None: The name of the downloaded file.
        exit_code (int): The exit code indicating the status of the operation.
    """
    url = (
        f"https://bigpanda.cern.ch/job?pandaid={panda_id}&json"
        if jsondata
        else f"https://bigpanda.cern.ch/filebrowser/?pandaid={panda_id}&json&filename={filename}"
    )
    logger.info(f"Downloading file from: {url}")

    exit_code, response = download_data(url) #  post(url)
    if exit_code == errorcodes.EC_NOTFOUND:
        logger.error(f"File not found for PandaID {panda_id} with filename {filename}.")
        return exit_code, None
    elif exit_code == errorcodes.EC_UNKNOWN_ERROR:
        logger.error(f"Unknown error occurred while fetching data for PandaID {panda_id} with filename {filename}.")
        return exit_code, None

    if response and isinstance(response, str):
        return errorcodes.EC_OK, response
    if response:
        response = response.decode('utf-8')
        response = re.sub(r'([a-zA-Z0-9\])])(?=[A-Z])', r'\1\n', response)  # ensure that each line ends with \n
        return errorcodes.EC_OK, response
    else:
        logger.error(f"Failed to fetch data for PandaID {panda_id} with filename {filename}.")
        return errorcodes.EC_UNKNOWN_ERROR, None


def read_json_file(file_path: str) -> Optional[dict]:
    """
    Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict or None: The contents of the JSON file as a dictionary, or None if the file cannot be read.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read JSON file {file_path}: {e}")
        return None

    return data


def read_file(file_path: str) -> Optional[str]:
    """
    Reads a text file and returns its contents as a string.

    Args:
        file_path (str): The path to the text file.

    Returns:
        str or None: The contents of the text file as a string, or None if the file cannot be read.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as e:
        logger.warning(f"Failed to read file {file_path}: {e}")
        return None


def fetch_all_data(pandaid: int, log_files: list) -> tuple[int, dict or None, dict or None]:
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
    exit_code, json_file_name = fetch_data(pandaid, jsondata=True)
    if not json_file_name:
        logger.warning(f"Error: Failed to fetch the JSON data for PandaID {pandaid}.")
        return exit_code, None, None
    logger.info(f"Downloaded JSON file: {json_file_name}")
    _file_dictionary["json"] = json_file_name

    # Verify that the current job is actually a failed job (otherwise, we don't want to download the log files)
    job_data = read_json_file(json_file_name)
    if not job_data:
        logger.warning(f"Error: Failed to read the JSON data from {json_file_name}.")
        return errorcodes.EC_UNKNOWN_ERROR, None, None
    if not job_data['job']['jobstatus'] == 'failed':
        logger.warning(f"Error: The job with PandaID {pandaid} is not in a failed state - nothing to explain.")
        return errorcodes.EC_UNKNOWN_ERROR, None, None
    logger.info(f"Confirmed that job {pandaid} is in a failed state.")

    # Fetch pilot error descriptions
    pilot_error_descriptions = read_json_file("pilot_error_codes_and_descriptions.json")
    if not pilot_error_descriptions:
        logger.warning(f"Error: Failed to read the pilot error descriptions.")
        return errorcodes.EC_UNKNOWN_ERROR, None, None

    # Fetch transform error descriptions
    transform_error_descriptions = read_json_file("trf_error_codes_and_descriptions.json")
    if not transform_error_descriptions:
        logger.warning(f"Error: Failed to read the transform error descriptions.")
        return errorcodes.EC_UNKNOWN_ERROR, None, None

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
        return errorcodes.EC_UNKNOWN_ERROR, None, None

    # Proceed to download the log files
    for log_file in log_files:
        exit_code, log_file_name = fetch_data(pandaid, filename=log_file)
        if not log_file_name:
            logger.warning(f"Error: Failed to fetch the log file {log_file}.")
            return exit_code, None, _metadata_dictionary

        # Keep track of the file names
        _file_dictionary[log_file] = log_file_name

        # Process the log file content as needed
        # For example, you can print it or analyze it further
        logger.info(f"Downloaded file: {log_file}, stored as {log_file_name}")

    return errorcodes.EC_OK, _file_dictionary, _metadata_dictionary


def extract_preceding_lines_streaming(log_file: str, error_string: str, num_lines: int = 20, output_file: str = None):
    """
    Extracts the preceding lines from a log file when a specific error string is found.

    Note: can handle very large files efficiently by using a sliding window approach.

    Args:
        log_file (str): The path to the log file to be analyzed.
        error_string (str): The error string to search for in the log file.
        num_lines (int): The number of preceding lines to extract (default is 20).
        output_file (str, optional): If provided, the extracted lines will be saved to this file.
    """
    logger.info(f"Searching for error string '{error_string}' in log file '{log_file}'.")
    buffer = deque(maxlen=num_lines)

    with open(log_file, 'r', encoding='utf-8') as file:
        for line in file:
            buffer.append(line)
            if error_string in line:
                # Match found; output the preceding lines
                if output_file:
                    with open(output_file, 'w') as out_file:
                        out_file.writelines(buffer)
                    logger.info(f"Extracted lines saved to: {output_file}")
                else:
                    logger.warning("".join(buffer))
                return

    logger.warning("Error string not found in the log file.")


def get_relevant_error_string(metadata_dictionary: dict) -> str:
    """
    Construct a relevant error string based on the metadata dictionary.

    This functino will select a proper error string to use when extracting the relevant context from the log file.

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
        1150: "pilot has decided to kill looping job",  # i.e. this string will appear in the log when the pilot has decided that the job is looping
        1201: "caught signal: SIGTERM",  # need to add all other kill signals here
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
        "1) Explanations and suggestions for non-expert users, and only show information that is relevant for users, 2) Explanations and suggestions for experts and/or system admins\n")

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
    if ec == errorcodes.EC_TIMEOUT:
        logger.warning(f"Timeout while trying to connect to {MCP_SERVER_URL}.")
        sleep(5)  # Wait for a while before retrying
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
    exit_code, file_dictionary, metadata_dictionary = fetch_all_data(args.pandaid, log_files)
    if exit_code == errorcodes.EC_NOTFOUND:
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
        if file_dictionary and log_file not in file_dictionary and exit_code != errorcodes.EC_NOTFOUND:
            logger.warning(f"Error: Log file {log_file} not found in the fetched files.")
            sys.exit(1)
        output_file = f"{args.pandaid}-{log_file}_extracted.txt"
        log_file_path = file_dictionary.get(log_file) if file_dictionary else None
        if log_file_path:
            # Create an output file for the log extracts
            error_string = get_relevant_error_string(metadata_dictionary)
            extract_preceding_lines_streaming(log_file_path, error_string[:40], output_file=output_file)
        if not os.path.exists(output_file):
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