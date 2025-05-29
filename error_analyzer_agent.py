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
import re
import requests
import sys

from collections import deque
from fastmcp import FastMCP
from typing import Optional
from https import download_data

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
    server_url = "http://localhost:8000/rag_ask"
    response = requests.post(server_url, json={"question": question, "model": model})
    if response.ok:
        return response.json()["answer"]

    return f"Error: {response.text}"


def fetch_data(panda_id: int, filename: str = None, jsondata: bool = False) -> Optional[str]:
    """
    Fetches a given file from PanDA.

    Args:
        panda_id (int): The job or task ID.
        filename (str): The name of the file to fetch.
        jsondata (bool): If True, return a JSON string for the job.

    Returns:
        str or None: The name of the downloaded file.
    """
    url = (
        f"https://bigpanda.cern.ch/job?pandaid={panda_id}&json"
        if jsondata
        else f"https://bigpanda.cern.ch/filebrowser/?pandaid={panda_id}&json&filename={filename}"
    )
    logger.info(f"Downloading file from: {url}")

    response = download_data(url) #  post(url)
    if response and isinstance(response, str):
        return response
    if response:
        response = response.decode('utf-8')
        response = re.sub(r'([a-zA-Z0-9\])])(?=[A-Z])', r'\1\n', response)  # ensure that each line ends with \n
        return response
    else:
        logger.error(f"Failed to fetch data for PandaID {panda_id} with filename {filename}.")
        return None


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


def fetch_all_data(pandaid: int, log_files: list) -> tuple[dict or None, dict or None]:
    """
    Fetches all files and metadata from PanDA for a given job ID.

    Args:
        pandaid (int): The panda job ID.
        log_files (list): A list of log files to fetch.

    Returns:
        File dictionary (dict): A dictionary containing the file names and their corresponding paths.
        Metadata dictionary (dict): A dictionary containing the relevant metadata for the job.
    """
    _metadata_dictionary = {}
    _file_dictionary = {}
    json_file_name = fetch_data(pandaid, jsondata=True)
    if not json_file_name:
        logger.warning(f"Error: Failed to fetch the JSON data for PandaID {pandaid}.")
        return None, None
    logger.info(f"Downloaded JSON file: {json_file_name}")
    _file_dictionary["json"] = json_file_name

    # Verify that the current job is actually a failed job (otherwise, we don't want to download the log files)
    job_data = read_json_file(json_file_name)
    if not job_data:
        logger.warning(f"Error: Failed to read the JSON data from {json_file_name}.")
        return None, None
    if not job_data['job']['jobstatus'] == 'failed':
        logger.warning(f"Error: The job with PandaID {pandaid} is not in a failed state - nothing to explain.")
        return None, None
    logger.info(f"Confirmed that job {pandaid} is in a failed state.")

    # Fetch pilot error descriptions
    pilot_error_descriptions = read_json_file("pilot_error_codes_and_descriptions.json")
    if not pilot_error_descriptions:
        logger.warning(f"Error: Failed to read the pilot error descriptions.")
        return None, None

    # Fetch transform error descriptions
    transform_error_descriptions = read_json_file("trf_error_codes_and_descriptions.json")
    if not transform_error_descriptions:
        logger.warning(f"Error: Failed to read the transform error descriptions.")
        return None, None

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
        return None, None

    # Proceed to download the log files
    for log_file in log_files:
        log_file_name = fetch_data(pandaid, filename=log_file)
        if not log_file_name:
            logger.warning(f"Error: Failed to fetch the log file {log_file}.")
            return None, None

        # Keep track of the file names
        _file_dictionary[log_file] = log_file_name

        # Process the log file content as needed
        # For example, you can print it or analyze it further
        logger.info(f"Downloaded file: {log_file}, stored as {log_file_name}")

    return _file_dictionary, _metadata_dictionary


def extract_preceding_lines_streaming(log_file: str, error_string: str, num_lines: int = 200, output_file: str = None):
    """
    Extracts the preceding lines from a log file when a specific error string is found.

    Note: can handle very large files efficiently by using a sliding window approach.

    Args:
        log_file (str): The path to the log file to be analyzed.
        error_string (str): The error string to search for in the log file.
        num_lines (int): The number of preceding lines to extract (default is 200).
        output_file (str, optional): If provided, the extracted lines will be saved to this file.
    """
    buffer = deque(maxlen=num_lines)

    with open(log_file, 'r', encoding='utf-8') as file:
        for line in file:
            if error_string in line:
                # Match found; output the preceding lines
                if output_file:
                    with open(output_file, 'w') as out_file:
                        out_file.writelines(buffer)
                    logger.info(f"Extracted lines saved to: {output_file}")
                else:
                    logger.warning("".join(buffer))
                return
            buffer.append(line)

    logger.warning("Error string not found in the log file.")


def main():
    """
    Check if the correct number of command-line arguments is provided.

    This ensures that the script is executed with exactly two arguments:
    a question and a model.

    Raises:
        SystemExit: If the number of arguments is not equal to 3.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument('--log-files', type=str, required=True,
                        help='Comma-separated list of log files')
    parser.add_argument('--pandaid', type=int, required=True,
                        help='PandaID (integer)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use (e.g., openai, anthropic, etc.)')
    args = parser.parse_args()

    # Split the log files into a list
    log_files = args.log_files.split(',')

    # Fetch the files from PanDA
    file_dictionary, metadata_dictionary = fetch_all_data(args.pandaid, log_files)
    if not file_dictionary:
        logger.warning(f"Error: Failed to fetch files for PandaID {args.pandaid}.")
        sys.exit(1)
    logger.info(metadata_dictionary)
    # Extract the relevant parts for error analysis

    #answer = ask(question, model)
    #print(f"Answer from {model.capitalize()} (via RAG):\n{answer}")

if __name__ == "__main__":
    main()