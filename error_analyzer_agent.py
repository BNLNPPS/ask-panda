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
import re
import requests
import sys
from fastmcp import FastMCP

from https import download_data

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


def fetch_data(panda_id: int, filename: str = None, jsondata: bool = False) -> str or None:
    """
    Fetches a given file from PanDA.

    Args:
        panda_id (int): The job or task ID.
        filename (str): The name of the file to fetch.
        jsondata (bool): If True, return a JSON string for the job.

    Returns:
        str or None: The name of the downloaded file.
    """
    if jsondata:
        url = f"https://bigpanda.cern.ch/job?pandaid={panda_id}&json"
    else:
        url = f"https://bigpanda.cern.ch/filebrowser/?pandaid={panda_id}&json&filename={filename}"
    print(f"Will download file from: {url}")

    response = download_data(url) #  post(url)
    if response and isinstance(response, str):
        return response
    if response:
        response = response.decode('utf-8')
        response = re.sub(r'([a-zA-Z0-9\])])(?=[A-Z])', r'\1\n', response)  # ensure that each line ends with \n
        return response
    else:
        return None


def read_json_file(file_path: str) -> dict or None:
    """
    Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict or None: The contents of the JSON file as a dictionary, or None if the file cannot be read.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def fetch_all_files(pandaid: int, log_files: list) -> dict or None:
    """
    Fetches all files from PanDA for a given job ID.

    Args:
        pandaid (int): The panda job ID.
        log_files (list): A list of log files to fetch.

    Returns:
        File dictionary (dict): A dictionary containing the file names and their corresponding paths.
    """
    _file_dictionary = {}
    json_file_name = fetch_data(pandaid, jsondata=True)
    if not json_file_name:
        print(f"Error: Failed to fetch the JSON data for PandaID {args.pandaid}.")
        return None
    print(f"Downloaded JSON file: {json_file_name}")
    _file_dictionary["json"] = json_file_name

    # Verify that the current job is actually a failed job (otherwise, we don't want to download the log files)
    job_data = read_json_file(json_file_name)
    if not job_data:
        print(f"Error: Failed to read the JSON data from {json_file_name}.")
        return None
    if not job_data['files'][0]['status'] == 'failed':
        print(f"Error: The job with PandaID {pandaid} is not in a failed state - nothing to explain.")
        return None
    print(f"Confirmed that job {pandaid} is in a failed state.")

    # Proceed to download the log files
    for log_file in log_files:
        log_file_name = fetch_data(pandaid, filename=log_file)
        if not log_file_name:
            print(f"Error: Failed to fetch the log file {log_file}.")
            return None

        # Keep track of the file names
        _file_dictionary[log_file] = log_file_name

        # Process the log file content as needed
        # For example, you can print it or analyze it further
        print(f"Downloaded file: {log_file}, stored as {log_file_name}")

    return _file_dictionary


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
    file_dictionary = fetch_all_files(args.pandaid, log_files)
    if not file_dictionary:
        print(f"Error: Failed to fetch files for PandaID {args.pandaid}.")
        sys.exit(1)

    # Extract the relevant parts for error analysis

    #answer = ask(question, model)
    #print(f"Answer from {model.capitalize()} (via RAG):\n{answer}")

if __name__ == "__main__":
    main()