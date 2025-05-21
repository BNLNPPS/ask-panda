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
import re
import requests
import sys
from fastmcp import FastMCP

from https import download_log_file

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


def fetch_file(panda_id: int, filename: str) -> str or None:
    """
    Fetches a given file from PanDA.

    Args:
        panda_id (int): The job or task ID.
        filename (str): The name of the file to fetch.

    Returns:
        str or None: The name of the downloaded file.
    """
    url = f"https://bigpanda.cern.ch/filebrowser/?pandaid={panda_id}&json&filename={filename}"
    response = download_log_file(url) #  post(url)
    if response and isinstance(response, str):
        return response
    if response:
        response = response.decode('utf-8')
        response = re.sub(r'([a-zA-Z0-9\])])(?=[A-Z])', r'\1\n', response)  # ensure that each line ends with \n
        return response
    else:
        return None


def main():
    """
    Check if the correct number of command-line arguments is provided.

    This ensures that the script is executed with exactly two arguments:
    a question and a model.

    Raises:
        SystemExit: If the number of arguments is not equal to 3.
    """
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

    # Fetch the log files from PanDA
    for log_file in log_files:
        log_file_name = fetch_file(args.pandaid, log_file)
        if not log_file_name:
            print(f"Error: Failed to fetch the log file {log_file}.")
            sys.exit(1)

        # Process the log file content as needed
        # For example, you can print it or analyze it further
        print(f"Downloaded file: {log_file}")

    # Extract the relevant parts for error analysis

    #answer = ask(question, model)
    #print(f"Answer from {model.capitalize()} (via RAG):\n{answer}")

if __name__ == "__main__":
    main()