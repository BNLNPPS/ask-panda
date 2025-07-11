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

"""This module provides tools for the MCP server and agents,"""

# entries = read_json("error_data_24h.json")
# errors = extract_errors(entries) -> can be written to new file and read back when needed
# component_errors = extract_component_errors("bnl", "pilot", errors)
# count = error_count(1305, component_errors)
# queue_errors = error_counts(1305, errors)

import json
import logging
import os
import re
from typing import Optional

from tools.errorcodes import EC_OK, EC_NOTFOUND, EC_UNKNOWN_ERROR
from tools.https import download_data
from tools.vectorstore_manager import VectorStoreManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("tools.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def fetch_data(panda_id: int, filename: str = None, jsondata: bool = False, url: str = None) -> tuple[int, Optional[str]]:
    """
    Fetches a given file from PanDA.

    Args:
        panda_id (int): The job or task ID.
        filename (str): The name of the file to fetch.
        jsondata (bool): If True, return a JSON string for the job.
        url (str, optional): If provided, use this URL instead of constructing one.
    Returns:
        str or None: The name of the downloaded file.
        exit_code (int): The exit code indicating the status of the operation.
    """
    if not url:
        url = (
            f"https://bigpanda.cern.ch/job?pandaid={panda_id}&json"
            if jsondata
            else f"https://bigpanda.cern.ch/filebrowser/?pandaid={panda_id}&json&filename={filename}"
        )
    logger.info(f"Downloading file from: {url}")

    # Use the download_data function to fetch the file - it will return an exit code and the filename
    exit_code, response = download_data(url, prefix=filename)
    if exit_code == EC_NOTFOUND:
        logger.error(f"File not found for PandaID {panda_id} with filename {filename}.")
        return exit_code, None
    elif exit_code == EC_UNKNOWN_ERROR:
        logger.error(f"Unknown error occurred while fetching data for PandaID {panda_id} with filename {filename}.")
        return exit_code, None

    if response and isinstance(response, str):
        return EC_OK, response
    if response:
        response = response.decode('utf-8')
        response = re.sub(r'([a-zA-Z0-9\])])(?=[A-Z])', r'\1\n', response)  # ensure that each line ends with \n
        return EC_OK, response
    else:
        logger.error(f"Failed to fetch data for PandaID {panda_id} with filename {filename}.")
        return EC_UNKNOWN_ERROR, None


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


def get_vectorstore_manager(resources_dir: str, chroma_dir: str) -> VectorStoreManager:
    """
    Return the VectorStoreManager instance.

    Args:
        resources_dir (str): The path to the resources directory.
        chroma_dir (str): The path to the ChromaDB directory.
    Returns:
        VectorStoreManager: The instance of the VectorStoreManager.
    """
    if not chroma_dir.exists():
        # create the directory if it does not exist
        os.makedirs(chroma_dir, exist_ok=True)

    if not resources_dir.exists():
        logger.error(f"Resources directory {resources_dir} does not exist.")
        return None
    if not chroma_dir.exists():
        logger.error(f"ChromaDB directory {chroma_dir} does not exist.")
        return None

    return VectorStoreManager(resources_dir, chroma_dir)


def read_error_data_json(filename: str) -> list:
    """
    Read a JSON file containing error info and return the 'errsBySite' data.

    Args:
        filename (str): The path to the JSON file.
    Returns:
        list: 'errsBySite' data.
    """
    with open(filename, "r") as f:
        data = json.load(f)

    return data.get('errsBySite')


def extract_errors(entries: list) -> dict:
    """
    Extract errors from a list of entries.

    Args:
        entries (list): A list of entries, where each entry is a dictionary containing 'name' and 'errors'.
    Returns:
        dict: A dictionary where the keys are the 'name' of each entry and the values are the corresponding 'errors'.
    """
    errors = {}
    for entry in entries:
        errors[entry.get('name')] = entry.get('errors')

    return errors


def reformat_errors(infile: str, outfile: str) -> bool:
    """
    Reformat the error data from a JSON file and save it to a new file for later use.

    Args:
        infile (str): The path to the input JSON file containing messy error data.
        outfile (str): The path to the output JSON file where the reformatted data will be saved.
    Returns:
        bool: True if the reformatting was successful, False otherwise.
    """
    # Get the messy error data from the input file
    entries = read_error_data_json(infile)
    if not entries:
        logger.error(f"Failed to read error data from {infile}.")
        return False

    # Extract errors from the entries
    errors = extract_errors(entries)
    if not errors:
        logger.error("No errors found in the entries.")
        return False

    # Save the reformatted errors to the output file
    try:
        with open(outfile, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=4)
        logger.info(f"Reformatted errors saved to {outfile}.")
        return True
    except IOError as e:
        logger.error(f"Failed to write reformatted errors to {outfile}: {e}")
        return False


def extract_component_errors(queue: str, component: str, errors) -> dict:
    """
    Extract errors for a specific component from the errors dictionary for a given queue.

    Output example: {'1305': 889, '1150': 38, '1324': 2, '1201': 3, '1213': 135, '1361': 21, .. }

    Args:
        queue (str): The name of the queue (e.g., 'pilot', 'pilot2').
        component (str): The name of the component to filter errors for (e.g., 'pilot', 'pilot2').
        errors (dict): A dictionary containing error information, where keys are queue names and values are lists of errors.
    Returns:
        dict: A dictionary containing error codes and their counts for the specified component in the specified queue.
    """
    component_errors = {}
    for key in errors.keys():
        if queue == key.lower():
            queue_errors = errors.get(key)
            for error in queue_errors:
                if component in error:
                    info = queue_errors.get(error)
                    count = info.get('count')
                    code = error[error.find(':') + 1:]
                    component_errors[code] = count

    return component_errors


def error_count(code: int, component_errors: dict) -> int:
    """
    Get the count of a specific error code from the component errors dictionary.

    Args:
        code (int): The error code to look for.
        component_errors (dict): A dictionary containing error codes and their counts for a specific component.
    Returns:
        int: The count of the error code for the specified component in the specified queue.
    """
    return component_errors.get(str(code), 0)


def error_counts(code: int, errors: dict) -> dict:
    """
    Get the counts of a specific error code across all queues.

    Example output: {'FZK-LCG2': 990, 'EMMY_KIT': 487, 'UKI-LT2-QMUL': 1434, 'TRIUMF': 1456, .. }

    Args:
        code (int): The error code to look for.
        errors (dict): A dictionary containing error codes and their counts for a specific queue.
    Returns:
        dict: A dictionary where keys are queue names and values are the counts of the specified error code.
    """
    total_queue_errors = {}
    for key in errors.keys():  # queue names
        queue_errors = errors.get(key)
        for error in queue_errors:
            if str(code) in error:
                info = queue_errors.get(error)
                count = info.get('count')
                total_queue_errors[key] = count

    return total_queue_errors
