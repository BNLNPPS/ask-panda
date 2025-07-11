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

import json
import logging
import re
from typing import Optional

from tools.errorcodes import EC_OK, EC_NOTFOUND, EC_UNKNOWN_ERROR
from tools.https import download_data

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
    exit_code, response = download_data(url, prefix=filename) #  post(url)
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
