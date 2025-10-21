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

import argparse
import logging
import os
import psutil
import sys
import time

from fastmcp import FastMCP
from time import sleep

from tools.errorcodes import EC_TIMEOUT
from tools.https import download_data
from tools.server_utils import MCP_SERVER_URL, check_server_health
from tools.tools import reformat_errors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("maintenance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

mcp = FastMCP("panda")


def main():
    """
    Check if the correct number of command-line arguments is provided.
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

    parser.add_argument('--pid', type=str, required=True,
                        help='MCP server process ID')
    parser.add_argument('--cache-dir', type=str, required=True,
                        help='Cache directory for storing downloaded data')

    args = parser.parse_args()

    # download error data from the given url every 24h, 12h, 6h, 3h, 1h
    url_errors = "https://bigpanda.cern.ch/errors/?json&hours=NNN&limit=10000000&fields=errsBySite"
    prefix = "error_data"
    error_data = {24: f"{prefix}_24h.json",
                  12: f"{prefix}_12h.json",
                  6: f"{prefix}_6h.json",
                  3: f"{prefix}_3h.json",
                  1: f"{prefix}_1h.json"}

    # If the cache directory does not exist, create it
    if not os.path.exists(args.cache_dir):
        logger.info(f"Creating cache directory: {args.cache_dir}")
        os.makedirs(args.cache_dir)

    # go into a loop until the user interrupts
    iteration = 0
    while True:
        # Verify that the MCP server is still running
        if not psutil.pid_exists(int(args.pid)):
            logger.error(f"MCP server with PID {args.pid} is no longer running. Exiting mainenance client.")
            sys.exit(1)

        # Download error data from the given URL
        for key in error_data:
            filename = error_data[key]
            path = os.path.join(args.cache_dir, filename)
            messy_path = os.path.join(args.cache_dir, filename.replace(".json", "_messy.json"))
            url = url_errors.replace("NNN", str(key))

            if not os.path.exists(path):
                logger.info(f"Downloading {filename} from {url}")
                exit_code, _ = download_data(url, filename=messy_path)
                if exit_code != 0:
                    logger.error(f"Failed to download {filename}")
            else:
                # If the file exists, check if it is older than N hours
                file_mtime = os.path.getmtime(path)
                file_age_hours = (time.time() - file_mtime) / 3600
                if file_age_hours > key:
                    logger.info(f"File {filename} is older than {key} hours, downloading again.")
                    exit_code, _ = download_data(url, filename=path)
                    if exit_code != 0:
                        logger.error(f"Failed to download {filename}")

            # Reformat the error data
            if os.path.exists(messy_path):
                _ = reformat_errors(messy_path, path)

                # Remove the messy file
                logger.info(f"Removing messy file: {messy_path}")
                os.remove(messy_path)

        # Download the CRIC data once per 10 minutes
        # .. see Claude code

        # Sleep
        logger.info(f"iteration #{iteration}")
        iteration += 1
        sleep(10)


if __name__ == "__main__":
    main()
