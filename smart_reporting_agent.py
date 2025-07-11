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
import asyncio
import json
import logging
import os
import psutil
import re
import requests
import sys

from collections import deque
from fastmcp import FastMCP
from time import sleep
from typing import Optional

import errorcodes
from https import download_data
from server import MCP_SERVER_URL, check_server_health
from tools import fetch_data, read_json_file, read_file

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


def main():
    """
    Check if the correct number of command-line arguments is provided.
    """
    # Check server health before proceeding
    ec = check_server_health()
    if ec == errorcodes.EC_TIMEOUT:
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

    parser.add_argument('--url', type=str, required=True,
                        help='Data source URL')
    parser.add_argument('--pid', type=str, required=True,
                        help='MCP server process ID')

    args = parser.parse_args()

    # download error data from the given url every 24h, 12h, 6h, 3h, 1h
    prefix = "error_data"
    error_data = {24: f"{prefix}_24h.json",
                  12: f"{prefix}_12h.json",
                  6: f"{prefix}_6h.json",
                  3: f"{prefix}_3h.json",
                  1: f"{prefix}_1h.json"}

    # go into a loop until the user interrupts
    while True:
        # Verify that the MCP server is still running
        if not psutil.pid_exists(int(args.pid)):
            logger.error(f"MCP server with PID {args.pid} is no longer running. Exiting.")
            sys.exit(1)

        # Sleep
        sleep(10)

if __name__ == "__main__":
    main()
