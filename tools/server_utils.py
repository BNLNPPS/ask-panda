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

"""Utility functions for server management."""

import logging
import os
import requests

from tools.errorcodes import EC_OK, EC_SERVERNOTRUNNING, EC_CONNECTIONPROBLEM, EC_TIMEOUT, EC_UNKNOWN_ERROR

# MCP server IP and env vars
MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("server_utils.log"),
        logging.StreamHandler()  # Optional: keeps logs visible in console too
    ]
)
logger = logging.getLogger(__name__)


def check_server_health() -> int:
    """
    Check the health of the MCP server.
    This function attempts to connect to the MCP server's health endpoint
    and checks if it returns a status of "ok".
    If the server is reachable and healthy, it returns EC_OK.
    If the server is not running or there is a connection problem,
    it returns EC_SERVERNOTRUNNING or EC_CONNECTIONPROBLEM respectively.

    Returns:
        Server error code (int): Exit code indicating the health status of the MCP server.
    """
    try:
        response = requests.get(f"{MCP_SERVER_URL}/health", timeout=5)
        response.raise_for_status()
        if response.json().get("status") == "ok":
            logger.info("MCP server is running.")
            return EC_OK
        logger.warning("MCP server is not running.")
        return EC_SERVERNOTRUNNING
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout):
        logger.warning(f"Timeout while trying to connect to MCP server at {MCP_SERVER_URL}.")
        return EC_TIMEOUT
    except requests.RequestException as e:
        logger.warning(f"Cannot connect to MCP server at {MCP_SERVER_URL}: {e}")
        return EC_CONNECTIONPROBLEM
    except Exception as e:
        logger.error(f"An unexpected error occurred while checking MCP server health: {e}")
        return EC_UNKNOWN_ERROR
