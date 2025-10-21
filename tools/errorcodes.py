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

"""Error codes for the MCP server and clients."""

EC_OK = 0  # Success
EC_SERVERNOTRUNNING = 1  # MCP server is not running
EC_CONNECTIONPROBLEM = 2  # MCP server connection problem
EC_TIMEOUT = 3  # MCP server timeout
EC_NOTFOUND = 4  # Log file not found
EC_UNKNOWN_ERROR = 5  # Unknown error occurred
