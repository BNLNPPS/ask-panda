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

# This module can be used to download files from PanDA monitor
# It is using OIDC tokens, i.e. the env vars below must be set.
#
# OIDC tokens for pilot-PanDA server communications:
#
# export OIDC_AUTH_TOKEN=panda_token
# (export OIDC_AUTH_TOKEN=c3d71281d64881e87e24415865b78e48)
# export OIDC_AUTH_ORIGIN=atlas.pilot
# export PANDA_AUTH_TOKEN_KEY=panda_token_key

import json
import os
import platform
import requests
import shlex
import ssl
import sys
import tempfile
import urllib.request
import urllib.error
import urllib.parse
from urllib.parse import parse_qs
from typing import IO

EC_OK = 0
EC_NOTFOUND = 1
EC_UNKNOWN_ERROR = 2


def open_file(filename: str, mode: str) -> IO:
    """
    Open and return a file pointer for the given mode.

    Note: the caller needs to close the file.

    :param filename: file name (str)
    :param mode: file mode (str)
    :raises PilotException: FileHandlingFailure
    :return: file pointer (IO).
    """
    _file = None
    try:
        _file = open(filename, mode, encoding='utf-8')
    except IOError as exc:
        print(f"Exception caught: {exc}")

    return _file


def read_file(filename: str, mode: str = 'r') -> str:
    """
    Open, read and close a file.

    :param filename: file name (str)
    :param mode: file mode (str)
    :return: file contents (str).
    """
    out = ""
    _file = open_file(filename, mode)
    if _file:
        out = _file.read()
        _file.close()

    return out


def locate_token(auth_token: str, key: bool = False) -> str:
    """
    Locate the OIDC token file.

    Primary means the original token file, not the refreshed one.
    The primary token is needed for downloading new tokens (i.e. 'refreshed' ones).

    Note that auth_token is only the file name for the primary token, but has the full path for any
    refreshed token.

    :param auth_token: file name of token (str)
    :param key: if true, token key is used (bool)
    :return: path to token (str).
    """
    primary_basedir = os.getcwd()
    if auth_token:
        paths = [os.path.join(primary_basedir, auth_token),
                 os.path.join(os.environ.get('HOME', ''), auth_token)]
    else:
        paths = []

    # remove duplicates
    paths = list(set(paths))

    path = ""
    for _path in paths:
        if os.path.exists(_path):
            print(f'found {_path}')
            path = _path
            break

    if path == "":
        print(f'did not find any local token file ({auth_token}) in paths={paths}')

    return path


def get_auth_token_content(auth_token: str, key: bool = False) -> str:
    """
    Get the content of the auth token.

    :param auth_token: token name (str)
    :param key: if true, token key is used (bool)
    :return: token content (str).
    """
    path = locate_token(auth_token, key=key)
    if os.path.exists(path):
        auth_token_content = read_file(path)
        if not auth_token_content:
            print(f'failed to read file {path}')
            return ""
        else:
            print(f'read contents from file {path} (length = {len(auth_token_content)})')
    else:
        print(f'path does not exist: {path}')
        return ""

    return auth_token_content


def get_ssl_context() -> ssl.SSLContext:
    """
    Get the SSL context.

    :return: SSL context (ssl.SSLContext).
    """
    try:  # for ssl version 3.0 and python 3.10+
        ssl_context = ssl.SSLContext(protocol=None)
    except Exception:  # for ssl version 1.0
        ssl_context = ssl.SSLContext()

    return ssl_context


def get_headers(use_oidc_token: bool, auth_token_content: str = None, auth_origin: str = None, content_type: str = "application/json") -> dict:
    """
    Get the headers for the request.

    :param use_oidc_token: True if OIDC token should be used (bool)
    :param auth_token_content: token content (str)
    :param auth_origin: token origin (str)
    :return: headers (dict).
    """
    if use_oidc_token:
        headers = {
            "Authorization": f"Bearer {shlex.quote(auth_token_content)}",
            "Origin": shlex.quote(auth_origin),
        }
    else:
        headers = {}

    # always add the user agent
    headers["User-Agent"] =  f'Ask_PanDA/1.0.0 (Python {sys.version.split()[0]}; {platform.system()} {platform.machine()})'

    # only add the content type if there is a body to send (that is of type application/json)
    if content_type:
        headers["Content-Type"] = content_type

    return headers


def get_local_oidc_token_info() -> tuple[str or None, str or None]:
    """
    Get the OIDC token locally.

    :return: token (str), token origin (str) (tuple).
    """
    auth_token = os.environ.get('OIDC_AUTH_TOKEN', os.environ.get('PANDA_AUTH_TOKEN'))

    # origin of the token (panda_dev.pilot, ..)
    auth_origin = os.environ.get('OIDC_AUTH_ORIGIN', os.environ.get('PANDA_AUTH_ORIGIN'))

    return auth_token, auth_origin


def request2(url: str = "", data: dict = None, secure: bool = True) -> str or dict:  # noqa: C901
    """
    Send a request using HTTPS (using urllib module).

    :param url: the URL of the resource (str)
    :param data: data to send (dict)
    :param secure: use secure connection (bool)
    :param compressed: compress data (bool)
    :return: server response (str or dict).
    """
    if data is None:
        data = {}

    # should tokens be used?
    auth_token, auth_origin = get_local_oidc_token_info()
    use_oidc_token = auth_token and auth_origin
    auth_token_content = get_auth_token_content(auth_token) if use_oidc_token else ""
    if not auth_token_content and use_oidc_token:
        print('OIDC_AUTH_TOKEN/PANDA_AUTH_TOKEN content could not be read')
        return ""

    # get the relevant headers
    headers = get_headers(use_oidc_token, auth_token_content, auth_origin)
    print(f'headers = {hide_token(headers.copy())}')
    print(f'data = {data}')

    # set up the request
    data_json = {}  # to implement
    req = urllib.request.Request(url, data_json, headers=headers)

    # create a context with certificate verification
    ssl_context = get_ssl_context()

    if not secure:
        ssl_context.verify_mode = False
        ssl_context.check_hostname = False

    # Send the request securely
    try:
        print('sending data to server')
        with urllib.request.urlopen(req, context=ssl_context, timeout=60) as response:
            # Handle the response here
            print(f"response.status={response.status}, response.reason={response.reason}")
            ret = response.read().decode('utf-8')
            if 'getProxy' not in url:
                print(f"response={ret}")
        print('sent request to server')
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        print(f'failed to send request: {exc}')
        ret = ""
    else:
        if secure and isinstance(ret, str):
            if ret == 'Succeeded':  # this happens for sending modeOn (debug mode)
                ret = {'StatusCode': '0'}
            elif ret.startswith('{') and ret.endswith('}'):
                try:
                    ret = json.loads(ret)
                except json.JSONDecodeError as e:
                    print(f'failed to parse response: {e}')
            else:  # response="StatusCode=_some number_"
                # Parse the query string into a dictionary
                query_dict = parse_qs(ret)

                # Convert lists to single values
                ret = {k: v[0] if len(v) == 1 else v for k, v in query_dict.items()}

    return ret


def hide_token(headers: dict) -> dict:
    """
    Hide the token in the headers.

    :param headers: Copy of headers (dict)
    :return: headers with token hidden (dict).
    """
    if 'Authorization' in headers:
        headers['Authorization'] = 'Bearer ********'

    return headers


def download_data(url: str) -> tuple[int, str]:
    """
    Download a log file or JSON from the given URL and save it to a temporary file.

    Args:
        url (str): The URL of the log file to download.

    Returns:
        file name (str or None): The filename of the downloaded file, or None in case of failure.
        exit code (int): 0 if successful, non-zero if an error occurred.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        if "Not Found for url" in str(e):
            return EC_NOTFOUND, None
        else:
            return EC_UNKNOWN_ERROR, None
    with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)

        return EC_OK, tmp_file.name  # Return the file path

    return EC_UNKNOWN_ERROR, None


def download_file(url: str, timeout: int = 20, headers: dict = None) -> str:
    """
    Download url content.

    The optional headers should in fact be used for downloading OIDC tokens.

    :param url: url (str)
    :param timeout: optional timeout (int)
    :param headers: optional headers (dict)
    :return: url content (str).
    """
    # first get the token key
    token_key = os.environ.get("PANDA_AUTH_TOKEN_KEY")
    if not token_key:
        print('PANDA_AUTH_TOKEN_KEY is not set')
        #return False

    panda_token_key = get_auth_token_content(token_key, key=True)
    if panda_token_key:
        print(f'read token key: {token_key}')
    else:
        print('failed to get panda_token_key')
        #return False

    # now get the actual token
    auth_token, auth_origin = get_local_oidc_token_info()
    auth_token_content = get_auth_token_content(auth_token)
    if not auth_token_content:
        print(f'failed to get auth token content for {auth_token}')
        #return False

    headers = get_headers(True, auth_token_content, auth_origin, content_type=None)

    # define the request headers
    if headers is None:
        headers = {"User-Agent": "Ask_PanDA-1.0.0"}
    print(f"headers = {hide_token(headers.copy())}")

    # create a context with certificate verification
    ssl_context = get_ssl_context()

    print(f"url={url}")
    req = urllib.request.Request(url, headers=headers)

    # download the file
    try:
        with urllib.request.urlopen(req, context=ssl_context, timeout=timeout) as response:
            content = response.read()
    except urllib.error.URLError as exc:
        print(f"error occurred with urlopen: {exc.reason}")
        # Handle the error, set content to None or handle as needed
        content = "Failed to download the log file - cannot assist with this error"

    return content

# Example
# panda_type = "job"
# panda_id = "6473373419"
# url = f"https://bigpanda.cern.ch/{panda_type}?pandaid={panda_id}&json"
# print(f"url={url}")
# document = download_file(url)
# if document:
#     document = document.decode('utf-8')
#     print(str(document))
# else:
#     print("failed to download document")
