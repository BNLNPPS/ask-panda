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

"""This client can download task metadata from PanDA and ask an LLM to analyze the relevant parts."""

import argparse
import ast
# import asyncio
import logging
import io
import json
import os
import re
import requests
import sys
import tokenize
from collections import deque
# from fastmcp import FastMCP
from time import sleep

from tools.context_memory import ContextMemory
from tools.errorcodes import EC_NOTFOUND, EC_OK, EC_UNKNOWN_ERROR, EC_TIMEOUT
from tools.https import get_base_url
from tools.server_utils import MCP_SERVER_URL, check_server_health
from tools.tools import fetch_data, read_json_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("data_query.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
memory = ContextMemory()
# mcp = FastMCP("panda")
_CODEBLOCK_RE = re.compile(r"```(?:json|python)?\s*([\s\S]*?)```", re.IGNORECASE)
_VALUE_END_TOKENS = {tokenize.NUMBER, tokenize.STRING, tokenize.NAME}
_OPEN_TOKENS = {'{', '['}
_CLOSE_TOKENS = {'}', ']'}


def _strip_comments_tokenize(s: str) -> str:
    """Remove Python '#' comments without touching '#' inside strings."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(s).readline):
            if tok.type == tokenize.COMMENT:
                continue
            out.append(tok.string)
        return "".join(out)
    except Exception:
        return s


def _extract_first_code_block(text: str) -> str or None:
    # language tag optional (```json, ```python, or plain ```)
    m = re.search(r"```(?:\w+)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def _extract_top_level_brace(text: str) -> str | None:
    # Fallback: find first balanced {...} (simple but effective for these payloads)
    i = text.find("{")
    if i == -1:
        return None
    depth = 0
    in_str = False
    quote = ""
    esc = False
    for j, ch in enumerate(text[i:], start=i):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
        else:
            if ch in ("'", '"'):
                in_str = True
                quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[i:j + 1]
    return None


def parse_answer_to_dict(raw: str) -> dict:
    """
    Robust parser for LLM outputs that may include prose, code fences,
    Python-style comments, and JSON/Python differences.
    """
    s = (raw or "").strip()

    # Prefer fenced block if present
    block = _extract_first_code_block(s)
    if block:
        s = block

    # If we still don't start with '{', try to slice the first {...}
    if not s.lstrip().startswith("{"):
        maybe = _extract_top_level_brace(s)
        if maybe:
            s = maybe

    # 1) Try JSON
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) Try Python literal (accepts True/False/None, single quotes, etc.)
    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    # 3) Strip Python comments and try again
    try:
        cleaned = _strip_comments_tokenize(s)
        return ast.literal_eval(cleaned)
    except Exception as e:
        # Bubble a concise error upward; caller logs full context
        raise SyntaxError(f"Could not parse structured payload: {e}") from e


class TaskStatus:
    """
    A simple client that can give information about task status.
    This client fetches metadata from PanDA, extracts relevant parts, and asks an LLM for analysis.
    """
    def __init__(self, model: str, taskid: str, cache: str, session_id: str) -> None:
        """
        Initialize the TaskStatus client with a model.

        Args:
            model (str): The model to use for generating the answer (e.g., 'openai', 'anthropic').
            taskid (str): The PanDA job ID to analyze.
            cache (str): The location of the cache directory for storing downloaded files.
            session_id (str): The session ID for tracking the conversation.
        """
        self.model = model  # e.g., OpenAI or Anthropic wrapper
        try:
            self.taskid = int(taskid)  # PanDA task ID for the analysis
        except ValueError:
            logger.error(f"Invalid task ID: {taskid}. It should be an integer.")
            sys.exit(1)
        self.session_id = session_id
        self.cache = cache
        if not os.path.exists(self.cache):
            logger.info(f"Cache directory {self.cache} does not exist. Creating it.")
            try:
                os.makedirs(self.cache)
            except OSError as e:
                logger.error(f"Failed to create cache directory {self.cache}: {e}")
                sys.exit(1)

        path = os.path.join(os.path.join(self.cache, "tasks"), str(taskid))
        if not os.path.exists(path):
            logger.info(f"Creating directory for task {taskid} in cache.")
            try:
                os.makedirs(path)
            except OSError as e:
                logger.error(f"Failed to create directory {path}: {e}")
                sys.exit(1)

    def ask(self, question: str) -> str:
        """
        Send a question to the LLM and retrieve the answer.

        Args:
            question (str): The question to ask the LLM.

        Returns:
            str: The answer returned by the MCP server.
        """
        def _insert_missing_commas_between_members(s: str) -> str:
            """
            Insert commas between adjacent object members when the LLM forgot them, e.g.:
                "a": 1
                "b": 2
            becomes:
                "a": 1,
                "b": 2
            Heuristic rules (tokenize-based, so strings stay intact):
              - Only inside {...} (objects), not inside [...]
              - If a *value* (NUMBER/STRING/NAME True/False/None or closing '}'/']')
                is followed (across whitespace/comments/newlines) by a STRING token
                that looks like a new key (i.e., followed somewhere by a ':' before a comma/'}'),
                and there was no comma after the value, we inject a comma.
            """
            toks = list(tokenize.generate_tokens(io.StringIO(s).readline))
            out = []
            stack = []  # track '{' vs '[' contexts by OP tokens

            def is_true_false_none(tok):
                return tok.type == tokenize.NAME and tok.string in ('True', 'False', 'None')

            i = 0
            while i < len(toks):
                tok = toks[i]
                ttype, tstr = tok.type, tok.string

                # track braces/brackets
                if ttype == tokenize.OP and tstr in _OPEN_TOKENS:
                    stack.append(tstr)
                elif ttype == tokenize.OP and tstr in _CLOSE_TOKENS:
                    if stack:
                        stack.pop()

                out.append(tok)

                # Only attempt comma insertion if we're inside an object
                inside_object = bool(stack) and stack[-1] == '{'

                # Detect a value end token (or closing brace/bracket) that could end a member
                is_value_end = (
                    ttype in _VALUE_END_TOKENS and (
                        ttype != tokenize.NAME or is_true_false_none(tok)
                    )
                ) or (ttype == tokenize.OP and tstr in _CLOSE_TOKENS)

                if inside_object and is_value_end:
                    # Skip through whitespace/newlines/comments to peek the next significant token
                    j = i + 1
                    seen_comma = False
                    while j < len(toks):
                        ttype_j, tstr_j = toks[j].type, toks[j].string
                        if ttype_j == tokenize.OP and tstr_j == ',':
                            seen_comma = True
                            break
                        if ttype_j == tokenize.COMMENT or ttype_j in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                                                                      tokenize.DEDENT):
                            j += 1
                            continue
                        # Next significant token encountered
                        break

                    if not seen_comma and j < len(toks):
                        next_tok = toks[j]
                        # If the next significant token starts a new key: it should be a STRING then ':' later
                        if next_tok.type == tokenize.STRING:
                            # Look ahead for ':' before a ',' or '}' at the same brace depth
                            k = j + 1
                            brace_depth = 0
                            found_colon = False
                            while k < len(toks):
                                tt, ts = toks[k].type, toks[k].string
                                if tt == tokenize.OP:
                                    if ts in _OPEN_TOKENS:
                                        brace_depth += 1
                                    elif ts in _CLOSE_TOKENS:
                                        if brace_depth == 0:
                                            break
                                        brace_depth -= 1
                                    elif ts == ':' and brace_depth == 0:
                                        found_colon = True
                                        break
                                    elif ts in (',',) and brace_depth == 0:
                                        break
                                if tt in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT,
                                          tokenize.DEDENT):
                                    k += 1
                                    continue
                                k += 1
                            if found_colon:
                                # Inject a comma token right before the skipped trivia
                                out.insert(-1, tokenize.TokenInfo(type=tokenize.OP, string=',', start=tok.start,
                                                                  end=tok.end, line=tok.line))

                i += 1

            # Reconstruct source from tokens (tokenize.untokenize handles spacing)
            try:
                return tokenize.untokenize(out)
            except Exception:
                # Fallback: join strings if untokenize fails (rare)
                return ''.join(t.string for t in out)

        def _sanitize_llmish(s: str) -> str:
            """
            Sanitize common LLM artifacts so the payload becomes valid Python literal:
              - Remove trailing '%' after numeric literals:  96.19% -> 96.19
              - Remove parenthetical annotations that appear OUTSIDE strings:  "MWT2" (Midwest ..) -> "MWT2"
              - Insert missing commas between adjacent members inside objects
            """
            # Token pass 1: remove trailing % after numbers
            toks = list(tokenize.generate_tokens(io.StringIO(s).readline))
            out = []
            i = 0
            while i < len(toks):
                tok = toks[i]
                if tok.type == tokenize.NUMBER:
                    j = i + 1
                    if j < len(toks) and toks[j].type == tokenize.OP and toks[j].string == '%':
                        out.append(tok)  # keep number, drop '%'
                        i = j + 1
                        continue
                out.append(tok)
                i += 1
            s = tokenize.untokenize(out)

            # Token pass 2: strip parenthetical annotations outside strings
            toks = list(tokenize.generate_tokens(io.StringIO(s).readline))
            out = []
            i = 0
            while i < len(toks):
                tok = toks[i]
                if tok.type == tokenize.OP and tok.string == '(':
                    # Drop until matching ')'
                    depth = 1
                    i += 1
                    while i < len(toks) and depth > 0:
                        if toks[i].type == tokenize.OP:
                            if toks[i].string == '(':
                                depth += 1
                            elif toks[i].string == ')':
                                depth -= 1
                        i += 1
                    continue
                out.append(tok)
                i += 1
            s = tokenize.untokenize(out)

            # Token pass 3: insert missing commas
            s = _insert_missing_commas_between_members(s)

            return s

        def parse_answer_to_dict(raw: str) -> dict:
            s = (raw or "").strip()

            # Prefer code fence content if present
            block = _extract_first_code_block(s)
            if block:
                s = block

            # If not starting with '{', try to clip the first {...}
            if not s.lstrip().startswith("{"):
                maybe = _extract_top_level_brace(s)
                if maybe:
                    s = maybe

            # >>> Normalize LLM artifacts (%, comments, missing commas, annotations)
            s = _sanitize_llmish(s)

            # Try JSON
            try:
                return json.loads(s)
            except Exception:
                pass

            # Try Python literal
            try:
                return ast.literal_eval(s)
            except Exception:
                pass

            # Strip any remaining '#' comments and try again
            try:
                cleaned = _strip_comments_tokenize(s)
                return ast.literal_eval(cleaned)
            except Exception as e:
                raise SyntaxError(f"Could not parse structured payload: {e}") from e

        server_url = f"{MCP_SERVER_URL}/rag_ask"
        response = requests.post(server_url, json={"question": question, "model": self.model})
        if response.ok:
            answer = response.json()["answer"]
            if not answer:
                err = "No answer returned from the LLM."
                logger.error(f"{err}")
                return 'error: {err}'

            # Strip code block formatting
            try:
                clean_code = re.sub(r"^```(?:python)?\n|\n```$", "", answer.strip())
            except re.error as e:
                logger.error(f"Regex error while cleaning code: {e}")
                return 'error: {err}'

            # convert the answer to a Python dictionary
            try:
                answer_dict = ast.literal_eval(clean_code)
            except (SyntaxError, ValueError) as e:
                err = f"Error converting answer to dictionary: {e}"
                logger.error(f"{err}")
                return 'error: {err}'

            if not answer_dict:
                err = "Failed to store the answer as a Python dictionary."
                logger.error(f"{err}")
                return 'error: {err}'

            # format the answer for better readability
            formatted_answer = format_answer(answer_dict)
            if not formatted_answer:
                logger.error(f"Failed to format the answer from the LLM using: {answer_dict}")
                sys.exit(1)

            logger.info(f"Answer from {self.model.capitalize()}:\n{formatted_answer}")

            # store the answer in the session memory
            if self.session_id != "None":
                memory.store_turn(self.session_id, "Investigate the job failure", formatted_answer)
                logger.info(f"Answer stored in session ID {self.session_id}:\n\nquestion={question}\n\nanswer={formatted_answer}")

            return formatted_answer

        return f"requests.post() error: {response.text}"

    # async def fetch_all_data(self) -> tuple[int, dict or None, dict or None]:
    def fetch_all_data(self) -> tuple[int, dict or None, dict or None]:
        """
        Fetch metadata from PanDA for a given task ID.

        Returns:
            Exit code (int): The exit code indicating the status of the operation.
            File dictionary (dict): A dictionary containing the file names and their corresponding paths.
            Metadata dictionary (dict): A dictionary containing the relevant metadata for the task.
        """
        _metadata_dictionary = {}
        _file_dictionary = {}

        # Download metadata and pilot log concurrently
        workdir = os.path.join(self.cache, "tasks")
        base_url = get_base_url()
        # much info here (including all job ids belong to the task)
        # url = f"{base_url}/jobs/?jeditaskid={self.taskid}&json&mode=nodrop"
        url = f"{base_url}/task/{self.taskid}/?json"
        # metadata_task = asyncio.create_task(fetch_data(self.taskid, filename="metadata.json", jsondata=True, workdir=workdir, url=url))

        # Wait for download to complete
        # metadata_success, metadata_message = await metadata_task

        # metadata_success, metadata_message = await metadata_task
        metadata_success, metadata_message = fetch_data(self.taskid, filename="metadata.json", jsondata=True, workdir=workdir, url=url)

        if metadata_success != 0:
            logger.warning(f"Failed to fetch metadata for task {self.taskid} - will not be able to analyze the task status")
            return EC_NOTFOUND, _file_dictionary, _metadata_dictionary

        logger.info(f"Downloaded JSON file: {metadata_message}")
        _file_dictionary["json"] = metadata_message

        task_data = read_json_file(metadata_message)
        if not task_data:
            logger.warning(f"Error: Failed to read the JSON data from {metadata_message}.")
            return EC_UNKNOWN_ERROR, None, None

        # Extract relevant metadata from the JSON data
        try:
            _metadata_dictionary["jobs"] = {}
            for job in task_data['jobs']:
                jobstatus = job.get('jobstatus', 'unknown')
                if jobstatus not in _metadata_dictionary["jobs"]:
                    _metadata_dictionary["jobs"][jobstatus] = 0
                _metadata_dictionary["jobs"][jobstatus] += 1

                for key in job:
                    if "errordiag" in key:
                        if "errordiags" not in _metadata_dictionary:
                            _metadata_dictionary["errordiags"] = deque()
                        if job[key] != "":
                            _metadata_dictionary["errordiags"].append(job[key])
                    if "errorcode" in key:
                        if "errorcodes" not in _metadata_dictionary:
                            _metadata_dictionary["errorcodes"] = deque()
                        if job[key] > 0:
                            _metadata_dictionary["errorcodes"].append(job[key])
        except KeyError:
            _metadata_dictionary = task_data.copy()
            # logger.warning(f"Error: Missing key in JSON data: {e}")
            # return EC_UNKNOWN_ERROR, None, None

        return EC_OK, _file_dictionary, _metadata_dictionary

    def formulate_question(self, metadata_dictionary: dict) -> str:
        """
        Construct a question to ask the LLM based on the extracted lines and metadata.

        Args:
            metadata_dictionary:

        Returns:
            str: A formatted question string to be sent to the LLM.
        """
        # jobs = metadata_dictionary.get("jobs", None)
        # if not jobs:
        #    logger.warning("Error: No jobs information found in the metadata dictionary.")
        #    # return ""

        question = "You are an expert on distributed analysis.\n\n"
        question += """
Please provide a summary of the task status based on the metadata provided. The task is identified by its PanDA TaskID.
The dictionary should have the task id as the key (an integer), and its value should include the following fields:

"description": A detailed summary (long and thorough) of the task status in plain English. If there are no jobs, state that the task has not started yet.

"problems": A plain-language explanation of any issues (job failures as listed in the dictionary).

"details": A detailed analysis of the task status, including job counts per status, common error codes, and error diagnostics.

Return only a valid Python dictionary. Here's the metadata dictionary:
        """
        question = question.replace("TaskID", f"task ID ({self.taskid})")
        description = str(metadata_dictionary)
        question += f"\n\n{description}\n\n"

        return question

    def generate_question(self) -> str:
        """
        Generate a question to ask the LLM based on the task metadata.

        Returns:
            str: A formatted question string to be sent to the LLM.
        """
        # Fetch the files from PanDA
        # exit_code, file_dictionary, metadata_dictionary = asyncio.run(self.fetch_all_data())
        exit_code, file_dictionary, metadata_dictionary = self.fetch_all_data()
        logger.info(f"metadata_dictionary: {metadata_dictionary}")

        if exit_code == EC_NOTFOUND:
            logger.warning(
                f"No metadata found for task {self.taskid}")
        elif not file_dictionary:
            logger.warning(f"Error: Failed to metadata files for PandaID {self.taskid}.")
            sys.exit(1)

        # Formulate the question based on the extracted lines and metadata
        question = self.formulate_question(metadata_dictionary)
        if not question:
            logger.warning("No question could be generated.")
            sys.exit(1)

        return question


def format_answer(answer: dict) -> str:
    """
    Format the answer dictionary into a human-readable string.

    Args:
        answer (dict): The answer dictionary returned by the LLM.

    Returns:
        str: A formatted string containing the description, non-expert guidance, and expert guidance.
    """
    # the dictionary will only ever contain a single key-value pair
    logger.info(f"answer dictionary to format: {answer} type={type(answer)}")
    task_id, value = next(iter(answer.items()))

    to_store = ""

    # for metadata, the value is a string - otherwise a dictionary
    if isinstance(value, str):
        to_store += value
        return to_store

    description = value.get('description')
    if description:
        to_store += f"**Description:**\n{description}\n\n"

    problems = value.get('problems')
    if problems:
        to_store += f"**Problems:**\n{problems}\n\n"

    details = value.get('details')
    if details:
        to_store += f"**Details:**\n{details}\n\n"

    return to_store


def main():
    """
    Check if the correct number of command-line arguments is provided.

    This ensures that the script is executed with exactly two arguments:
    a question and a model.

    Raises:
        SystemExit: If the number of arguments is not equal to 4.
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

    parser.add_argument('--taskid', type=int, required=True,
                        help='PanDA TaskID (integer)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use (e.g., openai, anthropic, etc.)')
    parser.add_argument('--cache', type=str, default="cache",
                        help='Location of cache directory (default: cache)')
    parser.add_argument('--session-id', type=str, default="None",
                        help='Session ID for the context memory')
    args = parser.parse_args()

    client = TaskStatus(args.model, args.taskid, args.cache)

    # Generate a proper question to ask the LLM based on the metadata and log files
    question = client.generate_question()
    logger.info(f"Asking question: \n\n{question}")

    # Ask the question to the LLM
    answer = client.ask(question)

    logger.info(f"Answer from {args.model.capitalize()}:\n{answer}")

    # store the answer in the session memory
    # ..

    sys.exit(0)


if __name__ == "__main__":
    main()
