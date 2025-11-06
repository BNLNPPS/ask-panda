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

"""Handles conversational memory using an SQLite-backed store."""

import datetime
import os
import sqlite3
from typing import List, Optional, Callable, Tuple

DB_DEFAULT_PATH = os.path.expanduser("cache/context_memory.sqlite")


class ContextMemory:
    """Handle conversational memory using an SQLite-backed store."""

    def __init__(self, db_path: str = DB_DEFAULT_PATH) -> None:
        """
        Initialize the database and memory system.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the conversation table if it doesn't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS conversation (
                    session_id TEXT,
                    timestamp TEXT,
                    user_input TEXT,
                    agent_response TEXT
                )
            """)
            conn.commit()

    def store_turn(self, session_id: str, user_input: str, agent_response: str) -> None:
        """
        Store a user-client interaction in the conversation log.

        Args:
            session_id (str): Unique ID for the conversation session.
            user_input (str): The user query or prompt.
            agent_response (str): The response from the client/LLM.
        """
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO conversation (session_id, timestamp, user_input, agent_response)
                VALUES (?, ?, ?, ?)
            """, (session_id, datetime.datetime.now(datetime.UTC).isoformat(), user_input, agent_response))
            conn.commit()

    def get_history(self, session_id: str, max_turns: int = 4) -> List[Tuple[str, str]]:
        """
        Retrieve recent conversation history for a given session.

        Args:
            session_id (str): The session identifier.
            max_turns (int): Maximum number of previous turns to retrieve.

        Returns:
            List[Tuple[str, str]]: A list of (user_input, agent_response) pairs.
        """
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT user_input, agent_response FROM conversation
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, max_turns))
            return list(reversed(c.fetchall()))

    def summarize_history(
        self,
        session_id: str,
        max_turns: int = 2,
        summarize_fn: Optional[Callable[[str], str]] = None
    ) -> str:
        """
        Summarize the recent conversation history using a user-provided summarizer.

        Args:
            session_id (str): The session identifier.
            max_turns (int): Maximum number of recent turns to include.
            summarize_fn (Optional[Callable[[str], str]]): A function that takes
                a text prompt and returns a summary using an LLM.

        Returns:
            str: The conversation summary, or the summary prompt if no function is provided.
        """
        history = self.get_history(session_id, max_turns)
        if not history:
            return "No conversation history available."

        conversation_text = ""
        for user_input, agent_response in history:
            conversation_text += f"User: {user_input}\nAssistant: {agent_response}\n"

        prompt = (
            "Summarize the following conversation between a user and an assistant:\n\n"
            f"{conversation_text}\n\nSummary:"
        )

        if summarize_fn:
            return summarize_fn(prompt)
        return prompt
