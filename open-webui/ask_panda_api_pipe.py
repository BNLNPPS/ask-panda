"""
AskPanDA API Pipe for Open WebUI

This pipe calls the ask-panda HTTP API instead of importing Python code directly.
Much simpler and avoids dependency conflicts!
"""

from typing import Any, List, Dict
from pydantic import BaseModel, Field
import requests


class Pipe:
    class Valves(BaseModel):
        ask_panda_url: str = Field(
            default="http://localhost:8000/agent_ask",
            description="Ask PanDA endpoint",
        )
        model: str = Field(
            default="gemini",
            description="LLM model",
        )

    def __init__(self):
        self.name = "Ask PanDA"
        self.valves = self.Valves()

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[Dict[str, Any]],
        body: Dict[str, Any],
    ) -> str:
        try:
            r = requests.post(
                self.valves.ask_panda_url,
                json={"question": user_message, "model": self.valves.model},
                timeout=90,
            )
            r.raise_for_status()
            return r.json().get("answer", "No answer")
        except Exception as e:
            return f"Error: {e}"

