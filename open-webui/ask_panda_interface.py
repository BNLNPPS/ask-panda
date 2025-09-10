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

"""Document Query Interface for Open WebUI"""

from pydantic import BaseModel, Field
from typing import Optional
from agents.selection_agent import SelectionAgent, figure_out_agents


class Pipe:
    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        show_status: bool = Field(
            default=True, description="Show status of the function."
        )

    def __init__(self):
        self.valves = self.Valves()

    async def pipe(
        self,
        body: dict,
        __user__: dict = None,
        __metadata__: dict = None,
        __messages__: dict = None,
        __event_emitter__: dict = None,
        __event_call__: dict = None,
    ) -> Optional[dict]:
        chat_id = (__metadata__ or {}).get("chat_id")
        meta_type = (__metadata__ or {}).get(
            "type"
        )  # may be "user_response" for real turns

        user_valves = __user__.get("valves") if __user__ else None
        if not user_valves:
            user_valves = self.UserValves()

        model = "gemini"
        # user_id = __user__.get("id")
        last_assistant_message = body["messages"][-1]
        prompt = last_assistant_message["content"]

        # --- NEW: detect follow-up/aux calls ---
        # Treat UI-generated follow-ups (###...) or any non-user_response meta_type as "follow-up mode"
        is_followup = bool(
            prompt.startswith("###") or (meta_type and meta_type != "user_response")
        )

        if not prompt.startswith("###"):
            session_id = chat_id
        else:
            session_id = "None"  # do not store follow-up suggestions etc from the UI

        print(f"session id={session_id}")
        print(f"prompt: {prompt}")
        print(f"is_followup: {is_followup} (meta_type={meta_type})")  # NEW: debug

        if user_valves.show_status and __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Processing your input", "done": False},
                }
            )
            print(f"__event_emitter__={__event_emitter__}")

        try:
            agents = figure_out_agents(
                prompt,
                model,
                session_id,
                cache="/Users/nilsnilsson/Development/ask-panda/cache",
            )
            selection_agent = SelectionAgent(agents, model)
            category = selection_agent.answer(prompt)

            # --- NEW: if a follow-up would route to log_analyzer, override to document ---
            if is_followup and category != "document":
                # Follow-up detected: overriding category to 'document'
                category = "document"

            agent = agents.get(category)
            print(f"Selected agent category: {category}")

            # --- OPTIONAL: normalize follow-up prompt for document agent by stripping leading ### ---
            doc_prompt = (
                prompt.lstrip("#").strip()
                if (is_followup and category == "document")
                else prompt
            )

            if category == "document":
                print(f"Selected agent category: {category} (DocumentQueryAgent)")
                answer = agent.ask(doc_prompt)  # use normalized prompt on follow-ups
            elif category == "log_analyzer":
                print(f"Selected agent category: {category} (LogAnalysisAgent)")
                question = agent.generate_question("pilotlog.txt")
                answer = agent.ask(question)
            elif category == "task":
                print(f"Selected agent category: {category} (TaskStatusAgent)")
                question = agent.generate_question()
                answer = agent.ask(question)
            else:
                answer = "Not yet implemented"
                print(answer)
                return "Sorry, I don’t have enough information to answer that kind of question."
        except Exception as e:
            answer = f"[ERROR] {type(e).__name__}: {str(e)}"
            final_answer = answer

        # the returned answer is a dictionary with the format
        #                     answer = {
        #                         "session_id": self.session_id,
        #                         "question": question,
        #                         "model": self.model,
        #                         "answer": answer
        #                     }
        if isinstance(answer, dict):
            final_answer = answer.get("answer", "No answer provided")
        else:
            final_answer = answer

        print(f"Answer: {final_answer}")

        if user_valves.show_status and __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Response ready", "done": True},
                }
            )

        # response = {"answer": final_answer, "follow_ups": []}
        return final_answer  # json.dumps(response)
