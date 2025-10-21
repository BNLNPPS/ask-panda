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
from clients.selection import Selection, figure_out_clients, get_id


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

        # turn off follow ups for now

        # 1) Hard-stop if the UI is calling the follow-up/title/notes generators
        #    (__event_call__ naming is present in recent builds; keep the content-based fallback as well)
        if __event_call__ in {"follow_ups", "followups", "title", "notes"}:
            # Return an explicit empty structure so OWUI has nothing to show
            return {"follow_ups": []}  # benign, ignored by chat renderer

        # 2) Fallback heuristic: detect the default follow-up template in the messages
        #    (Open WebUI’s default follow-up prompt contains wording like below)
        sys_texts = " \n".join(
            m.get("content", "")
            for m in body.get("messages", [])
            if m.get("role") in ("system", "assistant")
        ).lower()
        if (
            "suggest 3-5 relevant follow-up" in sys_texts
            or "follow-up questions" in sys_texts
        ):
            return {"follow_ups": []}

        # end stop follow ups

        user_valves = __user__.get("valves") if __user__ else None
        if not user_valves:
            user_valves = self.UserValves()

        model = "mistral"

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

            print("NO FOLLOW-UPS")
            return {"follow_ups": []}

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

        # use the full chat history for context if available
        dialogue_str = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in body["messages"]
            if m["role"] in ("user", "assistant")
        )
        print(f"Chat history:\n{dialogue_str}\n--- End of chat history ---")

        _id = get_id(dialogue_str)
        print(
            f"Extracted id: {_id} (don't know if it's a task id or a job id at this point)"
        )

        try:
            clients = figure_out_clients(
                dialogue_str,
                model,
                session_id,
                cache="/Users/nilsnilsson/Development/ask-panda/cache",
            )
            selection_client = Selection(clients, model)
            category = selection_client.answer(dialogue_str)

            # --- NEW: if a follow-up would route to log_analyzer, override to document ---
            if is_followup and category != "document":
                # Follow-up detected: overriding category to 'document'
                category = "document"

            client = clients.get(category)
            print(f"Selected client category: {category}")

            # --- OPTIONAL: normalize follow-up prompt for document client by stripping leading ### ---
            doc_prompt = (
                prompt.lstrip("#").strip()
                if (is_followup and category == "document")
                else prompt
            )

            if category == "document":
                print(f"Selected client category: {category} (DocumentQuery)")
                answer = client.ask(doc_prompt)  # use normalized prompt on follow-ups
            elif category == "log_analyzer":
                print(f"Selected client category: {category} (LogAnalysis)")
                question = client.generate_question("pilotlog.txt")
                answer = client.ask(question)
            elif category == "task":
                print(f"Selected client category: {category} (TaskStatus)")
                if not _id:
                    _id = get_id(prompt)
                if not _id:
                    answer = "failed to find the task id in the dialogue"
                    print(answer)
                    return answer

                # reinitialize the client with the correct task id
                clients = figure_out_clients(
                    prompt,
                    model,
                    session_id,
                    cache="/Users/nilsnilsson/Development/ask-panda/cache",
                    task_id=_id,
                )
                client = clients.get(category)
                if not client:
                    answer = f"failed to reinitialize the TaskStatus with task id {_id}"
                    print(answer)
                    return answer

                question = client.generate_question()
                answer = client.ask(question)
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
