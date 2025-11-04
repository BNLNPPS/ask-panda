# Ask PanDA
This project contains MCP powered tools for 1) an AI chatbot agent for static
conversations, 2) an error analysis agent for PanDA jobs and 3) a data query agent for queries related to JSON data. There is a common interface for all agents, which allows
you to use the same command line interface for all agents. The agents can be used to interact with various AI models, including OpenAI, Anthropic, Gemini and Llama..

The goal is to provide a simple and efficient way to interact with various AI models and analyze PanDA documentation, jobs, tasks, JSON and related errors.

The MCP server is a simple FastAPI server that serves the agents. The agents contain the logic for different purposes (described below), and are simple
command line interfaces that allow for interaction with various AI models.

**Note**: This project is in development. It is currently using HuggingFace embedding models for static conversations.
Note also that even though a "chatbot" is mentioned multiple times, the project is not a chatbot in the traditional sense. Currently,
one can only submit questions in a command line interface and receive answers. The tools can nevertheless be used to build a standard
chatbot eventually.

![Ask PanDA MCP Service.](https://cern.ch/atlas-panda-pilot/images/Ask_PanDA_diagram2.png)

# Features
- **Multi-Model Support**: Interact with various AI models including OpenAI, Anthropic, Gemini, and Llama.
- **Customizable**: Easily add or modify models and their configurations.
- **User-Friendly**: Simple command-line interface for easy interaction.
- **Lightweight**: Minimal dependencies for quick setup and deployment.
- **Open Source**: Built with transparency in mind, allowing for community contributions and improvements.

# Installation

## Local Installation
```
pip install -r requirements.txt
```

## Docker Installation

You can run Ask-PanDA and the Ollama shim in Docker containers with Docker Compose:

1. Copy the environment template and configure your API keys:
```bash
cp .env.example .env
# Edit .env and add at least MISTRAL_API_KEY (or other LLM API keys)
```

2. Build and start the services:
```bash
docker compose up -d
```

This will start two services:
- **ask-panda**: Main server on port 8001 (external) → 8000 (internal)
  - Uses `open-mistral-7b` model by default
  - Includes RAG with vector store for PanDA documentation
- **ollama-shim**: Ollama-compatible API on port 11435 (external) → 11434 (internal)
  - Exposes `mistral-proxy:latest` model
  - Forwards requests to Ask-PanDA's RAG endpoint
  - Avoids conflict with localhost Ollama on port 11434

3. Check service status:
```bash
docker compose ps
docker compose logs -f
```

4. Test the setup:
```bash
./test-docker-setup.sh
```

5. Stop the services:
```bash
docker compose down
```

**Network Configuration:**
- Services run in an isolated Docker network (`ask-panda-network`)
- Internal communication uses service names (e.g., `http://ask-panda:8000`)
- External access via mapped ports (8001 and 11435)

**Port Mapping Summary:**
- `8000` - Local Ask-PanDA server (non-Docker, if running)
- `8001` - Ask-PanDA Docker container
- `11434` - Local Ollama server (if running)
- `11435` - Ollama shim Docker container

# Environment Variables
Ensure that at least one of these keys are set in your environment for secure API access, and select the
model accordingly (see below):
```
export ANTHROPIC_API_KEY='your_anthropic_api_key'
export OPENAI_API_KEY='your_openai_api_key'
export GEMINI_API_KEY='your_gemini_api_key'
export LLAMA_API_URL='http://localhost:11434/api/generate'  # For Ollama Llama3 model
```

# MCP server and Document Query Agent

1. Start the Server:
```
uvicorn ask_panda_server:app --reload &
```
When the server is started, it will create a vector store (Chroma DB) for the static conversation based on documents in the
resources directory. The server will monitor the resources directory for changes (once per minute) and will update the vector store when necessary.
New documents can be added to the resources directory, and the server will automatically update the vector store.

**Note**: when the vector store is being updated, it is currently not available, i.e. any queries will be delayed until the update is complete.
The update and usage is using thread locking, so the server will not crash if a query is made while the vector store is being updated.

The server will write all log messages to the `ask_panda_server_log.txt` file in the current directory.

An optional session ID is used to keep track of the conversation, i.e. to enable context memory. The context memory is stored in an sqlite database.
If the session ID is not provided, a new session will be created for each query.

2. Run the Agent (example queries):
```
python3 -m agents.document_query_agent --question=QUESTION --model=MODEL [--session-id=SESSION_ID]

Examples:
python3 -m agents.document_query_agent  --question="What is PanDA?" --model=openai --session-id=111
- python3 -m agents.document_query_agent  --question="I do not understand, please explain that better!" --model=openai --session-id=111
python3 -m agents.document_query_agent  --question="How does the PanDA pilot work?" --model=anthropic --session-id=222
python3 -m agents.document_query_agent  --question="What is the purpose of the PanDA server?" --model=llama --session-id=333
python3 -m agents.document_query_agent  --question="What is the PanDA WMS?" --model=gemini --session-id=444 (shows that PanDA WMS is not properly defined)
python3 -m agents.document_query_agent  --question="Please list all of the PanDA pilot error codes" --model=gemini --session-id=555 (demonstration of the limitations of the size of the context window)
```

The agent can also be imported and used in Python code as follows:
```python
from agents.document_query_agent import DocumentQueryAgent
agent = DocumentQueryAgent(model='openai', session_id='12345')
answer = agent.query("What is PanDA?")
print(answer)
answer = agent.query("Please give more details")
print(answer)
```

The agent returns the answer as a dictionary.

# Log Analysis Agent

1. Start the Server as described above.

2. Run the Error Analysis Agent with a custom model:
```
python3 -m agents.log_analysis_agent [-h] --pandaid PANDAID --model MODEL [-session-id=SESSION_ID] [--log-file LOG_FILE]
```

By providing a session ID, the conversation will be tracked and the Document Query Agent will be able to answer follow-up questions
(if the same session ID is used).

**Note**: The error analysis agent will use the provided PanDA ID to fetch a log files from
the given PanDA job. The script will then extract the error codes from the log file, along with relevant/nearby log message
and build a context for the model. The script will then use the provided model to analyze the reason for the error.

**Note**: Due to the limited context window of the models, the agent will
not be able to answer all of the questions. The last example can be used as a benchmark test.
The agent will return a list of the error codes, but it will not be able to provide the full context of each error code.
This is a limitation of the current implementation and is not a reflection of the capabilities of the models.

```
python3 -m agents.log_analysis_agent --pandaid 6681623402 --model gemini --log-file pilotlog.txt
```

The following pilot error codes have been verified to work with the error analysis agent:
```
1099, 1104, 1137, 1150, 1152, 1201, 1213, 1235, 1236, 1305, 1322, 1324, 1354, 1361, 1368.
```

#   Data Query Agent

The data query agent is a tool that can be used to query JSON files e.g. for the status of a PanDA task
or data contained in a JSON file (e.g. site/queue data). It uses the PanDA Monitor API to
retrieve relevant JSON file and returns it as a python dictionary that can be analyzed by the LLM.
The agent can be run as follows:
```
python3 -m agents.data_query_agent --taskid TASKID --model MODEL [--session-id SESSION_ID]  (for task related queries)
python3 -m agents.data_query_agent --queue QUEUENAME --model MODEL [--session-id SESSION_ID] (for site/queue related queries)
```
where `TASKID` is the PanDA ID of the task to be queried, `QUEUENAME` is the name of the queue to be queried,
`MODEL` is the model to be used (e.g. openai, anthropic, gemini, llama),
and `SESSION_ID` is the unique identifier for the conversation (i.e. continue using the same id if the conversation is ongoing).

**NOTE** This agent is early in development and is not yet fully functional, but it already gives a summary of the task status
with a breakdown on the number of jobs in each state (e.g. running, waiting, failed, etc.) as well as a summary of any failures.
It can be used to get a quick overview of the task status.

# Maintenance Agent

The maintenance agent is a tool that can be used to run maintenance on the cache dir; including cleaning up outdated
files, but also to generate reports based on the PanDA job records. Currently,
it downloads JSON files from the PanDA Monitor that contains the error code information for jobs that ran in the last 24, 12, 6, 3 and 1 hours.
The corresponding JSON files will be refreshed automatically, corresponding to the time period of the job records.

Note: This is work in progress. The idea is that these reports can be used by other agents.

The agent is run as follows:
```
python3 -m agents.maintenance_agent --pid PID --cache-dir CACHE_DIR
```
where `PID` is the process id of the MCP server and `CACHE_DIR` is the directory where the JSON files will be stored.
The agent will abort when it sees that the MCP server is no longer running. The cache directory will be created if it does not exist
(e.g. set it to 'cache').

Eventually, the agent will be launched automatically by the MCP server, but for now it needs to be run manually.

# Selection Agent

The selection agent is a tool that can be used to determine which agent to use for a given question. It is arguably better
to implement multiple agents that are specialized for different tasks, rather than a single agent that tries to do everything.
Also, a complication is to determine the correct prompt for different types of questions, e.g. whether the question is about PanDA jobs, number of
running jobs, or general questions about PanDA, etc. One would be forced to implement complicated logic and parsing of the
user's question. Instead, this agent uses an LLM to determine which agent to use for a given question, and therefore acts
as a middle layer between the user, the other agents and the LLM.

The selection agent is run as follows:
```
python3 -m agents.selection_agent --question QUESTION --model MODEL [--session-id SESSION_ID]
```
where `QUESTION` is the question to be answered, `MODEL` is the model to be used (e.g. openai, anthropic, gemini, llama), and
`SESSION_ID` is the unique identier for the conversation (i.e. continue using the same id if the conversation is ongoing).

When run, the agent determines which agent to use for the given question, calls that agent and returns the answer.

**NOTE**: Currently, only the Document Query, Log Analysis and Data Query Agents are supported, but more agents
can be added in the future.
The maintenance agent is not relevant for the selection agent as it is supposed to be run from the server.

# Open Web UI

A "pipe function" for Open Web UI is provided in the open-webui directory. It can be used to run the Ask PanDA agents from within
the Open Web UI. The pipe function is a simple wrapper around the agents that allows them to be used in the Open Web UI.

It can be installed in the Open Web UI by copying the source code (cut-and-paste) into the Open Web UI directly.

Explicit instructions:

* Click on the user name on the bottom left of the UI and select "Admin Panel"
* Select the "Functions" tab and click on the "+" sign on the top right to create a new function
* Select "New function" and paste the Ask PanDA code into the code editor
* Save the function and it will be available in the Open Web UI from the models drop-down menu.

If you gave the function the name "Ask PanDA", it will be available as a model called "Ask PanDA". Before the
Open Web UI is run, you need to set the `GEMINI_API_KEY` environment variable to your Gemini API key as well as
the `PYTHONPATH` environment variable to include the path to the Ask PanDA source code.

*Notes*:
* Currently "gemini" is hardcoded in the default source code, but this can be changed to any other model by editing the source code.
It does not yet support selecting models from the Open Web UI, but this can be added in the future.
* The pipe function does not set the session ID properly (I have not figure out how to get a fixed session ID for
a given chat yet) and therefore the context memory is not preserved between queries on the Ask PanDA side. It might not
be needed since Open Web UI has its own context memory, but it is something to be aware of.
* The Ask PanDA MCP server should be run in its own virtual environment as the Open Web UI are currently using
very old versions of the libraries and the Ask PanDA MCP server requires more recent versions of the libraries.

![Open Web UI](https://atlas-panda-pilot.web.cern.ch/atlas-panda-pilot/images/OpenWebUI.png)

The Open Web UI web server (installed separately) can be started as follows:
```
open-webui serve
```

(Let it run in it's own virtual environment - it wil produce a lot of output).

## Ollama Shim for Open WebUI

### Docker Deployment (Recommended)

The Ollama shim is included in the Docker Compose setup and runs on port 11435 (to avoid conflicts with local Ollama on 11434).

**Configuration:**
- Port 11435 (external) → 11434 (internal container)
- Uses `open-mistral-7b` model via Mistral API
- Automatically connects to Ask-PanDA service via Docker network

**Setup for Open WebUI:**
1. Make sure Docker services are running:
   ```bash
   docker compose up -d
   ```

2. Configure Open WebUI to use the shim:
   - Go to Admin Panel → Settings → Connections
   - For Docker Open WebUI: Set `OLLAMA_BASE_URL=http://host.docker.internal:11435`
   - For local Open WebUI: Set `OLLAMA_BASE_URL=http://localhost:11435`

3. The model `mistral-proxy:latest` will appear in the model picker

**Model Configuration:**
- The shim currently uses the `/rag_ask` endpoint for better performance
- Uses Mistral's `open-mistral-7b` model (configured in `ask_panda_server.py`)
- Responses include RAG-enhanced context from your documentation

### Standalone Deployment

If you want to run the shim without Docker:

```bash
python3 ollama_shim.py
```

The shim listens on port `11434` by default and forwards the chat/generate
requests to the Ask PanDA HTTP API.

**Environment variables:**
- `ASK_PANDA_BASE_URL` (default `http://localhost:8000`) – base URL for the Ask PanDA server
- `OLLAMA_SHIM_MODEL` (default `mistral`) – backend model name
- `OLLAMA_SHIM_MODEL_DISPLAY` (default `mistral-proxy`) – model name shown in Open WebUI
- `OLLAMA_SHIM_PORT` (default `11434`) – port to listen on

**Note:** When running standalone, ensure your MISTRAL_API_KEY is set in the environment.

# Vector store

Note that the vector store (Chroma DB) is created and maintained by a manager. The contents of the vector store are stored in the `vectorstore` directory
and can be inspected using the `vectorstore_manager.py` script.

```
python3 -m tools.inspect_vectorstore --dump
```

If the --dump option is used, the script will dump the contents of the vector store in raw form (to stdout). If used without this option,
the script will print the contents of the vector store in a human-readable form (also to stdout).

# Syncing your GitHub repository

Before making a pull request, make sure that you are synced to the latest version.

```
git remote -v
git remote add upstream  https://github.com/BNLNPPS/ask-panda.git
git fetch upstream
git merge upstream/master
```
