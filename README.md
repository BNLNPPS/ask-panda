# Ask PanDA
This project contains MCP powered tools for 1) an AI chatbot for static
conversations and 2) an error analysis agent for PanDA jobs. The goal is to provide a simple and efficient
way to interact with various AI models and analyze errors in PanDA jobs.

For the AI chatbot, it is sufficient to use the agent.py and server.py files. The agent.py file contains the logic
for the agent, while the server.py file contains the logic for the server. The server.py file is a simple FastAPI
server that serves the agent.py file. The agent.py file contains the logic for the agent, which is a simple
command line interface that allows you to interact with various AI models.

Similarly, the error analysis agent is a simple command line interface that allows you to interact with various AI models
and analyze errors in PanDA jobs. The error analysis agent uses the same logic as the agent.py file, but it is
specifically designed for analyzing errors in PanDA jobs.

**Note**: This project is in development. It is currently using HuggingFace embedding models for static conversations.
Note also that even though a "chatbot" is mentioned multiple times, the project is not a chatbot in the traditional sense. Currently,
one can only submit questions in a command line interface and receive answers. The tools can nevertheless be used to build a standard
chatbot eventually.

![Ask PanDA MCP Service.](https://atlas-panda-pilot.web.cern.ch/atlas-panda-pilot/images/Ask_PanDA_diagram.png)

# Features
- **Multi-Model Support**: Interact with various AI models including OpenAI, Anthropic, Gemini, and Llama.
- **Customizable**: Easily add or modify models and their configurations.
- **User-Friendly**: Simple command-line interface for easy interaction.
- **Lightweight**: Minimal dependencies for quick setup and deployment.
- **Open Source**: Built with transparency in mind, allowing for community contributions and improvements.

# Installation
```
pip install -r requirements.txt
```

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

A (random) session ID is used to keep track of the conversation, i.e. to enable context memory. The context memory is stored in an sqlite database.

2. Run the Agent (example queries):
```
python3 -m agents.document_query_agent --question=QUESTION --model=MODEL --session-id=SESSION_ID

Examples:
python3 -m agents.document_query_agent  --question="What is PanDA?" --model=openai --session-id=111
- python3 -m agents.document_query_agent  --question="I do not understand, please explain that better!" --model=openai --session-id=111
python3 -m agents.document_query_agent  --question="How does the PanDA pilot work?" --model=anthropic --session-id=222
python3 -m agents.document_query_agent  --question="What is the purpose of the PanDA server?" --model=llama --session-id=333
python3 -m agents.document_query_agent  --question="What is the PanDA WMS?" --model=gemini --session-id=444 (shows that PanDA WMS is not properly defined)
python3 -m agents.document_query_agent  --question="Please list all of the PanDA pilot error codes" --model=gemini --session-id=555 (demonstration of the limitations of the size of the context window)
```
The agent returns the answer as a dictionary.

# Log Analysis Agent

1. Start the Server as described above.

2. Run the Error Analysis Agent with a custom model:
```
python3 -m agents.log_analysis_agent [-h] --log-files LOG_FILES --pandaid PANDAID --model MODEL --mode MODE
```
**Note**: The error analysis agent will use the provided PanDA ID to fetch one or more log files from
the given PanDA job. The script will then extract the error codes from the log files, along with relevant/nearby log message
and build a context for the model. The script will then use the provided model to analyze the reason for the error.

**Note**: Due to the limited context window of the models, the agent will
not be able to answer all of the questions. The last example can be used as a benchmark test.
The agent will return a list of the error codes, but it will not be able to provide the full context of each error code.
This is a limitation of the current implementation and is not a reflection of the capabilities of the models.

**Note**: For now, use --log-files pilotlog.txt, --model openai or gemini, and --mode contextual. E.g. analyze a job that failed with pilot error code 1150, "Looping job killed by pilot":

```
python3 -m agents.log_analysis_agent --pandaid 6681623402 --log-files pilotlog.txt --model gemini --mode contextual
```

The following pilot error codes have been verified to work with the error analysis agent:
```
1099, 1104, 1137, 1150, 1152, 1201, 1213, 1235, 1236, 1305, 1322, 1324, 1354, 1361, 1368.
```

# Smart Reporting Agent

The smart reporting agent is a tool that can be used to generate reports based on the PanDA job records. Currently,
it downloads JSON files from the PanDA Monitor that contains the error code information for jobs that ran in the last 24, 12, 6, 3 and 1 hours.
The corresponding JSON files will be refreshed automatically, corresponding to the time period of the job records.

Note: This is work in progress. The idea is that these reports can be used by other agents.

The agent is run as follows:
```
python3 -m agents.smart_reporting_agent --pid PID --cache-dir CACHE_DIR
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
python3 -m agents.selection_agent --question QUESTION --model MODEL [--session-id SESSION_ID] [--mode MODE]
```
where `QUESTION` is the question to be answered, `MODEL` is the model to be used (e.g. openai, anthropic, gemini, llama), and
`SESSION_ID` and `MODE` are optional arguments. For details, see same arguments defined above.

When run, the agent determines which agent to use for the given question, calls that agent and returns the answer.

NOTE: Currently, only the `document_query_agent` and `log_analysis_agent` are supported, but more agents can be added in the future.

[As of now, the agent only gives the name of the agent to use, but does not call it automatically. This is a work in progress.]

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
git merge upstream/next
```
