# Ask PanDA
This project contains MCP powered tools for 1) an AI chatbot for static (and soon, possibly, dynamic)
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

# AI Chatbot
0. Create the vector store for the static conversation:
```
python create_vectorstore.py
```
This will create a vector store for the static conversation. The vector store is a simple SQLite database that 
contains the embeddings for the static conversation. The vector store is used to retrieve the most relevant 
context for the given question.

This step is only needed once. The vector store will be created in the same directory as the script.

1. Start the Server:
```
uvicorn server:app --reload
```
2. Run the Agent (example queries):
```
python agent.py "What is PanDA?" openai
python agent.py "How does the PanDA pilot work?" anthropic
python agent.py "What is the purpose of the PanDA server?" llama
python agent.py "What is the PanDA WMS?" gemini  (shows that PanDA WMS is not properly defined)
python agent.py "Please list all of the PanDA pilot error codes" gemini  (demonstration of the limitations of the size of the context window)
```

# Error Analysis Agent

1. Start the Server as described above.

2. Run the Error Analysis Agent with a custom model:
```
python error_analyzer_agent.py [-h] --log-files LOG_FILES --pandaid PANDAID --model MODEL
```
**Note**: The error analysis agent will use the provided PanDA ID to fetch one or more log files from
the given PanDA job. The script will then extract the error codes from the log files, along with relevant/nearby log message
and build a context for the model. The script will then use the provided model to analyze the reason for the error. 

IN DEVELOPMENT: The error analysis agent is not yet fully functional. It will return a list of the error codes, but it will not be able to provide the full context of each error code. This is a limitation of the current implementation and is not a reflection of the capabilities of the models.

**Note**: Due to the limited context window of the models, the agent will 
not be able to answer all of the questions. The last example can be used as a benchmark test.
The agent will return a list of the error codes, but it will not be able to provide the full context of each error code. 
This is a limitation of the current implementation and is not a reflection of the capabilities of the models.


