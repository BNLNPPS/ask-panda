# ask-panda
An MCP powered AI chatbot for static (and soon dynamic) conversations.

**Note**: This project is in development. It is currently using HuggingFace embedding models for static conversations.

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
Ensure that at least one of these keys are set in your environment for secure API access:
```
export ANTHROPIC_API_KEY='your_anthropic_api_key'
export OPENAI_API_KEY='your_openai_api_key'
export GEMINI_API_KEY='your_gemini_api_key'
export LLAMA_API_URL='http://localhost:11434/api/generate'  # For Ollama Llama3 model
```

# Usage
0. Create the vector store for the static conversation:
```
python create_vectorstore.py
```
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
3. Run the Error Analysis Agent with a custom model:
```
python error_analysis_agent.py <PanDA ID: int> gemini
```
Note: The error analysis agent will use the provided PanDA ID to fetch the pilotlog.txt for
the given PanDA job. It will then use the provided model to analyze the reason for the error. Use an existing PanDA ID for a failed job.
IN DEVELOPMENT: The error analysis agent is not yet fully functional. It will return a list of the error codes, but it will not be able to provide the full context of each error code. This is a limitation of the current implementation and is not a reflection of the capabilities of the models.

**Note**: Due to the limited context window of the models, the agent will 
not be able to answer all of the questions. The last example can be used as a benchmark test.
The agent will return a list of the error codes, but it will not be able to provide the full context of each error code. 
This is a limitation of the current implementation and is not a reflection of the capabilities of the models.


