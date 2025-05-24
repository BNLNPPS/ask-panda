import pytest
import os
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

# Import necessary classes and constants from server.py
from server import PandaMCP

# Import specific exceptions for mocking
import anthropic
import openai
import httpx
import google.generativeai as genai

# Mock the vectorstore before it's accessed by PandaMCP instance or rag_query
# Patching FAISS.load_local might be too early if server.py is imported directly.
# Instead, we'll patch vectorstore.similarity_search within each test or fixture
# where rag_query is called.

DUMMY_SIMILARITY_SEARCH_RESULT = [MagicMock(page_content="dummy content 1"), MagicMock(page_content="dummy content 2")]

@pytest.fixture
def mcp_instance(mocker):
    """Fixture to create a PandaMCP instance with mocked API keys."""
    # Mock os.getenv used for API keys at the module level in server.py
    # This needs to be active when server.py is imported or when PandaMCP accesses them.
    # The ideal way is to patch 'server.ANTHROPIC_API_KEY', 'server.OPENAI_API_KEY', etc.
    mocker.patch('server.ANTHROPIC_API_KEY', "fake_anthropic_key")
    mocker.patch('server.OPENAI_API_KEY', "fake_openai_key")
    mocker.patch('server.GEMINI_API_KEY', "fake_gemini_key")
    mocker.patch('server.LLAMA_API_URL', "http://fake-llama-api:11434/api/generate")
    
    instance = PandaMCP("panda_test")
    return instance

# Helper for patching vectorstore.similarity_search
def patch_vectorstore_search(mocker, return_value=DUMMY_SIMILARITY_SEARCH_RESULT):
    return mocker.patch('server.vectorstore.similarity_search', return_value=return_value)

# --- Test Cases for API Key Checks (targeting helper methods) ---

@pytest.mark.asyncio
async def test_call_anthropic_missing_key(mcp_instance, mocker):
    mocker.patch('server.ANTHROPIC_API_KEY', None)
    with pytest.raises(ValueError, match="Anthropic API key is not set"):
        await mcp_instance._call_anthropic("test prompt")

@pytest.mark.asyncio
async def test_call_openai_missing_key(mcp_instance, mocker):
    mocker.patch('server.OPENAI_API_KEY', None)
    with pytest.raises(ValueError, match="OpenAI API key is not set"):
        await mcp_instance._call_openai("test prompt")

@pytest.mark.asyncio
async def test_call_gemini_missing_key(mcp_instance, mocker):
    mocker.patch('server.GEMINI_API_KEY', None)
    with pytest.raises(ValueError, match="Gemini API key is not set"):
        await mcp_instance._call_gemini("test prompt")

# Test that it proceeds if key IS set (implicitly tested by API error tests for helper methods)
@pytest.mark.asyncio
async def test_call_anthropic_key_present(mcp_instance, mocker):
    # ANTHROPIC_API_KEY is mocked to "fake_anthropic_key" in mcp_instance fixture
    # We expect an error from the API call itself (mocked here), not a ValueError for the key.
    # Patching 'server.AsyncAnthropic' as it's directly used in _call_anthropic
    with patch('server.AsyncAnthropic', new_callable=AsyncMock) as mock_anthropic_client_class:
        mock_client_instance = mock_anthropic_client_class.return_value
        mock_client_instance.messages.create = AsyncMock(side_effect=anthropic.APIError("Simulated API Error"))
        
        response = await mcp_instance._call_anthropic("test prompt")
        assert "Error interacting with Anthropic API" in response

# --- Test Cases for API Call Error Handling (targeting helper methods) ---

# Anthropic
@pytest.mark.asyncio
async def test_call_anthropic_api_error(mcp_instance, mocker):
    with patch('server.AsyncAnthropic', new_callable=AsyncMock) as mock_anthropic_client_class:
        mock_client_instance = mock_anthropic_client_class.return_value
        mock_client_instance.messages.create = AsyncMock(side_effect=anthropic.APIError("Test Anthropic API Error"))
        
        response = await mcp_instance._call_anthropic("test prompt")
        assert response == "Error interacting with Anthropic API: Test Anthropic API Error"

@pytest.mark.asyncio
async def test_call_anthropic_connection_error(mcp_instance, mocker):
    with patch('server.AsyncAnthropic', new_callable=AsyncMock) as mock_anthropic_client_class:
        mock_client_instance = mock_anthropic_client_class.return_value
        mock_client_instance.messages.create = AsyncMock(side_effect=anthropic.APIConnectionError("Test Connection Error"))
        
        response = await mcp_instance._call_anthropic("test prompt")
        assert response == "Error interacting with Anthropic API: Connection error - Test Connection Error"

# OpenAI
@pytest.mark.asyncio
async def test_call_openai_api_error(mcp_instance, mocker):
    api_error = openai.APIError("Test OpenAI API Error", request=None, body=None)
    with patch('server.AsyncOpenAI', new_callable=AsyncMock) as mock_openai_client_class:
        mock_client_instance = mock_openai_client_class.return_value
        mock_client_instance.chat.completions.create = AsyncMock(side_effect=api_error)
        
        response = await mcp_instance._call_openai("test prompt")
        assert response == f"Error interacting with OpenAI API: {api_error}"

@pytest.mark.asyncio
async def test_call_openai_authentication_error(mcp_instance, mocker):
    auth_error = openai.AuthenticationError("Test Auth Error", response=MagicMock(), body=None)
    with patch('server.AsyncOpenAI', new_callable=AsyncMock) as mock_openai_client_class:
        mock_client_instance = mock_openai_client_class.return_value
        mock_client_instance.chat.completions.create = AsyncMock(side_effect=auth_error)
        
        response = await mcp_instance._call_openai("test prompt")
        assert response == f"Error interacting with OpenAI API: Authentication failed - {auth_error}"

# LLaMA (httpx)
@pytest.mark.asyncio
async def test_call_llama_http_status_error(mcp_instance, mocker):
    mock_response = httpx.Response(500, content=b"Server Error", request=MagicMock())
    http_error = httpx.HTTPStatusError("Server Error", request=MagicMock(), response=mock_response)
    
    with patch('server.httpx.AsyncClient', new_callable=AsyncMock) as mock_http_client_class: # Note: server.httpx.AsyncClient
        mock_client_instance = mock_http_client_class.return_value.__aenter__.return_value
        mock_client_instance.post = AsyncMock(side_effect=http_error)
        
        response = await mcp_instance._call_llama("test prompt")
        assert f"Error interacting with LLaMA API: HTTP error ({mock_response.status_code}) - {mock_response.text}" in response

@pytest.mark.asyncio
async def test_call_llama_request_error(mcp_instance, mocker):
    req_error = httpx.RequestError("Test Request Error", request=MagicMock())
    with patch('server.httpx.AsyncClient', new_callable=AsyncMock) as mock_http_client_class:
        mock_client_instance = mock_http_client_class.return_value.__aenter__.return_value
        mock_client_instance.post = AsyncMock(side_effect=req_error)
        
        response = await mcp_instance._call_llama("test prompt")
        assert response == f"Error interacting with LLaMA API: Request error - {req_error}"

# Gemini
@pytest.mark.asyncio
async def test_call_gemini_google_api_error(mcp_instance, mocker):
    api_error = genai.core.exceptions.GoogleAPIError("Test Gemini GoogleAPIError")
    # Patching 'server.genai.GenerativeModel' as it's used like 'genai.GenerativeModel(...)' in _call_gemini
    with patch('server.genai.GenerativeModel', new_callable=AsyncMock) as mock_gemini_model_class:
        mock_model_instance = mock_gemini_model_class.return_value
        mock_model_instance.generate_content_async = AsyncMock(side_effect=api_error)
        
        response = await mcp_instance._call_gemini("test prompt")
        assert response == f"Error interacting with Gemini API: Google API error - {api_error}"

@pytest.mark.asyncio
async def test_call_gemini_blocked_prompt_error(mcp_instance, mocker):
    blocked_error = genai.types.BlockedPromptException("Test Blocked Prompt")
    with patch('server.genai.GenerativeModel', new_callable=AsyncMock) as mock_gemini_model_class:
        mock_model_instance = mock_gemini_model_class.return_value
        mock_model_instance.generate_content_async = AsyncMock(side_effect=blocked_error)
        
        response = await mcp_instance._call_gemini("test prompt")
        assert response == f"Error interacting with Gemini API: Prompt was blocked - {blocked_error}"

# Generic Exception Test (using Anthropic helper as an example)
@pytest.mark.asyncio
async def test_call_anthropic_generic_exception(mcp_instance, mocker):
    generic_exception = Exception("A wild generic error appears!")
    with patch('server.AsyncAnthropic', new_callable=AsyncMock) as mock_anthropic_client_class:
        mock_client_instance = mock_anthropic_client_class.return_value
        mock_client_instance.messages.create = AsyncMock(side_effect=generic_exception)
        
        response = await mcp_instance._call_anthropic("test prompt")
        assert response == f"An unexpected error occurred with Anthropic API: {generic_exception}"

# --- Test Cases for Successful API Call (targeting helper methods) ---
@pytest.mark.asyncio
async def test_call_openai_successful(mcp_instance, mocker):
    mock_choice = MagicMock()
    mock_choice.message.content = "  Mocked OpenAI Response  "
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    
    with patch('server.AsyncOpenAI', new_callable=AsyncMock) as mock_openai_client_class:
        mock_client_instance = mock_openai_client_class.return_value
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_completion)
        
        response = await mcp_instance._call_openai("test prompt")
        assert response == "Mocked OpenAI Response"

# --- Tests for rag_query Dispatching ---
DUMMY_PROMPT = "Generated prompt for testing"

@pytest.mark.asyncio
async def test_rag_query_dispatches_to_anthropic(mcp_instance, mocker):
    patch_vectorstore_search(mocker) # rag_query uses this
    # Mock the helper method _call_anthropic on the instance
    mock_helper = mocker.patch.object(mcp_instance, '_call_anthropic', new_callable=AsyncMock)
    mock_helper.return_value = "anthropic_response"
    
    # The actual prompt generation in rag_query needs to be considered for the assert
    docs = DUMMY_SIMILARITY_SEARCH_RESULT
    context = "\n\n".join(doc.page_content for doc in docs)
    expected_prompt = f"Answer based on the following context:\n{context}\n\nQuestion: test question"

    response = await mcp_instance.rag_query("test question", "anthropic")
    mock_helper.assert_called_once_with(expected_prompt)
    assert response == "anthropic_response"

@pytest.mark.asyncio
async def test_rag_query_dispatches_to_openai(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    mock_helper = mocker.patch.object(mcp_instance, '_call_openai', new_callable=AsyncMock)
    mock_helper.return_value = "openai_response"
    
    docs = DUMMY_SIMILARITY_SEARCH_RESULT
    context = "\n\n".join(doc.page_content for doc in docs)
    expected_prompt = f"Answer based on the following context:\n{context}\n\nQuestion: test question"

    response = await mcp_instance.rag_query("test question", "openai")
    mock_helper.assert_called_once_with(expected_prompt)
    assert response == "openai_response"

@pytest.mark.asyncio
async def test_rag_query_dispatches_to_llama(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    mock_helper = mocker.patch.object(mcp_instance, '_call_llama', new_callable=AsyncMock)
    mock_helper.return_value = "llama_response"

    docs = DUMMY_SIMILARITY_SEARCH_RESULT
    context = "\n\n".join(doc.page_content for doc in docs)
    expected_prompt = f"Answer based on the following context:\n{context}\n\nQuestion: test question"
    
    response = await mcp_instance.rag_query("test question", "llama")
    mock_helper.assert_called_once_with(expected_prompt)
    assert response == "llama_response"

@pytest.mark.asyncio
async def test_rag_query_dispatches_to_gemini(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    mock_helper = mocker.patch.object(mcp_instance, '_call_gemini', new_callable=AsyncMock)
    mock_helper.return_value = "gemini_response"

    docs = DUMMY_SIMILARITY_SEARCH_RESULT
    context = "\n\n".join(doc.page_content for doc in docs)
    expected_prompt = f"Answer based on the following context:\n{context}\n\nQuestion: test question"

    response = await mcp_instance.rag_query("test question", "gemini")
    mock_helper.assert_called_once_with(expected_prompt)
    assert response == "gemini_response"

@pytest.mark.asyncio
async def test_rag_query_unsupported_model(mcp_instance, mocker):
    patch_vectorstore_search(mocker) # rag_query uses this before raising error for model
    with pytest.raises(ValueError, match="Unsupported model 'kryptonite'."):
        await mcp_instance.rag_query("test question", "kryptonite")

# --- Test LLaMA JSONDecodeError (targeting helper) ---
@pytest.mark.asyncio
async def test_call_llama_json_decode_error(mcp_instance, mocker):
    mock_llama_response = AsyncMock()
    mock_llama_response.raise_for_status = MagicMock() 
    decode_error = ValueError("Simulated JSONDecodeError") 
    mock_llama_response.json = MagicMock(side_effect=decode_error)

    with patch('server.httpx.AsyncClient', new_callable=AsyncMock) as mock_http_client_class:
        mock_client_instance = mock_http_client_class.return_value.__aenter__.return_value
        mock_client_instance.post = AsyncMock(return_value=mock_llama_response)
        
        response = await mcp_instance._call_llama("test prompt")
        assert f"An unexpected error occurred with LLaMA API: {decode_error}" in response
print("test_server.py content updated for refactoring.")
