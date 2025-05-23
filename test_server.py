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

# --- Test Cases for API Key Checks ---

@pytest.mark.asyncio
async def test_rag_query_anthropic_missing_key(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    mocker.patch('server.ANTHROPIC_API_KEY', None) # Mock the module-level variable
    with pytest.raises(ValueError, match="Anthropic API key is not set"):
        await mcp_instance.rag_query("test question", "anthropic")

@pytest.mark.asyncio
async def test_rag_query_openai_missing_key(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    mocker.patch('server.OPENAI_API_KEY', None)
    with pytest.raises(ValueError, match="OpenAI API key is not set"):
        await mcp_instance.rag_query("test question", "openai")

@pytest.mark.asyncio
async def test_rag_query_gemini_missing_key(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    mocker.patch('server.GEMINI_API_KEY', None)
    with pytest.raises(ValueError, match="Gemini API key is not set"):
        await mcp_instance.rag_query("test question", "gemini")

# Test that it proceeds if key IS set (implicitly tested by API error tests, but can be explicit)
@pytest.mark.asyncio
async def test_rag_query_anthropic_key_present(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    # ANTHROPIC_API_KEY is already mocked to "fake_anthropic_key" in mcp_instance fixture
    # We expect an error from the API call itself, not a ValueError for the key.
    with patch('anthropic.AsyncAnthropic', new_callable=AsyncMock) as mock_anthropic_client:
        mock_client_instance = mock_anthropic_client.return_value
        mock_client_instance.messages.create = AsyncMock(side_effect=anthropic.APIError("Simulated API Error"))
        
        response = await mcp_instance.rag_query("test question", "anthropic")
        assert "Error interacting with Anthropic API" in response # Confirms it passed key check

# --- Test Cases for API Call Error Handling ---

# Anthropic
@pytest.mark.asyncio
async def test_rag_query_anthropic_api_error(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    with patch('anthropic.AsyncAnthropic', new_callable=AsyncMock) as mock_anthropic_client:
        mock_client_instance = mock_anthropic_client.return_value
        mock_client_instance.messages.create = AsyncMock(side_effect=anthropic.APIError("Test Anthropic API Error"))
        
        response = await mcp_instance.rag_query("test question", "anthropic")
        assert response == "Error interacting with Anthropic API: Test Anthropic API Error"

@pytest.mark.asyncio
async def test_rag_query_anthropic_connection_error(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    with patch('anthropic.AsyncAnthropic', new_callable=AsyncMock) as mock_anthropic_client:
        mock_client_instance = mock_anthropic_client.return_value
        mock_client_instance.messages.create = AsyncMock(side_effect=anthropic.APIConnectionError("Test Connection Error"))
        
        response = await mcp_instance.rag_query("test question", "anthropic")
        assert response == "Error interacting with Anthropic API: Connection error - Test Connection Error"

# OpenAI
@pytest.mark.asyncio
async def test_rag_query_openai_api_error(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    # For openai.APIError, the constructor takes at least one argument, e.g., a message.
    # Depending on the version and specific error type, it might need `request` or `body`.
    # For a generic APIError, a message should suffice.
    api_error = openai.APIError("Test OpenAI API Error", request=None, body=None)
    with patch('openai.AsyncOpenAI', new_callable=AsyncMock) as mock_openai_client:
        mock_client_instance = mock_openai_client.return_value
        mock_client_instance.chat.completions.create = AsyncMock(side_effect=api_error)
        
        response = await mcp_instance.rag_query("test question", "openai")
        assert response == f"Error interacting with OpenAI API: {api_error}"


@pytest.mark.asyncio
async def test_rag_query_openai_authentication_error(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    auth_error = openai.AuthenticationError("Test Auth Error", response=MagicMock(), body=None)
    with patch('openai.AsyncOpenAI', new_callable=AsyncMock) as mock_openai_client:
        mock_client_instance = mock_openai_client.return_value
        mock_client_instance.chat.completions.create = AsyncMock(side_effect=auth_error)
        
        response = await mcp_instance.rag_query("test question", "openai")
        assert response == f"Error interacting with OpenAI API: Authentication failed - {auth_error}"

# LLaMA (httpx)
@pytest.mark.asyncio
async def test_rag_query_llama_http_status_error(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    # Create a mock response for HTTPStatusError
    mock_response = httpx.Response(500, content=b"Server Error", request=MagicMock())
    http_error = httpx.HTTPStatusError("Server Error", request=MagicMock(), response=mock_response)
    
    with patch('httpx.AsyncClient', new_callable=AsyncMock) as mock_http_client:
        mock_client_instance = mock_http_client.return_value.__aenter__.return_value # Handle async context manager
        mock_client_instance.post = AsyncMock(side_effect=http_error)
        
        response = await mcp_instance.rag_query("test question", "llama")
        assert f"Error interacting with LLaMA API: HTTP error ({mock_response.status_code}) - {mock_response.text}" in response

@pytest.mark.asyncio
async def test_rag_query_llama_request_error(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    req_error = httpx.RequestError("Test Request Error", request=MagicMock())
    with patch('httpx.AsyncClient', new_callable=AsyncMock) as mock_http_client:
        mock_client_instance = mock_http_client.return_value.__aenter__.return_value
        mock_client_instance.post = AsyncMock(side_effect=req_error)
        
        response = await mcp_instance.rag_query("test question", "llama")
        assert response == f"Error interacting with LLaMA API: Request error - {req_error}"

# Gemini
@pytest.mark.asyncio
async def test_rag_query_gemini_google_api_error(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    # google.api_core.exceptions.GoogleAPIError is a broad class.
    # A subclass like InternalServerError or ServiceUnavailable might be more specific if needed.
    # For now, the base class with a message.
    api_error = genai.core.exceptions.GoogleAPIError("Test Gemini GoogleAPIError")
    with patch('google.generativeai.GenerativeModel', new_callable=AsyncMock) as mock_gemini_model_class:
        mock_model_instance = mock_gemini_model_class.return_value
        mock_model_instance.generate_content_async = AsyncMock(side_effect=api_error)
        
        response = await mcp_instance.rag_query("test question", "gemini")
        assert response == f"Error interacting with Gemini API: Google API error - {api_error}"

@pytest.mark.asyncio
async def test_rag_query_gemini_blocked_prompt_error(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    blocked_error = genai.types.BlockedPromptException("Test Blocked Prompt")
    with patch('google.generativeai.GenerativeModel', new_callable=AsyncMock) as mock_gemini_model_class:
        mock_model_instance = mock_gemini_model_class.return_value
        mock_model_instance.generate_content_async = AsyncMock(side_effect=blocked_error)
        
        response = await mcp_instance.rag_query("test question", "gemini")
        assert response == f"Error interacting with Gemini API: Prompt was blocked - {blocked_error}"


# Generic Exception Test (using Anthropic as an example)
@pytest.mark.asyncio
async def test_rag_query_anthropic_generic_exception(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    generic_exception = Exception("A wild generic error appears!")
    with patch('anthropic.AsyncAnthropic', new_callable=AsyncMock) as mock_anthropic_client:
        mock_client_instance = mock_anthropic_client.return_value
        mock_client_instance.messages.create = AsyncMock(side_effect=generic_exception)
        
        response = await mcp_instance.rag_query("test question", "anthropic")
        assert response == f"An unexpected error occurred with Anthropic API: {generic_exception}"

# --- Test Case for Successful API Call ---
@pytest.mark.asyncio
async def test_rag_query_openai_successful_call(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    
    # Mock successful response from OpenAI
    # Based on server.py: completion.choices[0].message.content.strip()
    mock_choice = MagicMock()
    mock_choice.message.content = "  Mocked OpenAI Response  "
    
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    
    with patch('openai.AsyncOpenAI', new_callable=AsyncMock) as mock_openai_client:
        mock_client_instance = mock_openai_client.return_value
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_completion)
        
        response = await mcp_instance.rag_query("test question", "openai")
        assert response == "Mocked OpenAI Response"

# --- Test for unsupported model ---
@pytest.mark.asyncio
async def test_rag_query_unsupported_model(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    with pytest.raises(ValueError, match="Unsupported model 'kryptonite'."):
        await mcp_instance.rag_query("test question", "kryptonite")

# --- Test LLaMA JSONDecodeError ---
@pytest.mark.asyncio
async def test_rag_query_llama_json_decode_error(mcp_instance, mocker):
    patch_vectorstore_search(mocker)
    
    # Mock the response object that will be returned by httpx.post
    mock_llama_response = AsyncMock()
    mock_llama_response.raise_for_status = MagicMock() # Does not raise error
    # Simulate json.JSONDecodeError when .json() is called
    decode_error = ValueError("Simulated JSONDecodeError") # In httpx, this might be json.JSONDecodeError
    mock_llama_response.json = MagicMock(side_effect=decode_error)

    with patch('httpx.AsyncClient', new_callable=AsyncMock) as mock_http_client:
        mock_client_instance = mock_http_client.return_value.__aenter__.return_value
        mock_client_instance.post = AsyncMock(return_value=mock_llama_response)
        
        response = await mcp_instance.rag_query("test question", "llama")
        # The current code in server.py for LLaMA catches general Exception for JSONDecodeError.
        # Let's ensure the message reflects that.
        assert f"An unexpected error occurred with LLaMA API: {decode_error}" in response
        # If server.py was more specific:
        # assert f"Error interacting with LLaMA API: Could not decode JSON response - {decode_error}" in response

# To make the LLaMA JSONDecodeError test more precise against the server code's
# actual generic exception handler for LLaMA, the assertion should match:
# "An unexpected error occurred with LLaMA API: {e}"
# The current server.py LLaMA error handling for JSONDecodeError is:
#       except Exception as e: # Catch-all for other errors, e.g. JSONDecodeError if response.json() fails
#           return f"An unexpected error occurred with LLaMA API: {e}"
# So the test `test_rag_query_llama_json_decode_error` is correctly asserting against this.

# Note on openai.APIError:
# The constructor for openai.APIError is openai.APIError(message, request, body).
# Providing None for request and body is acceptable for testing.
# Example: openai.APIError("Test API Error", request=MagicMock(), body=None)
# Corrected this in the test_rag_query_openai_api_error.

# Note on httpx.HTTPStatusError:
# Constructor: httpx.HTTPStatusError(message, *, request, response)
# The test was: httpx.HTTPStatusError("Server Error", request=MagicMock(), response=MagicMock())
# Need to ensure the mocked response has status_code and text attributes for the f-string in server.py.

# Note on genai.core.exceptions.GoogleAPIError:
# This is often a base class. More specific errors like `InternalServerError` or `ServiceUnavailable`
# might be raised by the SDK. For testing the broad catch, `GoogleAPIError` is fine.
# Constructor is simple: `GoogleAPIError(message)`.

# Final check on AsyncMock for httpx.AsyncClient context manager:
# `mock_client_instance = mock_http_client.return_value.__aenter__.return_value` is correct.
# `mock_client_instance.post` is then the async method to mock.

# The OpenAI AuthenticationError also needs a `response` argument.
# openai.AuthenticationError(message, response, body)
# Corrected this in the test_rag_query_openai_authentication_error.
# Using MagicMock() for the response object in error constructors is fine.

# Looks good.
print("test_server.py content generated.")
