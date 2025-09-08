"""
Ollama service integration for the LLM Network Gateway.

This module handles all communication with the Ollama API, including:
- Model discovery and validation
- Chat completions with streaming support
- Connection health checks and error handling
- Context management for conversations
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
from datetime import datetime
import json

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = 120  # seconds for LLM responses
CONNECTION_TIMEOUT = 10  # seconds for connection attempts


class OllamaConnectionError(Exception):
    """Raised when unable to connect to Ollama service."""
    pass


class OllamaModelError(Exception):
    """Raised when model-related errors occur."""
    pass


class OllamaService:
    """
Service class for interacting with Ollama API.
Handles model management, chat completions, and error handling.
"""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(CONNECTION_TIMEOUT, read=OLLAMA_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=5)
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def health_check(self) -> bool:
        """
Check if Ollama service is accessible and responding.
Returns True if healthy, False otherwise.
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
Retrieve list of available models from Ollama.
Returns list of model dictionaries with name, size, and modified date.

Raises:
OllamaConnectionError: If unable to connect to Ollama
OllamaModelError: If no models are available
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")

            if response.status_code != 200:
                raise OllamaConnectionError(f"Ollama API returned status {response.status_code}")

            data = response.json()
            models = data.get("models", [])

            if not models:
                raise OllamaModelError("No models available in Ollama")

            # Format model information for frontend consumption
            formatted_models = []
            for model in models:
                formatted_models.append({
                    "name": model.get("name", "unknown"),
                    "size": self._format_size(model.get("size", 0)),
                    "modified": model.get("modified_at", ""),
                    "digest": model.get("digest", "")[:12]  # Short digest for identification
                })

            return formatted_models

        except httpx.RequestError as e:
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")
        except json.JSONDecodeError:
            raise OllamaConnectionError("Invalid response from Ollama API")

    async def validate_model(self, model_name: str) -> bool:
        """
Check if a specific model is available in Ollama.

Args:
model_name: Name of the model to validate

Returns:
True if model exists, False otherwise
        """
        try:
            models = await self.get_available_models()
            return any(model["name"] == model_name for model in models)
        except (OllamaConnectionError, OllamaModelError):
            return False

    async def chat_completion(
            self,
            model: str,
            messages: List[Dict[str, str]],
            stream: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
Send chat completion request to Ollama with conversation history.

Args:
model: Name of the model to use
messages: List of message dictionaries with 'role' and 'content'
stream: Whether to stream the response or return complete response

Yields:
Dictionary containing response chunks or complete response

Raises:
OllamaModelError: If model is not available
OllamaConnectionError: If request fails
        """
        # Validate model exists
        if not await self.validate_model(model):
            raise OllamaModelError(f"Model '{model}' is not available")

        # Prepare the request payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        try:
            if stream:
                # Stream the response for real-time updates
                async with self.client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=payload
                ) as response:

                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise OllamaConnectionError(
                            f"Ollama API error {response.status_code}: {error_text.decode()}"
                        )

                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                yield chunk
                            except json.JSONDecodeError:
                                continue  # Skip malformed JSON lines
            else:
                # Get complete response
                response = await self.client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )

                if response.status_code != 200:
                    raise OllamaConnectionError(
                        f"Ollama API error {response.status_code}: {response.text}"
                    )

                yield response.json()

        except httpx.RequestError as e:
            raise OllamaConnectionError(f"Request to Ollama failed: {e}")

    async def generate_title(self, model: str, first_message: str) -> str:
        """
Generate a concise title for a chat session based on the first message.

Args:
model: Model to use for title generation
first_message: The first user message in the conversation

Returns:
Generated title string (max 50 characters)
        """
        try:
            # Create a simple prompt for title generation
            title_prompt = f"Generate a short, descriptive title (max 5 words) for a conversation that starts with: '{first_message[:100]}...'"

            messages = [{"role": "user", "content": title_prompt}]

            async for response in self.chat_completion(model, messages, stream=False):
                title = response.get("message", {}).get("content", "New Chat").strip()
                # Clean up the title and limit length
                title = title.replace('"', '').replace("'", "").strip()
                return title[:50] if len(title) > 50 else title

        except Exception:
            # Fallback to timestamp-based title if generation fails
            return f"Chat {datetime.now().strftime('%H:%M')}"

        return "New Chat"

    def _format_size(self, size_bytes: int) -> str:
        """
Format file size in bytes to human-readable format.

Args:
size_bytes: Size in bytes

Returns:
Formatted size string (e.g., "1.2 GB", "500 MB")
        """
        if size_bytes == 0:
            return "Unknown"

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0

        return f"{size_bytes:.1f} PB"

    async def close(self):
        """Close the HTTP client connection."""
        await self.client.aclose()


# Global service instance
_ollama_service: Optional[OllamaService] = None


async def get_ollama_service() -> OllamaService:
    """
Dependency function to get Ollama service instance.
Creates a new instance if none exists.
    """
    global _ollama_service
    if _ollama_service is None:
        _ollama_service = OllamaService()
    return _ollama_service


async def close_ollama_service():
    """Clean up Ollama service connection."""
    global _ollama_service
    if _ollama_service:
        await _ollama_service.close()
        _ollama_service = None
