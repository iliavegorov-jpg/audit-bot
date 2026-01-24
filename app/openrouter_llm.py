import requests
from typing import List, Dict
import numpy as np

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

class OpenRouterLLM:
    """
    OpenRouter API wrapper configured for Claude Sonnet 4.5
    """
    def __init__(self, api_key: str, embedding_dim: int = 256):
        self.api_key = api_key
        self.completion_model = "anthropic/claude-sonnet-4.5"  # Latest Sonnet 4.5
        self.embedding_model = "openai/text-embedding-3-small"  # для embeddings
        self.embedding_dim = embedding_dim

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://audit-bot.local",  # optional
            "X-Title": "Audit Bot"  # optional
        }

    def embed(self, text: str) -> np.ndarray:
        """Generate embeddings using OpenRouter"""
        payload = {
            "model": self.embedding_model,
            "input": text,
            "dimensions": self.embedding_dim
        }
        
        r = requests.post(
            OPENROUTER_EMBEDDINGS_URL, 
            headers=self._headers(), 
            json=payload, 
            timeout=60
        )
        r.raise_for_status()
        data = r.json()
        
        # Extract embedding from response
        if "data" in data and len(data["data"]) > 0:
            vec = data["data"][0]["embedding"]
            return np.array(vec, dtype=np.float32)
        
        raise RuntimeError(f"unexpected embeddings response: {data}")

    def completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2, 
        max_tokens: int = 16000
    ) -> str:
        """
        Generate completion using Claude Sonnet 4.5 via OpenRouter
        
        Args:
            messages: List of message dicts with 'role' and 'text' or 'content'
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text
        """
        # Convert messages format and extract system message
        formatted_messages = []
        system_message = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("text") or msg.get("content", "")
            
            if role == "system":
                system_message = content  # Claude uses separate system parameter
            else:
                formatted_messages.append({"role": role, "content": content})
        
        payload = {
            "model": self.completion_model,  # anthropic/claude-sonnet-4-20250514
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add system message if present (Claude-specific)
        if system_message:
            payload["system"] = system_message
        
        r = requests.post(
            OPENROUTER_API_URL, 
            headers=self._headers(), 
            json=payload, 
            timeout=180
        )
        
        # Raise detailed error if request fails
        if r.status_code != 200:
            error_details = f"OpenRouter Error {r.status_code}: {r.text}"
            raise RuntimeError(error_details)
        
        data = r.json()
        
        # Extract text from OpenRouter response
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        
        raise RuntimeError(f"unexpected completion response: {data}")
