import requests
from typing import List, Dict
import numpy as np

EMBEDDINGS_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

class YandexLLM:
    def __init__(self, api_key: str, completion_model_uri: str, embedding_model_uri: str, embedding_dim: int):
        self.api_key = api_key
        self.completion_model_uri = completion_model_uri
        self.embedding_model_uri = embedding_model_uri
        self.embedding_dim = embedding_dim

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, text: str) -> np.ndarray:
        payload = {
            "modelUri": self.embedding_model_uri,
            "text": text,
            "dim": str(self.embedding_dim),
        }
        r = requests.post(EMBEDDINGS_URL, headers=self._headers(), json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        vec = data.get("embedding") or data.get("vector") or data.get("data", {}).get("embedding")
        if vec is None:
            raise RuntimeError(f"unexpected embeddings response: {data}")
        return np.array(vec, dtype=np.float32)

    def completion(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 3500) -> str:
        payload = {
            "modelUri": self.completion_model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens),
            },
            "messages": messages,
        }
        r = requests.post(COMPLETION_URL, headers=self._headers(), json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        txt = None
        if "result" in data and "alternatives" in data["result"] and data["result"]["alternatives"]:
            txt = data["result"]["alternatives"][0]["message"]["text"]
        if txt is None:
            txt = data.get("text") or data.get("resultText")
        if txt is None:
            raise RuntimeError(f"unexpected completion response: {data}")
        return txt
