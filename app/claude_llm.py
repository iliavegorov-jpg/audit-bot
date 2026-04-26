import anthropic
from typing import List, Dict

class ClaudeLLM:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def completion(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 8000) -> str:
        system_msg = ""
        user_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_msg = msg.get("text", "")
            elif msg.get("role") == "user":
                user_messages.append({
                    "role": "user",
                    "content": msg.get("text", "")
                })
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_msg,
            messages=user_messages
        )
        
        return response.content[0].text
