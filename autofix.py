#!/usr/bin/env python3
"""Автофикс yc_completion_model_uri и OpenRouterLLM"""
import sys

if len(sys.argv) < 2:
    print("Usage: python autofix.py <bot.py file>")
    sys.exit(1)

filepath = sys.argv[1]

with open(filepath, 'r') as f:
    content = f.read()

# Fix 1: yc_completion_model_uri
content = content.replace(
    'completion_model_uri=SET.yc_completion_model_uri,',
    'completion_model_uri="",  # не используется'
)

# Fix 2: import
content = content.replace(
    'from .claude_llm import ClaudeLLM',
    'from .openrouter_llm import OpenRouterLLM'
)

# Fix 3: initialization
content = content.replace(
    'llm = ClaudeLLM(api_key=SET.claude_api_key)',
    'llm = OpenRouterLLM(api_key=SET.openrouter_api_key, embedding_dim=256)'
)

with open(filepath, 'w') as f:
    f.write(content)

print(f"✅ Fixed {filepath}")
print("  - yc_completion_model_uri")
print("  - OpenRouterLLM import")
print("  - OpenRouterLLM initialization")
