import json
import os
from dataclasses import dataclass
from typing import Any
from openai import OpenAI

@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 800


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig) -> None:
        """Create an OpenAI-compatible client from config."""
        api_key = config.api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("Missing API key. Set config.endpoint.api_key or OPENAI_API_KEY.")
        self.config = config
        self.client = OpenAI(api_key=api_key, base_url=config.base_url)

    def chat_text(self, system_prompt: str, user_prompt: str, temperature: float | None = None) -> str:
        """Send a chat completion request and return the text response."""
        completion = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content or ""

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float | None = None) -> dict[str, Any]:
        """Send a chat completion request and parse JSON from the response."""
        raw = self.chat_text(system_prompt, user_prompt, temperature=temperature)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            cleaned = _extract_json(raw)
            return json.loads(cleaned)


def _extract_json(raw: str) -> str:
    """Extract a JSON object substring from a raw model response."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model output is not valid JSON: {raw}")
    return text[start : end + 1]
