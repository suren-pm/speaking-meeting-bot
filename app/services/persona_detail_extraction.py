import json
import os
from typing import Any, Dict, Optional

from loguru import logger


async def extract_persona_details_from_prompt(
    prompt_text: str,
) -> Optional[Dict[str, Any]]:
    """
    Analyzes a prompt to extract persona details.
    Uses Anthropic Claude instead of OpenAI GPT-4o.
    """
    import anthropic

    extraction_prompt = f"""Analyze the following text prompt and extract the persona's name, gender, a brief description for image generation, and a list of characteristics. If no explicit name is mentioned, generate a concise, descriptive name that clearly indicates the persona's role or key trait, based on the description and characteristics. This name should not be a personal name unless explicitly provided in the prompt.

Prompt: {prompt_text}

Extract the information in the following JSON format. If a field cannot be determined, use null or an empty list:
{{
  "name": "string or null",
  "gender": "male, female, non-binary, or null",
  "description": "string or null",
  "characteristics": ["string", ...]
}}

Return ONLY valid JSON, no other text."""

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set.")
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": extraction_prompt}],
        )
        content = message.content[0].text.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        extracted_data = json.loads(content)
        extracted_data["name"] = extracted_data.get("name") or "Custom Bot"
        extracted_data["gender"] = extracted_data.get("gender") or "male"
        extracted_data["description"] = extracted_data.get("description") or prompt_text
        characteristics = extracted_data.get("characteristics")
        extracted_data["characteristics"] = characteristics if isinstance(characteristics, list) else []
        return extracted_data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {e}")
        return None
    except Exception as e:
        logger.error(f"Error during persona details extraction: {e}")
        return None
