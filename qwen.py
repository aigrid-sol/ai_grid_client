"""
Chat completion using the Qwen model (OpenAI-compatible endpoint).
Configuration via .env: AI_GRID_KEY, BASE_URL, QWEN_MODEL.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("QWEN_MODEL")
client = OpenAI(base_url=BASE_URL, api_key=AI_GRID_KEY)


def run_chat() -> str:
    """Send a single user message to the Qwen model and return the assistant reply."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    return response.choices[0].message.content or ""


def main():
    """Run a simple chat and print the reply."""
    print(run_chat())


if __name__ == "__main__":
    main()
