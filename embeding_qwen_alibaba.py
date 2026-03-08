"""
Embeddings using the configured embedding model (OpenAI-compatible endpoint).
Configuration via .env: AI_GRID_KEY, BASE_URL, EMBEDDING_MODEL.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("EMBEDDING_MODEL")
client = OpenAI(base_url=BASE_URL, api_key=AI_GRID_KEY)


def get_embedding(text: str) -> list[float]:
    """Return the embedding vector for the given text."""
    response = client.embeddings.create(model=MODEL, input=text)
    return response.data[0].embedding


def main():
    """Compute embedding for a sample text and print its dimension."""
    embedding = get_embedding("Explain what embeddings are.")
    print("Embedding length:", len(embedding))


if __name__ == "__main__":
    main()
