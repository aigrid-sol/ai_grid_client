from openai import OpenAI
from dotenv import load_dotenv
import os 
load_dotenv()

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("EMBEDDING_MODEL")
client = OpenAI(base_url=BASE_URL, api_key=AI_GRID_KEY)

response = client.embeddings.create(
    model=MODEL,
    input="Explain what embeddings are."
)

embedding = response.data[0].embedding
print("Embedding length:", len(embedding))