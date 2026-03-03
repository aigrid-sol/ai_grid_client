from openai import OpenAI
from dotenv import load_dotenv
import os 
load_dotenv()

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("QWEN_MODEL")
client = OpenAI(base_url=BASE_URL, api_key=AI_GRID_KEY)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)