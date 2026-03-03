from openai import OpenAI
from dotenv import load_dotenv
import os 
load_dotenv()

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("OSS_MODEL")
client = OpenAI(api_key=AI_GRID_KEY, base_url=BASE_URL)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
