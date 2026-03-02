import base64
import time
from openai import OpenAI
from dotenv import load_dotenv
import os 
load_dotenv()

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


image_base64 = encode_image("~/image4.jpg")

client = OpenAI(
    base_url="http://app.ai-grid.io:4000/v1",
    api_key=AI_GRID_KEY,
)

start = time.time()

response = client.chat.completions.create(
    model="deepseek-ocr",   
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": "Free OCR."
                }
            ]
        }
    ],
    temperature=0.0,
    max_tokens=2048,
    extra_body={
        "skip_special_tokens": False,
        "vllm_xargs": {
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": [128821, 128822],
        },
    },
)

print(f"Response time: {time.time() - start:.2f}s")
print("\n=== OCR RESULT ===")
print(response.choices[0].message.content)
