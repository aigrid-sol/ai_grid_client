import base64
import time
import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("AI_GRID_KEY")
OCR_MODEL = os.getenv("OCR_MODEL")
IMAGE_PATH = os.getenv("IMAGE_PATH")
IMAGE_FOLDER = Path(os.getenv("IMAGE_FOLDER")).expanduser().resolve()


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)

EXTRA_BODY = None
if "localhost" in BASE_URL or "127.0.0.1" in BASE_URL:
    EXTRA_BODY = {
        "skip_special_tokens": False,
        "vllm_xargs": {
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": [128821, 128822],
        },
    }


def run_ocr(image_path: Path) -> bool:
    print(f"\nProcessing: {image_path.name}")
    try:
        image_base64 = encode_image(image_path)
    except OSError as e:
        print(f"Error reading image: {e}", file=sys.stderr)
        return False

    start = time.time()
    kwargs = {
        "model": OCR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                    {"type": "text", "text": "Free OCR."},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 8000,
    }
    if EXTRA_BODY is not None:
        kwargs["extra_body"] = EXTRA_BODY

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"Error calling OCR API: {e}", file=sys.stderr)
        return False

    result_text = response.choices[0].message.content
    elapsed = time.time() - start

    print(f"Response time: {elapsed:.2f}s")
    print("=== OCR RESULT ===")
    print(result_text)

    output_file = image_path.with_suffix(".txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result_text)
    print(f"Saved OCR to: {output_file}")
    return True


def main():
    print(f"Base URL: {BASE_URL}")

    if IMAGE_PATH:
        path = Path(IMAGE_PATH).expanduser().resolve()
        if path.exists():
            sys.exit(0 if run_ocr(path) else 1)
        print(f"Warning: IMAGE_PATH not found: {path}, using IMAGE_FOLDER", file=sys.stderr)

    folder = Path(IMAGE_FOLDER)
    if not folder.exists():
        print(f"Error: IMAGE_FOLDER does not exist: {folder}", file=sys.stderr)
        print("Create it and add .jpg files, or set IMAGE_PATH=/path/to/image.jpg", file=sys.stderr)
        sys.exit(1)

    images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))
    if not images:
        print(f"No .jpg/.jpeg files in {folder}", file=sys.stderr)
        print("Add images or set IMAGE_PATH=/path/to/image.jpg", file=sys.stderr)
        sys.exit(1)

    ok, fail = 0, 0
    for image_path in images:
        if run_ocr(image_path):
            ok += 1
        else:
            fail += 1
    if fail:
        print(f"\nDone: {ok} succeeded, {fail} failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
