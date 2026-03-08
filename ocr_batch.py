"""
Batch OCR: process multiple images concurrently via the configured OCR API.
Uses IMAGE_FOLDER and OCR_BATCH_CONCURRENCY from .env; endpoint and key from BASE_URL and AI_GRID_KEY.
"""
import asyncio
import base64
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("AI_GRID_KEY")
OCR_MODEL = os.getenv("OCR_MODEL")
IMAGE_FOLDER = Path(os.getenv("IMAGE_FOLDER", "")).expanduser().resolve()
CONCURRENCY = int(os.getenv("OCR_BATCH_CONCURRENCY", "5"))

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)

EXTRA_BODY = None
if BASE_URL and ("localhost" in BASE_URL or "127.0.0.1" in BASE_URL):
    EXTRA_BODY = {
        "skip_special_tokens": False,
        "vllm_xargs": {
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": [128821, 128822],
        },
    }


def encode_image(path: Path) -> str:
    """Read image file and return base64-encoded bytes."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def ocr_one(image_path: Path, semaphore: asyncio.Semaphore) -> tuple[Path, str | None, float]:
    """
    Run OCR on a single image (rate-limited by semaphore).
    Returns (path, extracted_text_or_none, elapsed_seconds).
    """
    async with semaphore:
        try:
            image_base64 = encode_image(image_path)
        except OSError as e:
            print(f"Error reading {image_path.name}: {e}", file=sys.stderr)
            return (image_path, None, 0.0)

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
            "max_tokens": 4096,
        }
        if EXTRA_BODY is not None:
            kwargs["extra_body"] = EXTRA_BODY

        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content
            elapsed = time.perf_counter() - start
            return (image_path, text, elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"Error OCR {image_path.name}: {e}", file=sys.stderr)
            return (image_path, None, elapsed)


async def run_batch(image_paths: list[Path]) -> list[tuple[Path, str | None, float]]:
    """Process all images with bounded concurrency; returns list of (path, text, elapsed)."""
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [ocr_one(p, semaphore) for p in image_paths]
    return await asyncio.gather(*tasks)


def main():
    """Discover images in the configured folder, run batch OCR, and write .txt results next to each image."""
    print("Using configured endpoint. Concurrency:", CONCURRENCY)

    folder = Path(IMAGE_FOLDER)
    if not folder.exists():
        print("Error: IMAGE_FOLDER does not exist.", file=sys.stderr)
        sys.exit(1)

    images = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))
    if not images:
        print("No .jpg/.jpeg files in the configured folder.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(images)} images (max {CONCURRENCY} at a time)...\n")
    start = time.perf_counter()
    results = asyncio.run(run_batch(images))
    total_elapsed = time.perf_counter() - start

    ok, fail = 0, 0
    for image_path, text, elapsed in results:
        if text is not None:
            out = image_path.with_suffix(".txt")
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  {image_path.name}  {elapsed:.1f}s  -> {out.name}")
            ok += 1
        else:
            print(f"  {image_path.name}  FAILED")
            fail += 1

    print(f"\nDone: {ok} succeeded, {fail} failed  (total wall time: {total_elapsed:.1f}s)")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
