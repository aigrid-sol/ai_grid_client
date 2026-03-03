import argparse
import base64
import io
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("AI_GRID_KEY")
OCR_MODEL = os.getenv("OCR_MODEL")
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_WINDOWS = int(os.getenv("OCR_WINDOWS", "3"))

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)

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

WINDOW_NAMES = ("top", "medium", "bottom")


def pdf_to_images(pdf_path: Path, dpi: int) -> list[tuple[int, Image.Image]]:
    images = convert_from_path(str(pdf_path), dpi=dpi, fmt="JPEG", thread_count=2)
    return [(i + 1, img) for i, img in enumerate(images)]


def split_into_windows(image: Image.Image, n: int = 3) -> list[tuple[str, Image.Image]]:
    w, h = image.size
    if image.mode != "RGB":
        image = image.convert("RGB")
    step = h // n
    out = []
    for i in range(n):
        y0 = i * step
        y1 = h if i == n - 1 else (i + 1) * step
        crop = image.crop((0, y0, w, y1))
        name = WINDOW_NAMES[i] if i < len(WINDOW_NAMES) else f"window_{i+1}"
        out.append((name, crop))
    return out


def encode_pil_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ocr_window(image: Image.Image) -> str | None:
    b64 = encode_pil_image(image)
    kwargs = {
        "model": OCR_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Free OCR."},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    if EXTRA_BODY is not None:
        kwargs["extra_body"] = EXTRA_BODY
    try:
        r = client.chat.completions.create(**kwargs)
        return r.choices[0].message.content
    except Exception as e:
        print(f"  OCR error: {e}", file=sys.stderr)
        return None


def process_page(page_num: int, image: Image.Image, windows_per_page: int) -> str:
    parts = []
    windows = split_into_windows(image, n=windows_per_page)
    for name, crop in windows:
        t0 = time.perf_counter()
        text = ocr_window(crop)
        elapsed = time.perf_counter() - t0
        if text:
            parts.append(text)
            print(f"  page {page_num} {name} {elapsed:.1f}s")
        else:
            print(f"  page {page_num} {name} failed", file=sys.stderr)
    return "\n\n".join(parts)


def main():
    p = argparse.ArgumentParser(description="PDF → images → 3 windows (top/medium/bottom) → sequential OCR")
    p.add_argument("pdf", type=Path, help="Path to PDF file")
    p.add_argument("-o", "--output-dir", type=Path, default=None, help="Directory for page .txt files (default: next to PDF)")
    p.add_argument("--dpi", type=int, default=OCR_DPI, help="DPI for PDF→image (default from OCR_DPI)")
    p.add_argument("--windows", type=int, default=OCR_WINDOWS, help="Windows per page (default 3: top, medium, bottom)")
    args = p.parse_args()

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        print(f"Error: not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir or pdf_path.parent
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not BASE_URL or not API_KEY or not OCR_MODEL:
        print("Error: set BASE_URL, AI_GRID_KEY, OCR_MODEL in .env", file=sys.stderr)
        sys.exit(1)

    n_windows = max(1, args.windows)
    print(f"PDF: {pdf_path}  DPI: {args.dpi}  Windows/page: {n_windows}  Out: {out_dir}")

    pages = pdf_to_images(pdf_path, args.dpi)
    print(f"Pages: {len(pages)}\n")

    full_text = []
    for page_num, image in pages:
        page_text = process_page(page_num, image, n_windows)
        full_text.append(page_text)
        out_file = out_dir / f"{pdf_path.stem}_p{page_num:04d}.txt"
        out_file.write_text(page_text, encoding="utf-8")
        print(f"  saved {out_file.name}")

    combined = out_dir / f"{pdf_path.stem}_full.txt"
    combined.write_text("\n\n".join(full_text), encoding="utf-8")
    print(f"\nDone. Full doc: {combined}")


if __name__ == "__main__":
    main()
