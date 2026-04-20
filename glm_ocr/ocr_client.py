from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

from openai import APIStatusError, OpenAI


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".gif":
        mime = "image/gif"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_image_url(image_path: str | None, image_url: str | None) -> str:
    if image_url:
        return image_url
    if not image_path:
        raise ValueError("Pass --image or --image-url.")
    p = Path(image_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    return image_to_data_url(p)


def main() -> int:
    """
    Simple GLM OCR client using OpenAI-compatible chat completions.
    """

    parser = argparse.ArgumentParser(
        description="GLM OCR client (OpenAI-compatible image-to-text)."
    )
    parser.add_argument(
        "--image",/home/user/ali/client/images/1.jpg
        default="",
        help="Local image path.",
    )
    parser.add_argument("--image-url", help="Public image URL (or data URL).")
    parser.add_argument("--prompt", default="Free OCR.", help="OCR instruction text.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="OpenAI-compatible base URL (default: OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "you api key here"),
        help="API key (default: OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "zai-org/GLM-OCR"),
        help="Model name (default: OPENAI_MODEL or zai-org/GLM-OCR).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="Request timeout in seconds.",
    )
    args = parser.parse_args()

    if not args.base_url:
        print("Missing --base-url or OPENAI_BASE_URL.", file=sys.stderr)
        return 1
    if not args.api_key:
        print("Missing --api-key or OPENAI_API_KEY.", file=sys.stderr)
        return 1

    try:
        image_ref = build_image_url(args.image, args.image_url)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": "image_url", "image_url": {"url": image_ref}},
            ],
        }
    ]

    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0,
        )
    except APIStatusError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not resp.choices:
        print("No choices in response.", file=sys.stderr)
        return 1

    content = resp.choices[0].message.content
    if isinstance(content, str):
        print(content)
    else:
        print(str(content))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
