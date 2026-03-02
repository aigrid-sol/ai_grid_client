# AI Grid Client

Python client examples for the **AI Grid** API (`app.ai-grid.io`). Uses the OpenAI-compatible interface for chat, embeddings, and vision/OCR.

## Requirements

- Python 3.10+
- [openai](https://pypi.org/project/openai/) (OpenAI Python client)
- [python-dotenv](https://pypi.org/project/python-dotenv/)

```bash
pip install openai python-dotenv
```

## Setup

1. **Clone or use this repo.**

2. **Create a `.env` file** in this directory with your API key:

   ```
   AI_GRID_KEY=your_api_key_here
   ```

   Do not commit `.env` (it is in `.gitignore`).

## Scripts

| File | Description |
|------|-------------|
| **qwen.py** | Chat completion with **Qwen3-30B-A3B-Thinking**. Sends a simple "Hello!" and prints the reply. |
| **embeding_qwen_alibaba.py** | **Embeddings** with **Alibaba-NLP/gte-Qwen2-7B-instruct**. Gets a vector for the input text and prints its length. |
| **oss.py** | Chat completion with **voxtral-mini-2602**. Sends "Hello!" and prints the reply. |
| **ocr.py** | **Image OCR** with **deepseek-ocr**. Encodes a local image (e.g. `image4.jpg`) as base64, sends it with the prompt "Free OCR.", and prints the extracted text and response time. |

All scripts use:

- **Base URL:** `http://app.ai-grid.io:4000/v1`
- **API key:** from `AI_GRID_KEY` in `.env`

## Usage

Run any script from the project root (with `.env` present):

```bash
python qwen.py
python embeding_qwen_alibaba.py
python oss.py
python ocr.py
```

For **ocr.py**, the image path is hardcoded (`/home/user/ali/client/image4.jpg`). Change it in the script or pass your own path.

## License

Use according to your organization’s terms for the AI Grid API and the underlying models.
