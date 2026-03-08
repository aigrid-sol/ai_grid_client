# AI Grid Client

Python client examples for the **AI Grid** API (`app.ai-grid.io`). Uses the OpenAI-compatible interface for chat, embeddings, vision/OCR, and tool calling.

## Requirements

- Python 3.10+
- [openai](https://pypi.org/project/openai/) — OpenAI Python client
- [python-dotenv](https://pypi.org/project/python-dotenv/) — load `.env`
- [langgraph](https://pypi.org/project/langgraph/), [langchain-core](https://pypi.org/project/langchain-core/), [langchain-openai](https://pypi.org/project/langchain-openai/) — for `langraph_oss.py`
- [llama-index](https://pypi.org/project/llama-index/) + [llama-index-llms-openai-like](https://pypi.org/project/llama-index-llms-openai-like/) — optional, for `qwen_tool_caller.py` (ReActAgent + OpenAILike)

```bash
pip install -r requirements.txt
# or selectively:
pip install openai python-dotenv
pip install langgraph langchain-core langchain-openai   # for langraph_oss.py
pip install llama-index llama-index-llms-openai-like   # for qwen_tool_caller.py
```

## Setup

1. **Clone or use this repo.**

2. **Create a `.env` file** with at least:

   ```
   BASE_URL=http://app.ai-grid.io:4000/v1
   AI_GRID_KEY=your_api_key_here
   QWEN_MODEL=Qwen3-30B-A3B-Thinking
   OSS_MODEL=gpt-oss-120b
   EMBEDDING_MODEL=Alibaba-NLP/gte-Qwen2-7B-instruct
   OCR_MODEL=deepseek-ocr
   IMAGE_FOLDER=/path/to/ali/client/images
   OCR_BATCH_CONCURRENCY=5
   AI_GRID_BASE_URL=http://app.ai-grid.io:4000/v1
   AI_GRID_TOOL_MODEL=Qwen3-30B-A3B-Thinking
   ```

   For **voxtral.py** (local Voxtral, e.g. `voxtral-live` container), add:

   ```
   AUDIO_PATH=/path/to/audio.mp3
   VOXTRAL_BASE_URL=http://localhost:8000/v1   # optional
   VOXTRAL_MODEL=mistralai/Voxtral-Mini-4B-Realtime-2602   # optional
   ```

   Do not commit `.env` (it is in `.gitignore`).

3. **`images/`** — Sample images for testing OCR. Run `ocr.py` or `ocr_batch.py` to run OCR v2 on these images; results are saved as `.txt` files next to each image.

## Scripts

| File | Description |
|------|-------------|
| **qwen.py** | Chat completion with **Qwen3-30B-A3B-Thinking**. Sends a simple "Hello!" and prints the reply. |
| **oss.py** | Chat completion with **gpt-oss-120b** (OSS model). Sends "Hello!" and prints the reply. |
| **langraph_oss.py** | **LangGraph** example using the OSS backend: multi-node graph (generate → optional refine), conditional edge, and state with messages + step count. Uses `BASE_URL`, `AI_GRID_KEY`, `OSS_MODEL` from `.env`. |
| **embeding_qwen_alibaba.py** | **Embeddings** with **Alibaba-NLP/gte-Qwen2-7B-instruct**. Gets a vector for the input text and prints its length. |
| **ocr.py** | **Image OCR** (OCR v2). Reads images from `IMAGE_FOLDER`, sends them with prompt "Free OCR.", prints extracted text and response time. Use `images/` for test images; results are written as `.txt` next to each image. |
| **ocr_batch.py** | **Batch OCR** (OCR v2). Same as ocr.py but sends multiple images concurrently; set `OCR_BATCH_CONCURRENCY` in `.env`. |
| **ocr_doc.py** | **Long-doc OCR**: PDF → page images → each page split into 3 windows (top, medium, bottom) → sequential OCR per window. Output: one `.txt` per page plus a full-doc `.txt`. Requires `pdf2image` + system poppler. |
| **qwen_tool_caller.py** | **Tool calling** with LlamaIndex: ReActAgent, FunctionTool, OpenAILike (Qwen3-30B-A3B-Thinking). Custom model names; server must support tool calling (e.g. vLLM `--enable-auto-tool-choice`). |
| **voxtral.py** | **Voxtral** transcription: uses `AUDIO_PATH` from `.env`, calls Voxtral server (e.g. `voxtral-live` container). REST or realtime mode. |

All scripts use:

- **Base URL:** `http://app.ai-grid.io:4000/v1`
- **API key:** from `AI_GRID_KEY` in `.env`

## Usage

From the project root (with `.env` present):

```bash
python qwen.py
python oss.py
python langraph_oss.py
python embeding_qwen_alibaba.py
python ocr.py
python ocr_batch.py
python ocr_doc.py path/to/doc.pdf -o path/to/output_dir
python qwen_tool_caller.py
python voxtral.py
```

**Notes:**

- **langraph_oss.py** — Uses the same OSS endpoint as `oss.py`. Graph: user message → generate (LLM) → if reply long, refine (summarize) → end. You can call `run("your question")` or `run_stream("your question")` from code.
- **qwen_tool_caller.py** — Add tools with `FunctionTool.from_defaults(fn=your_function)`. Uses OpenAILike so model names like `Qwen3-30B-A3B-Thinking` work; endpoint must support tool calling.
- **voxtral.py** — Set `AUDIO_PATH` in `.env`. Run `python voxtral.py` for REST transcription or with `realtime` for WebSocket.
- **ocr.py / ocr_batch.py** — Use `IMAGE_FOLDER` in `.env` (e.g. `ali/client/images`). The `images/` folder includes sample images for testing; OCR v2 results are written as `.txt` next to each image.
- **ocr_doc.py** — For long PDFs: converts PDF to images, splits each page into 3 vertical windows (top, medium, bottom), runs OCR on each window sequentially. Optional `.env`: `OCR_DPI` (default 200), `OCR_WINDOWS` (default 3). Requires `poppler-utils` on the system.

**ocr_doc.py — install and run:**

```bash
cd /home/user/ali/client
pip install pdf2image Pillow   # if not already
# On Debian/Ubuntu: apt install poppler-utils  (for pdf2image)

python ocr_doc.py path/to/document.pdf
python ocr_doc.py path/to/document.pdf -o ./ocr_out --dpi 200 --windows 3
```

## License

Use according to your organization’s terms for the AI Grid API and the underlying models.
