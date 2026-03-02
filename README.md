# AI Grid Client

Python client examples for the **AI Grid** API (`app.ai-grid.io`). Uses the OpenAI-compatible interface for chat, embeddings, vision/OCR, and tool calling.

## Requirements

- Python 3.10+
- [openai](https://pypi.org/project/openai/) — OpenAI Python client
- [python-dotenv](https://pypi.org/project/python-dotenv/) — load `.env`
- [llama-index](https://pypi.org/project/llama-index/) + [llama-index-llms-openai-like](https://pypi.org/project/llama-index-llms-openai-like/) — for `qwen_tool_caller.py` (ReActAgent + OpenAILike; custom model names supported)

```bash
pip install openai python-dotenv
pip install llama-index llama-index-llms-openai-like   # for qwen_tool_caller.py
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
| **oss.py** | Chat completion with **gpt-oss-120b**. Sends "Hello!" and prints the reply. |
| **ocr.py** | **Image OCR** with **deepseek-ocr**. Encodes a local image (e.g. `image4.jpg`) as base64, sends it with the prompt "Free OCR.", and prints the extracted text and response time. |
| **qwen_tool_caller.py** | **Tool calling** with LlamaIndex: ReActAgent, FunctionTool, OpenAILike (Qwen3-30B-A3B-Thinking). Custom model names; server must support tool calling (e.g. vLLM `--enable-auto-tool-choice`). |

All scripts use:

- **Base URL:** `http://app.ai-grid.io:4000/v1`
- **API key:** from `AI_GRID_KEY` in `.env`

## Usage

From the project root (with `.env` present):

```bash
python qwen.py
python embeding_qwen_alibaba.py
python oss.py
python ocr.py
python qwen_tool_caller.py
```

**Notes:**

- **qwen_tool_caller.py** — Add tools with `FunctionTool.from_defaults(fn=your_function)`. Uses OpenAILike so model names like `Qwen3-30B-A3B-Thinking` work; endpoint must support tool calling.
- **ocr.py** — Image path is hardcoded; change it in the script if needed.

## License

Use according to your organization’s terms for the AI Grid API and the underlying models.
