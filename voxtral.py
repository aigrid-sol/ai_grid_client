#!/usr/bin/env python3
"""
Realtime Voxtral client: send audio over WebSocket and print transcription.
Host, port, and model can be set via CLI or environment; no sensitive values are printed.
"""
import argparse
import asyncio
import base64
import json
import os
import sys

import librosa
import numpy as np
import websockets

from dotenv import load_dotenv

load_dotenv()


def audio_to_pcm16_base64(audio_path: str) -> str:
    """Load an audio file and return base64-encoded PCM16 at 16 kHz mono."""
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    pcm16 = (audio * 32767).astype(np.int16)
    return base64.b64encode(pcm16.tobytes()).decode("utf-8")


async def realtime_transcribe(audio_path: str, host: str, port: int, model: str):
    """
    Connect to the Realtime API, send audio from the given path, and print transcription deltas.
    Host/port are not logged; connection and session messages are generic.
    """
    uri = f"ws://{host}:{port}/v1/realtime"
    print("Connecting to realtime endpoint...")

    async with websockets.connect(uri) as ws:
        response = json.loads(await ws.recv())
        if response["type"] == "session.created":
            print("Session created.")
        else:
            print("Unexpected response:", response.get("type"))
            return

        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        print("Loading audio...")
        audio_base64 = audio_to_pcm16_base64(audio_path)

        chunk_size = 4096
        audio_bytes = base64.b64decode(audio_base64)
        total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size
        print(f"Sending {total_chunks} audio chunks...")

        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )

        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        print("Audio sent. Waiting for transcription...\n")

        print("Transcription: ", end="", flush=True)
        while True:
            response = json.loads(await ws.recv())
            if response["type"] == "transcription.delta":
                print(response["delta"], end="", flush=True)
            elif response["type"] == "transcription.done":
                print(f"\n\nFinal transcription: {response['text']}")
                if response.get("usage"):
                    print("Usage:", response["usage"])
                break
            elif response["type"] == "error":
                print("\nError:", response.get("error", "unknown"))
                break


def main():
    """Parse CLI (or use env); run realtime transcription. Host/port defaults are not printed."""
    parser = argparse.ArgumentParser(description="Realtime WebSocket transcription client")
    parser.add_argument("--model", type=str, default=os.getenv("VOXTRAL_MODEL", "mistralai/Voxtral-Mini-4B-Realtime-2602"))
    parser.add_argument("--audio_path", type=str, default=os.getenv("AUDIO_PATH"))
    parser.add_argument("--host", type=str, default=os.getenv("VOXTRAL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VOXTRAL_PORT", "8000")))
    args = parser.parse_args()

    if not args.audio_path:
        print("Error: set --audio_path or AUDIO_PATH in .env", file=sys.stderr)
        sys.exit(1)

    asyncio.run(realtime_transcribe(args.audio_path, args.host, args.port, args.model))


if __name__ == "__main__":
    main()
