"""
Interactive multi-turn chat with conversation history, context management, and markdown rendering.
Configuration via .env: AI_GRID_KEY, BASE_URL, OSS_MODEL (or QWEN_MODEL).
Features:
  - Multi-turn conversation with full history
  - Context window management (token counting)
  - Session save/load
  - Rich markdown rendering
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("OSS_MODEL") or os.getenv("QWEN_MODEL", "gpt-oss-120b")

client = OpenAI(base_url=BASE_URL, api_key=AI_GRID_KEY)

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

CONTEXT_WINDOW = 4096
TOKENS_PER_MESSAGE = 50


class ConversationSession:
    """Manage conversation state, history, and persistence."""

    def __init__(self, session_name: Optional[str] = None):
        self.session_name = (
            session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.messages: list[dict] = []
        self.created_at = datetime.now().isoformat()
        self.session_path = SESSIONS_DIR / f"{self.session_name}.json"

        if self.session_path.exists():
            self.load()

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_api_messages(self) -> list[dict]:
        """Return messages in OpenAI API format (no timestamps)."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def estimate_tokens(self) -> int:
        """Rough estimate of token count in conversation."""
        return (
            sum(len(m["content"].split()) for m in self.messages) // 4
            + len(self.messages) * TOKENS_PER_MESSAGE
        )

    def should_truncate(self) -> bool:
        """Check if context window is getting full."""
        return self.estimate_tokens() > (CONTEXT_WINDOW * 0.8)

    def truncate_context(self, keep_last_n: int = 10) -> None:
        """Keep last N messages to fit context window."""
        if len(self.messages) > keep_last_n:
            console.print(
                f"[yellow]⚠️  Truncating history: keeping last {keep_last_n} messages[/yellow]"
            )
            self.messages = self.messages[-keep_last_n:]

    def save(self) -> None:
        """Persist session to JSON file."""
        with open(self.session_path, "w") as f:
            json.dump(
                {
                    "session_name": self.session_name,
                    "created_at": self.created_at,
                    "messages": self.messages,
                },
                f,
                indent=2,
            )
        console.print(f"[green]✓ Session saved: {self.session_path}[/green]")

    def load(self) -> None:
        """Load session from JSON file."""
        if self.session_path.exists():
            with open(self.session_path, "r") as f:
                data = json.load(f)
                self.messages = data.get("messages", [])
                self.created_at = data.get("created_at", datetime.now().isoformat())
            console.print(f"[green]✓ Loaded session: {self.session_name}[/green]")

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []
        console.print("[yellow]Conversation cleared[/yellow]")

    def show_summary(self) -> None:
        """Print session summary."""
        created = datetime.fromisoformat(self.created_at).strftime(
            "%b %d, %Y at %I:%M %p"
        )
        token_count = self.estimate_tokens()
        console.print(
            Panel(
                f"[bold]{self.session_name}[/bold]\n"
                f"Messages: {len(self.messages)} | "
                f"Tokens ~{token_count}/{CONTEXT_WINDOW} | "
                f"Created: {created}",
                title="📋 Session Info",
                border_style="blue",
            )
        )


def stream_chat(session: ConversationSession, user_message: str) -> str:
    """
    Send user message and stream assistant response with rich markdown rendering.
    Returns full response content.
    """
    session.add_message("user", user_message)

    if session.should_truncate():
        session.truncate_context(keep_last_n=10)

    console.print(f"\n[cyan]🤖 AI:[/cyan]")  # noqa: F541

    full_response = []
    with console.status("[yellow]Thinking...[/yellow]"):
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=session.get_api_messages(),
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    full_response.append(delta)

            response_text = "".join(full_response)

        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            return ""

    if full_response:
        render_response(response_text)

    session.add_message("assistant", response_text)
    return response_text


def render_response(content: str) -> None:
    """Render content as markdown, fallback to plain text."""
    try:
        md = Markdown(content)
        console.print(md)
    except Exception:
        console.print(content)


def print_session_history(session: ConversationSession) -> None:
    """Display session messages with formatting."""
    if session.messages:
        console.print(
            f"[green]Resuming conversation with {len(session.messages)} messages[/green]\n"
        )
        for msg in session.messages:
            role = (
                "[cyan]🤖 AI[/cyan]"
                if msg["role"] == "assistant"
                else "[green]You[/green]"
            )
            console.print(f"[dim]{role}:[/dim]")
            render_response(msg["content"])
            console.print()
    else:
        console.print("[dim]Empty session[/dim]")


def handle_session_command(cmd: str) -> Optional[str]:
    """Handle commands at session prompt. Returns session name to load or None to exit."""
    cmd_lower = cmd.lower()

    if cmd_lower in ("/help", "help"):
        show_help()
        return None

    if cmd_lower == "/list":
        list_sessions()
        return None

    if cmd_lower == "/model":
        console.print(f"[cyan]Current model:[/cyan] {MODEL}")
        return None

    if cmd_lower == "/info":
        console.print("[yellow]/info is only available in chat mode[/yellow]")
        return None

    if cmd_lower == "/clear":
        console.print("[yellow]/clear is only available in chat mode[/yellow]")
        return None

    if cmd.startswith("/load "):
        session_input = cmd[5:].strip()
        test_path = SESSIONS_DIR / f"{session_input}.json"
        if not test_path.exists():
            console.print(f"[yellow]Session '{session_input}' not found[/yellow]")
            return None
        return session_input

    if cmd_lower == "/quit":
        console.print("[yellow]Goodbye![/yellow]")
        return "quit"

    if cmd.startswith("/"):
        console.print(f"[red]Unknown command: {cmd}[/red]")

    return None


def handle_chat_command(user_input: str, session: ConversationSession) -> bool:
    """Handle commands in chat mode. Returns True to exit, False to continue."""
    if user_input == "/quit":
        console.print("[yellow]Exiting...Goodbye![/yellow]")
        session.save()
        return True

    if user_input == "/help":
        show_help()
        return False

    if user_input == "/save":
        session.save()
        return False

    if user_input.startswith("/load "):
        session_name = user_input[6:].strip()
        test_path = SESSIONS_DIR / f"{session_name}.json"
        if not test_path.exists():
            console.print(f"[yellow]Session '{session_name}' not found[/yellow]")
            return False
        session.save()
        session.__init__(session_name)
        print_session_history(session)
        return False

    if user_input == "/clear":
        session.clear()
        return False

    if user_input == "/model":
        console.print(f"[cyan]Current model:[/cyan] [yellow]{MODEL}[/yellow]")
        return False

    if user_input == "/info":
        session.show_summary()
        return False

    if user_input == "/list":
        list_sessions()
        return False

    if user_input.startswith("/"):
        console.print(f"[red]Unknown command: {user_input}[/red]")
        return False

    return None


def show_help() -> None:
    """Display interactive mode help."""
    help_text = """
[bold cyan]━━━ Interactive Chat Commands ━━━[/bold cyan]

[bold]Commands:[/bold]
  [green]/help[/green]     — Show this help message
  [green]/save[/green]     — Save current session
  [green]/load NAME[/green] — Load a session
  [green]/clear[/green]    — Clear conversation history
  [green]/model[/green]    — Show current model
  [green]/info[/green]     — Show session info
  [green]/list[/green]     — List saved sessions
  [green]/quit[/green]     — Exit chat

[bold]Tips:[/bold]
  • Responses are rendered as [bold]Markdown[/bold] for better formatting
  • Context window (~{ctx}K tokens) is monitored automatically
  • Sessions are auto-saved on exit; use [green]/save[/green] to save manually
  • Use [green]/load NAME[/green] to resume past conversations
""".format(ctx=CONTEXT_WINDOW // 1000)
    console.print(help_text)


def list_sessions() -> None:
    """List all saved sessions."""
    sessions = list(SESSIONS_DIR.glob("*.json"))
    if not sessions:
        console.print("[yellow]No saved sessions found[/yellow]")
        return

    console.print("[bold cyan]Saved Sessions:[/bold cyan]")
    for session_file in sorted(sessions, reverse=True):
        try:
            with open(session_file, "r") as f:
                data = json.load(f)
                msg_count = len(data.get("messages", []))
                created = data.get("created_at", "")[:10]
                console.print(
                    f"  [green]✓[/green] {session_file.stem:30} ({msg_count:2} messages, {created})"
                )
        except Exception:
            pass


def main():
    """Interactive chat loop with session management."""
    console.print(
        Panel(
            "[bold cyan]🚀 AI Grid Interactive Chat[/bold cyan]\n"
            f"Model: [yellow]{MODEL}[/yellow]\n\n"
            "[dim]Options:[/dim]\n"
            "  • Press [green]Enter[/green] for a new session\n"
            "  • Or type a [green]session name[/green] to load/resume\n"
            "  • Or type a [green]command[/green] (/help, /list, /model, /quit)",
            title="Welcome",
            border_style="cyan",
        )
    )

    while True:
        session_input = console.input("Session name (or Enter for new): ").strip()

        if not session_input:
            break

        if session_input.startswith("/") or session_input.lower() in (
            "help",
            "list",
            "model",
            "info",
            "clear",
            "quit",
        ):
            result = handle_session_command(session_input)
            if result == "quit":
                return
            if result:
                session_input = result
                break
            continue

        break

    session = ConversationSession(session_input if session_input else None)

    print_session_history(session)

    while True:
        console.print()
        user_input = console.input("You 👤: ").strip()

        if not user_input:
            continue

        if user_input.startswith("/"):
            handled = handle_chat_command(user_input, session)
            if handled is None:
                console.print(f"[red]Unknown command: {user_input}[/red]")
            elif handled:
                break
            continue

        handled = handle_chat_command(user_input, session)
        if handled:
            break

        stream_chat(session, user_input)


if __name__ == "__main__":
    main()
