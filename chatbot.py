"""
Memory Chatbot - a terminal chatbot that remembers the conversation.

The Gemini API is stateless: every request is handled on its own.
To give the bot a memory, we keep the whole conversation in a list
and send that list with every new message.
"""

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_TURNS = int(os.getenv("MAX_TURNS", "10"))  # user+model pairs kept in memory

SYSTEM_PROMPT = (
    "You are a friendly and helpful assistant. "
    "Use the earlier messages in this conversation to give relevant answers."
)

HELP_TEXT = """Commands:
  /history  show what the bot currently remembers
  /clear    forget the conversation and start fresh
  /help     show this message
  /exit     quit"""


class ChatMemory:
    """Stores the conversation as a list of role/parts messages."""

    def __init__(self, max_turns=MAX_TURNS):
        self.max_turns = max_turns
        self.history = []

    def add(self, role, text):
        self.history.append({"role": role, "parts": [{"text": text}]})

    def trim(self):
        """Sliding window: drop the oldest user+model pair when we go over the limit."""
        dropped = 0
        while len(self.history) > self.max_turns * 2:
            self.history = self.history[2:]
            dropped += 1
        return dropped

    def clear(self):
        self.history = []

    def __len__(self):
        return len(self.history)


def is_valid(message):
    """Empty or whitespace-only input makes the API return 400, so block it here."""
    return bool(message and message.strip())


def ask_model(client, memory, user_text):
    """Append the user message, send the full history, then save the reply."""
    memory.add("user", user_text)
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=memory.history,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        reply = (response.text or "").strip() or "(no response)"
    except Exception as error:
        # Remove the message we just added so the history stays clean
        memory.history.pop()
        raise error

    memory.add("model", reply)
    memory.trim()
    return reply


def show_history(memory):
    if not memory.history:
        print("  (memory is empty)")
        return
    for i, msg in enumerate(memory.history, start=1):
        text = msg["parts"][0]["text"].replace("\n", " ")
        preview = text if len(text) <= 70 else text[:67] + "..."
        print(f"  {i:>2}. {msg['role']:<5} | {preview}")
    print(f"  -> {len(memory)} messages ({len(memory) // 2}/{memory.max_turns} turns)")


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Missing GEMINI_API_KEY. Copy .env.example to .env and add your key.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    memory = ChatMemory()

    print("=" * 50)
    print(f"  Memory Chatbot  |  model: {MODEL}")
    print(f"  Remembers the last {MAX_TURNS} turns. Type /help for commands.")
    print("=" * 50)

    while True:
        try:
            user_text = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        command = user_text.strip().lower()
        if command in ("/exit", "/quit"):
            print("Goodbye!")
            break
        if command == "/help":
            print(HELP_TEXT)
            continue
        if command == "/history":
            show_history(memory)
            continue
        if command == "/clear":
            memory.clear()
            print("Memory cleared.")
            continue

        if not is_valid(user_text):
            print("Please type a message (empty input is not sent).")
            continue

        try:
            reply = ask_model(client, memory, user_text.strip())
            print(f"\nBot: {reply}")
        except Exception as error:
            print(f"\n[Error] Could not reach the model: {error}")


if __name__ == "__main__":
    main()
