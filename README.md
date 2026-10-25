# Memory Chatbot — Custom AI Chatbot with Memory

**DecodeLabs Generative AI Internship — Project 1**

A terminal chatbot built with Python and the Google Gemini API that remembers everything said during a live session.

## The Problem
LLM APIs are **stateless** — each request is treated as a brand-new conversation. Ask "What is my name?" right after telling it, and a plain API call has no idea.

## The Solution
The chatbot keeps an **in-memory history list** and sends the whole conversation with every request:

1. **Validate** — empty or whitespace-only input is blocked before reaching the API (it would return a 400 error).
2. **Append** — the user's message is added to the history as `{"role": "user", "parts": [...]}`.
3. **Send** — the full history is sent to Gemini.
4. **Record** — the model's reply is added as `{"role": "model", "parts": [...]}`.
5. **Trim** — a **sliding window (FIFO)** drops the oldest user/model pair once the limit is reached, so the context never overflows the token budget.

## Features
- Conversation memory within a session
- Input validation guard
- Sliding-window memory (configurable with `MAX_TURNS`)
- Commands: `/history`, `/clear`, `/help`, `/exit`
- Error handling — a failed API call doesn't break the history
- API key kept in `.env`, never committed

## Setup
```bash
git clone https://github.com/Moemenakari/memory-chatbot.git
cd memory-chatbot
pip install -r requirements.txt
```

Create a `.env` file (copy `.env.example`) and add your free API key from [Google AI Studio](https://aistudio.google.com/apikey):
```
GEMINI_API_KEY=your_api_key_here
```

## Run
```bash
python chatbot.py
```

## Memory Test
| Step | Input | Expected |
|------|-------|----------|
| 1. State initialization | `My name is Mouemen` | Bot acknowledges |
| 2. Context distraction | `Write a poem about tech` | Long response |
| 3. State extraction | `What is my name?` | **"Mouemen"** ✅ |

## Unit Tests
The memory logic is tested with a fake client (no API key needed):
```bash
python -m unittest test_chatbot.py
```

## Tech Stack
Python 3 · Google Gemini API (`google-genai`) · python-dotenv

## Author
**Mouemen Akkarie** — [GitHub](https://github.com/Moemenakari)
