"""Tests for the memory logic. They use a fake client, so no API key is needed.

Run with:  python -m unittest test_chatbot.py
"""

import unittest

from chatbot import ChatMemory, ask_model, is_valid


class FakeModels:
    """Pretends to be the Gemini client and answers from the history it receives."""

    def __init__(self):
        self.last_contents = None

    def generate_content(self, model, contents, config=None):
        self.last_contents = list(contents)
        last = contents[-1]["parts"][0]["text"]
        if last == "What is my name?":
            for msg in contents:
                text = msg["parts"][0]["text"]
                if msg["role"] == "user" and text.startswith("My name is "):
                    return type("R", (), {"text": "Your name is " + text[11:]})()
            return type("R", (), {"text": "I don't know your name."})()
        return type("R", (), {"text": "OK: " + last})()


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


class TestMemory(unittest.TestCase):
    def test_validation_blocks_empty_input(self):
        self.assertFalse(is_valid(""))
        self.assertFalse(is_valid("   "))
        self.assertTrue(is_valid("hello"))

    def test_history_is_sent_every_turn(self):
        client, memory = FakeClient(), ChatMemory(max_turns=10)
        ask_model(client, memory, "Hi")
        ask_model(client, memory, "How are you?")
        # 2nd request must contain: user, model, user
        self.assertEqual(len(client.models.last_contents), 3)
        self.assertEqual(len(memory), 4)

    def test_memory_exam(self):
        client, memory = FakeClient(), ChatMemory(max_turns=10)
        ask_model(client, memory, "My name is Mouemen")
        ask_model(client, memory, "Write a poem about tech")
        reply = ask_model(client, memory, "What is my name?")
        self.assertIn("Mouemen", reply)

    def test_sliding_window_drops_oldest(self):
        client, memory = FakeClient(), ChatMemory(max_turns=3)
        for i in range(5):
            ask_model(client, memory, f"message {i}")
        self.assertEqual(len(memory), 6)
        self.assertEqual(memory.history[0]["parts"][0]["text"], "message 2")

    def test_failed_call_keeps_history_clean(self):
        class BrokenModels:
            def generate_content(self, **kwargs):
                raise RuntimeError("network down")

        client = FakeClient()
        client.models = BrokenModels()
        memory = ChatMemory()
        with self.assertRaises(RuntimeError):
            ask_model(client, memory, "hello")
        self.assertEqual(len(memory), 0)


if __name__ == "__main__":
    unittest.main()
