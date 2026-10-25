import ast
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock


PROJECT_DIR = Path(__file__).resolve().parent
BOT_PATH = PROJECT_DIR / "bot.py"


class DummyBot:
    def __init__(self, *args, **kwargs):
        pass

    def message_handler(self, *args, **kwargs):
        return lambda handler: handler

    def infinity_polling(self, *args, **kwargs):
        pass


class DummyClient:
    pass


def load_bot_module():
    import telebot
    from google import genai

    spec = importlib.util.spec_from_file_location("bot_under_test", BOT_PATH)
    module = importlib.util.module_from_spec(spec)
    previous_cwd = os.getcwd()
    try:
        os.chdir(PROJECT_DIR)
        with (
            mock.patch.object(telebot, "TeleBot", DummyBot),
            mock.patch.object(genai, "Client", return_value=DummyClient()),
        ):
            spec.loader.exec_module(module)
    finally:
        os.chdir(previous_cwd)
    return module


class Retry503Tests(unittest.TestCase):
    def test_503_is_retried_until_the_call_succeeds(self):
        bot_module = load_bot_module()
        attempts = 0

        class ServiceUnavailable(Exception):
            code = 503

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ServiceUnavailable("503 Service Unavailable")
            return "ok"

        with mock.patch.object(bot_module.time, "sleep") as sleep:
            result = bot_module.call_with_503_retry(operation)

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 3)
        self.assertEqual(sleep.call_args_list, [mock.call(4), mock.call(4)])

    def test_high_demand_error_without_numeric_code_is_retried(self):
        bot_module = load_bot_module()
        attempts = 0

        def operation():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("The model is overloaded due to high demand")
            return "ok"

        with mock.patch.object(bot_module.time, "sleep") as sleep:
            result = bot_module.call_with_503_retry(operation)

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 2)
        sleep.assert_called_once_with(4)

    def test_every_gemini_request_uses_the_retry_helper(self):
        tree = ast.parse(BOT_PATH.read_text(encoding="utf-8"))
        wrapped_operations = []
        direct_requests = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "call_with_503_retry":
                if node.args and isinstance(node.args[0], ast.Attribute):
                    wrapped_operations.append(node.args[0].attr)
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "generate_images":
                direct_requests.append(node.lineno)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "send_message"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "get_chat_session"
            ):
                direct_requests.append(node.lineno)

        self.assertEqual(direct_requests, [])
        self.assertEqual(wrapped_operations.count("generate_images"), 1)
        self.assertEqual(wrapped_operations.count("send_message"), 4)


if __name__ == "__main__":
    unittest.main()
