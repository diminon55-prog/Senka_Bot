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

    def callback_query_handler(self, *args, **kwargs):
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

    def test_network_timeout_is_reported_without_internal_exception_details(self):
        bot_module = load_bot_module()

        class ReplyBot:
            def __init__(self):
                self.replies = []

            def reply_to(self, message, text, **kwargs):
                self.replies.append((text, kwargs))

        reply_bot = ReplyBot()
        bot_module.bot = reply_bot
        message = object()
        error = ConnectionError(
            "Connection aborted.",
            TimeoutError("The write operation timed out"),
        )

        with mock.patch.object(bot_module, "LOGGER"):
            bot_module.handle_ai_error(error, chat_id=77, message=message)

        text, _ = reply_bot.replies[0]
        self.assertIn("связ", text.casefold())
        self.assertIn("попроб", text.casefold())
        self.assertNotIn("connection aborted", text.casefold())
        self.assertNotIn("timeouterror", text.casefold())

    def test_generic_error_is_logged_but_not_exposed_to_user(self):
        bot_module = load_bot_module()

        class ReplyBot:
            def __init__(self):
                self.replies = []

            def reply_to(self, message, text, **kwargs):
                self.replies.append((text, kwargs))

        reply_bot = ReplyBot()
        bot_module.bot = reply_bot
        error = RuntimeError("private backend detail")

        with mock.patch.object(bot_module, "LOGGER") as logger:
            bot_module.handle_ai_error(error, chat_id=77, message=object())

        text, _ = reply_bot.replies[0]
        self.assertIn("не получилось", text.casefold())
        self.assertNotIn("private backend detail", text.casefold())
        logger.error.assert_called_once()

    def test_ai_response_retries_as_plain_text_when_markdown_is_invalid(self):
        bot_module = load_bot_module()

        class ParseRejectingBot:
            def __init__(self):
                self.calls = []

            def edit_message_text(self, **kwargs):
                self.calls.append(kwargs)
                if kwargs.get("parse_mode") == "Markdown":
                    raise RuntimeError(
                        "Bad Request: can't parse entities: can't find end of the entity "
                        "starting at byte offset 149"
                    )
                return "sent-as-plain-text"

        fake_bot = ParseRejectingBot()
        bot_module.bot = fake_bot

        result = bot_module.edit_ai_response(
            chat_id=77,
            message_id=123,
            text="Ответ Gemini с незакрытым _Markdown",
        )

        self.assertEqual(result, "sent-as-plain-text")
        self.assertEqual(fake_bot.calls[0]["parse_mode"], "Markdown")
        self.assertNotIn("parse_mode", fake_bot.calls[1])
        self.assertEqual(fake_bot.calls[1]["text"], "Ответ Gemini с незакрытым _Markdown")

    def test_no_ai_response_is_sent_as_unchecked_markdown(self):
        tree = ast.parse(BOT_PATH.read_text(encoding="utf-8"))
        unsafe_lines = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "edit_message_text":
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            text_value = keywords.get("text")
            parse_mode = keywords.get("parse_mode")
            if (
                isinstance(text_value, ast.Attribute)
                and isinstance(text_value.value, ast.Name)
                and text_value.value.id == "resp"
                and text_value.attr == "text"
                and isinstance(parse_mode, ast.Constant)
                and parse_mode.value == "Markdown"
            ):
                unsafe_lines.append(node.lineno)

        self.assertEqual(unsafe_lines, [])

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
        self.assertEqual(wrapped_operations.count("generate_content"), 1)
        self.assertEqual(wrapped_operations.count("send_message"), 5)


if __name__ == "__main__":
    unittest.main()
