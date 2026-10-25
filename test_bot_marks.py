import unittest
import threading
from types import SimpleNamespace

from test_bot_retry import load_bot_module


class FakeMarksStore:
    def __init__(self):
        self.import_calls = []

    def looks_like_marks_pdf(self, pdf_bytes):
        return True

    def import_pdf(self, chat_id, file_name, pdf_bytes, product_name=None):
        self.import_calls.append((chat_id, file_name, pdf_bytes, product_name))
        return SimpleNamespace(imported=2, total=2, available=2, duplicate=False)


class FakeSendingBot:
    def __init__(self):
        self.documents = []

    def send_document(self, chat_id, document, caption=None):
        self.documents.append((chat_id, document.name, document.read(), caption))


class FakeDocumentBot:
    def __init__(self, payload):
        self.payload = payload
        self.replies = []

    def get_file(self, file_id):
        return SimpleNamespace(file_path="telegram/file.pdf")

    def download_file(self, file_path):
        return self.payload

    def reply_to(self, message, text, **kwargs):
        self.replies.append(text)


class BotMarksIntegrationTests(unittest.TestCase):
    def test_register_marks_document_uses_caption_name_and_reports_stock(self):
        bot_module = load_bot_module()
        fake_store = FakeMarksStore()
        bot_module.marks_store = fake_store

        response = bot_module.register_marks_document(
            chat_id=77,
            file_name="ready_marks.pdf",
            pdf_bytes=b"pdf-data",
            caption="Название: Зелёные футболки",
        )

        self.assertIn("2", response)
        self.assertIn("свобод", response.casefold())
        self.assertEqual(
            fake_store.import_calls,
            [(77, "ready_marks.pdf", b"pdf-data", "Зелёные футболки")],
        )

    def test_tool_find_marks_sends_marked_pdf_and_reports_remaining_stock(self):
        bot_module = load_bot_module()
        sending_bot = FakeSendingBot()
        bot_module.bot = sending_bot
        bot_module.active_chat_id = 77
        bot_module.marks_store = SimpleNamespace(
            claim=lambda chat_id, query, quantity: SimpleNamespace(
                matched=1,
                available=9,
                pdf_bytes=b"%PDF-marked",
                output_name="ready_marks_marked.pdf",
            )
        )

        response = bot_module.tool_find_marks("MARK-001", quantity=1)

        self.assertIn("1", response)
        self.assertIn("9", response)
        self.assertEqual(
            sending_bot.documents,
            [(77, "ready_marks_marked.pdf", b"%PDF-marked", "✅ Найдено и отмечено: 1. Свободно: 9.")],
        )

    def test_tool_marks_stats_reports_total_selected_and_available(self):
        bot_module = load_bot_module()
        bot_module.active_chat_id = 77
        bot_module.marks_store = SimpleNamespace(
            stats=lambda chat_id: SimpleNamespace(total=12, selected=3, available=9, documents=1)
        )

        response = bot_module.tool_marks_stats()

        self.assertIn("12", response)
        self.assertIn("3", response)
        self.assertIn("9", response)

    def test_gemini_session_exposes_marks_tools(self):
        bot_module = load_bot_module()
        captured = {}

        class Chats:
            def create(self, **kwargs):
                captured.update(kwargs)
                return object()

        bot_module.ai_client = SimpleNamespace(chats=Chats())
        bot_module.user_chats.clear()

        bot_module.get_chat_session(77)

        tools = captured["config"].tools
        names = {getattr(tool, "__name__", "") for tool in tools}
        self.assertIn("tool_find_marks", names)
        self.assertIn("tool_marks_stats", names)

    def test_document_handler_registers_marks_pdf_before_standard_layout(self):
        bot_module = load_bot_module()
        document_bot = FakeDocumentBot(b"ready-pdf")
        bot_module.bot = document_bot
        calls = []
        bot_module.register_marks_document = lambda **kwargs: (
            calls.append(kwargs) or "Марок: 2. Свободно: 2."
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            document=SimpleNamespace(
                file_name="ready_marks.pdf",
                file_size=100,
                file_id="file-id",
            ),
            caption="Название: Футболки",
        )

        bot_module.handle_doc(message)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["chat_id"], 77)
        self.assertEqual(calls[0]["pdf_bytes"], b"ready-pdf")
        self.assertEqual(document_bot.replies, ["Марок: 2. Свободно: 2."])

    def test_findmark_command_parses_quantity_and_query(self):
        bot_module = load_bot_module()
        command_bot = FakeDocumentBot(b"")
        bot_module.bot = command_bot
        calls = []
        bot_module.tool_find_marks = lambda query, quantity=1: (
            calls.append((query, quantity)) or "✅ Найдено и отмечено: 2. Свободно: 7."
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            text="/findmark 2 Зелёные футболки",
        )

        bot_module.handle_findmark(message)

        self.assertEqual(calls, [("Зелёные футболки", 2)])
        self.assertEqual(bot_module.active_chat_id, 77)

    def test_active_chat_context_is_isolated_between_worker_threads(self):
        bot_module = load_bot_module()
        barrier = threading.Barrier(2)
        results = {}

        def worker(chat_id):
            bot_module.set_active_chat_id(chat_id)
            barrier.wait()
            results[chat_id] = bot_module.get_active_chat_id()

        first = threading.Thread(target=worker, args=(101,))
        second = threading.Thread(target=worker, args=(202,))
        first.start()
        second.start()
        first.join()
        second.join()

        self.assertEqual(results, {101: 101, 202: 202})


if __name__ == "__main__":
    unittest.main()
