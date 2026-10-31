import unittest
import threading
from types import SimpleNamespace

from marks_inventory import ArticleOcrError
from test_bot_retry import load_bot_module


class FakeMarksStore:
    def __init__(self):
        self.import_calls = []

    def looks_like_marks_pdf(self, pdf_bytes):
        return True

    def import_pdf(self, chat_id, file_name, pdf_bytes, product_name=None):
        self.import_calls.append((chat_id, file_name, pdf_bytes, product_name))
        return SimpleNamespace(
            document_id=314,
            imported=2,
            total=2,
            available=2,
            duplicate=False,
        )


class FakeSendingBot:
    def __init__(self):
        self.documents = []

    def send_document(self, chat_id, document, caption=None, parse_mode=None):
        self.documents.append((chat_id, document.name, document.read(), caption, parse_mode))


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


class FakePhotoBot:
    def __init__(self, payload):
        self.payload = payload
        self.replies = []

    def get_file(self, file_id):
        return SimpleNamespace(file_path="telegram/photo.jpg")

    def download_file(self, file_path):
        return self.payload

    def reply_to(self, message, text, **kwargs):
        self.replies.append((text, kwargs))
        return SimpleNamespace(message_id=1)


class FakeRuntimeBot:
    def __init__(self):
        self.commands = []
        self.polling_kwargs = None

    def set_my_commands(self, commands):
        self.commands = commands

    def infinity_polling(self, **kwargs):
        self.polling_kwargs = kwargs


class BotMarksIntegrationTests(unittest.TestCase):
    def test_help_command_lists_features_and_shows_persistent_button(self):
        bot_module = load_bot_module()
        help_bot = FakePhotoBot(b"")
        bot_module.bot = help_bot
        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/help")

        bot_module.handle_help(message)

        self.assertEqual(len(help_bot.replies), 1)
        help_text, kwargs = help_bot.replies[0]
        self.assertIn("/findmark", help_text)
        self.assertIn("/findarticle", help_text)
        self.assertIn("/markstats", help_text)
        self.assertIn("/draw", help_text)
        self.assertIn("Data Matrix", help_text)
        self.assertIn("CSV", help_text)
        keyboard = kwargs["reply_markup"].keyboard
        button_texts = {button["text"] for row in keyboard for button in row}
        self.assertIn(bot_module.HELP_BUTTON_TEXT, button_texts)

    def test_start_command_attaches_help_button(self):
        bot_module = load_bot_module()
        welcome_bot = FakePhotoBot(b"")
        bot_module.bot = welcome_bot
        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/start")

        bot_module.send_welcome(message)

        self.assertEqual(len(welcome_bot.replies), 1)
        reply_markup = welcome_bot.replies[0][1].get("reply_markup")
        self.assertIsNotNone(reply_markup)
        button_texts = {
            button["text"]
            for row in reply_markup.keyboard
            for button in row
        }
        self.assertIn(bot_module.HELP_BUTTON_TEXT, button_texts)

    def test_run_bot_registers_command_menu_before_polling(self):
        bot_module = load_bot_module()
        runtime_bot = FakeRuntimeBot()
        bot_module.bot = runtime_bot

        bot_module.run_bot()

        command_names = {command.command for command in runtime_bot.commands}
        self.assertTrue(
            {"help", "findmark", "findarticle", "markstats", "draw", "spusk", "reset"}
            <= command_names
        )
        self.assertEqual(
            runtime_bot.polling_kwargs,
            {
                "timeout": bot_module.TELEGRAM_POLLING_TIMEOUT,
                "long_polling_timeout": bot_module.TELEGRAM_LONG_POLLING_TIMEOUT,
            },
        )

    def test_photo_with_article_caption_searches_marks_after_gemini(self):
        bot_module = load_bot_module()
        photo_bot = FakePhotoBot(b"photo-bytes")
        photo_bot.reply_to = lambda msg, text, **kwargs: photo_bot.replies.append(
            (text, kwargs)
        ) or SimpleNamespace(message_id=1)
        photo_bot.edit_message_text = lambda **kwargs: None
        bot_module.bot = photo_bot
        bot_module.marks_store = SimpleNamespace(
            decode_image=lambda payload: []
        )
        session = SimpleNamespace(
            send_message=lambda parts: SimpleNamespace(
                text="Полуботинки GRÜNBERG, артикул 127575/06-03"
            )
        )
        bot_module.get_chat_session = lambda chat_id: session
        bot_module.call_with_503_retry = lambda fn, arg: fn(arg)
        bot_module.update_stats = lambda resp: None
        calls = []
        bot_module.tool_find_marks = lambda query, quantity=1: (
            calls.append((query, quantity))
            or "✅ Найдено и отмечено: 1. Свободно: 9."
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            content_type="photo",
            photo=[SimpleNamespace(file_id="photo-id", file_size=100)],
            caption="",
        )

        bot_module.handle_media(message)

        self.assertIn(("127575/06-03", 1), calls)

    def test_photo_without_article_stays_in_gemini_dialog(self):
        bot_module = load_bot_module()
        photo_bot = FakePhotoBot(b"photo-bytes")
        photo_bot.reply_to = lambda msg, text, **kwargs: photo_bot.replies.append(
            (text, kwargs)
        ) or SimpleNamespace(message_id=1)
        photo_bot.edit_message_text = lambda **kwargs: None
        bot_module.bot = photo_bot
        bot_module.marks_store = SimpleNamespace(
            decode_image=lambda payload: []
        )
        session = SimpleNamespace(
            send_message=lambda parts: SimpleNamespace(
                text="Красивый закат над морем"
            )
        )
        bot_module.get_chat_session = lambda chat_id: session
        bot_module.call_with_503_retry = lambda fn, arg: fn(arg)
        bot_module.update_stats = lambda resp: None
        bot_module.tool_find_marks = lambda query, quantity=1: self.fail(
            "Фото без артикула не должно запускать поиск марок"
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            content_type="photo",
            photo=[SimpleNamespace(file_id="photo-id", file_size=100)],
            caption="",
        )

        bot_module.handle_media(message)

        self.assertEqual(len(photo_bot.replies), 1)

    def test_photo_with_one_datamatrix_claims_matching_mark_without_gemini(self):
        bot_module = load_bot_module()
        photo_bot = FakePhotoBot(b"photo-bytes")
        bot_module.bot = photo_bot
        bot_module.marks_store = SimpleNamespace(
            decode_image=lambda payload: ["PHOTO-MARK-001"]
        )
        claims = []
        bot_module.tool_find_marks = lambda query, quantity=1: (
            claims.append((query, quantity))
            or "✅ Найдено и отмечено: 1. Свободно: 9."
        )
        bot_module.get_chat_session = lambda chat_id: self.fail(
            "Фото с Data Matrix не должно отправляться в Gemini"
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            content_type="photo",
            photo=[SimpleNamespace(file_id="photo-id", file_size=100)],
        )

        bot_module.handle_media(message)

        self.assertEqual(claims, [("PHOTO-MARK-001", 1)])
        self.assertEqual(bot_module.active_chat_id, 77)

    def test_photo_with_multiple_datamatrix_codes_does_not_consume_marks(self):
        bot_module = load_bot_module()
        photo_bot = FakePhotoBot(b"photo-bytes")
        bot_module.bot = photo_bot
        bot_module.marks_store = SimpleNamespace(
            decode_image=lambda payload: ["PHOTO-MARK-001", "PHOTO-MARK-002"]
        )
        bot_module.tool_find_marks = lambda *args, **kwargs: self.fail(
            "При нескольких кодах нельзя автоматически выбирать марку"
        )
        bot_module.get_chat_session = lambda chat_id: self.fail(
            "Фото с несколькими Data Matrix не должно отправляться в Gemini"
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            content_type="photo",
            photo=[SimpleNamespace(file_id="photo-id", file_size=100)],
        )

        bot_module.handle_media(message)

        self.assertEqual(len(photo_bot.replies), 1)
        self.assertIn("несколько", photo_bot.replies[0][0].casefold())
        self.assertIn("один", photo_bot.replies[0][0].casefold())

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
                locations=(),
            )
        )

        response = bot_module.tool_find_marks("MARK-001", quantity=1)

        self.assertIn("1", response)
        self.assertIn("9", response)
        self.assertEqual(
            sending_bot.documents,
            [(77, "ready_marks_marked.pdf", b"%PDF-marked", "✅ Найдено и отмечено: 1. Свободно: 9.", None)],
        )

    def test_tool_find_marks_caption_lists_article_name_sheet_row_column(self):
        bot_module = load_bot_module()
        sending_bot = FakeSendingBot()
        bot_module.bot = sending_bot
        bot_module.active_chat_id = 77
        bot_module.marks_store = SimpleNamespace(
            grid_columns=8,
            claim=lambda chat_id, query, quantity: SimpleNamespace(
                matched=1,
                available=9,
                pdf_bytes=b"%PDF-marked",
                output_name="ready_marks_marked.pdf",
                locations=(
                    SimpleNamespace(
                        article="COVANI SMS25-BHLM3-015H",
                        page_number=1,
                        position_on_page=21,
                        display_name="Ботинки COVANI SMS25",
                    ),
                ),
            ),
        )

        response = bot_module.tool_find_marks("MARK-001", quantity=1)

        self.assertIn("✅ Найдено и отмечено: 1. Свободно: 9.", response)
        self.assertIn(
            "<code>COVANI SMS25-BHLM3-015H</code> │ Ботинки COVANI SMS25 │ <u>л.1 №21 (р.3, к.5)</u>",
            response,
        )
        self.assertEqual(len(sending_bot.documents), 1)
        self.assertIn("№21", sending_bot.documents[0][3])
        self.assertEqual(sending_bot.documents[0][4], "HTML")

    def test_tool_find_marks_without_locations_keeps_old_caption(self):
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

        self.assertEqual(response, "✅ Найдено и отмечено: 1. Свободно: 9.")

    def test_tool_find_marks_routes_covani_article_to_article_search(self):
        bot_module = load_bot_module()
        bot_module.active_chat_id = 77
        calls = []
        bot_module.tool_find_article = lambda article, quantity=1: (
            calls.append((article, quantity))
            or "✅ Ваша марка найдена, её порядковый номер: 2."
        )
        bot_module.marks_store = SimpleNamespace(
            claim=lambda *args, **kwargs: self.fail(
                "Артикул COVANI не должен идти по обычному поиску кода"
            )
        )

        response = bot_module.tool_find_marks("COVANI SMS25-BHLM3-015H")

        self.assertTrue(response.startswith("✅"))
        self.assertEqual(calls, [("COVANI SMS25-BHLM3-015H", 1)])

    def test_tool_find_article_reports_position_on_sheet(self):
        bot_module = load_bot_module()
        sending_bot = FakeSendingBot()
        bot_module.bot = sending_bot
        bot_module.active_chat_id = 77
        bot_module.marks_store = SimpleNamespace(
            claim_current_by_article=lambda chat_id, article, quantity: SimpleNamespace(
                matched=1,
                available=9,
                pdf_bytes=b"%PDF-marked",
                output_name="ready_marks_marked.pdf",
                locations=(
                    SimpleNamespace(
                        article="COVANI SMS25-BHLM3-015H",
                        page_number=2,
                        position_on_page=17,
                    ),
                ),
            ),
            find_available_by_article=lambda chat_id, article: [],
        )

        response = bot_module.tool_find_article("COVANI SMS25-BHLM3")

        self.assertIn("<b>Марка найдена и отмечена!</b>", response)
        self.assertIn("COVANI SMS25-BHLM3-015H", response)
        self.assertIn("<b>Лист:</b> 2", response)
        self.assertIn("№ на листе:</b> 17", response)
        self.assertIn("9", response)
        self.assertEqual(len(sending_bot.documents), 1)
        self.assertEqual(sending_bot.documents[0][0], 77)
        self.assertEqual(sending_bot.documents[0][1], "ready_marks_marked.pdf")
        self.assertEqual(sending_bot.documents[0][4], "HTML")

    def test_tool_find_article_reports_retryable_ocr_error_without_sending_pdf(self):
        bot_module = load_bot_module()
        sending_bot = FakeSendingBot()
        bot_module.bot = sending_bot
        bot_module.active_chat_id = 77

        def fail_ocr(chat_id, article, quantity):
            raise ArticleOcrError("Tesseract internal details")

        bot_module.marks_store = SimpleNamespace(
            claim_current_by_article=fail_ocr,
            find_available_by_article=lambda chat_id, article: [],
        )

        response = bot_module.tool_find_article("COVANI SMS25-BHLM3")

        self.assertIn("распознать", response.casefold())
        self.assertIn("повтори", response.casefold())
        self.assertNotIn("internal", response.casefold())
        self.assertEqual(sending_bot.documents, [])

    def test_tool_find_article_limits_search_to_current_document(self):
        bot_module = load_bot_module()
        bot_module.active_chat_id = 77
        calls = []

        def claim_current_by_article(chat_id, article, quantity):
            calls.append((chat_id, article, quantity))
            return SimpleNamespace(matched=0, available=4)

        bot_module.marks_store = SimpleNamespace(
            claim_current_by_article=claim_current_by_article,
            selected_marks_by_query=lambda chat_id, query: [],
            find_available_by_article=lambda chat_id, article: [],
        )

        bot_module.tool_find_article("COVANI SMS25-BHLM3")

        self.assertEqual(calls, [(77, "COVANI SMS25-BHLM3", 1)])

    def test_tool_find_article_releases_claim_when_document_send_fails(self):
        bot_module = load_bot_module()

        class FailingSendingBot:
            def send_document(self, chat_id, document, caption=None, parse_mode=None):
                raise RuntimeError("telegram upload failed")

        releases = []
        confirmations = []
        bot_module.bot = FailingSendingBot()
        bot_module.active_chat_id = 77
        bot_module.marks_store = SimpleNamespace(
            claim_current_by_article=lambda chat_id, article, quantity: SimpleNamespace(
                matched=1,
                available=9,
                pdf_bytes=b"%PDF-marked",
                output_name="ready_marks_marked.pdf",
                claim_token="claim-123",
                locations=(
                    SimpleNamespace(
                        article="COVANI SMS25-BHLM3-015H",
                        page_number=1,
                        position_on_page=5,
                    ),
                ),
            ),
            release_claim=lambda chat_id, claim_token: releases.append(
                (chat_id, claim_token)
            ),
            finalize_claim=lambda chat_id, claim_token: confirmations.append(
                (chat_id, claim_token)
            ),
            find_available_by_article=lambda chat_id, article: [],
        )

        with self.assertRaisesRegex(RuntimeError, "telegram upload failed"):
            bot_module.tool_find_article("COVANI SMS25-BHLM3")

        self.assertEqual(releases, [(77, "claim-123")])
        self.assertEqual(confirmations, [])

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
        self.assertIn("tool_find_article", names)
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

    def test_document_handler_imports_csv_search_index_for_current_pdf(self):
        bot_module = load_bot_module()
        document_bot = FakeDocumentBot(b"csv-index-data")
        bot_module.bot = document_bot
        calls = []
        bot_module.marks_store = SimpleNamespace(
            import_search_index_csv=lambda chat_id, file_name, csv_bytes: (
                calls.append((chat_id, file_name, csv_bytes))
                or SimpleNamespace(imported=721, unmatched=0, total_rows=721)
            )
        )
        bot_module.register_marks_document = lambda **kwargs: self.fail(
            "CSV не должен проходить через распознавание PDF"
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            document=SimpleNamespace(
                file_name="marks_index.csv",
                file_size=100,
                file_id="csv-file-id",
            ),
            caption=None,
        )

        bot_module.handle_doc(message)

        self.assertEqual(calls, [(77, "marks_index.csv", b"csv-index-data")])
        self.assertEqual(len(document_bot.replies), 1)
        self.assertIn("721", document_bot.replies[0])
        self.assertIn("индекс", document_bot.replies[0].casefold())

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

    def test_findarticle_command_parses_quantity_and_article(self):
        bot_module = load_bot_module()
        command_bot = FakeDocumentBot(b"")
        bot_module.bot = command_bot
        calls = []
        bot_module.tool_find_article = lambda article, quantity=1: (
            calls.append((article, quantity))
            or "✅ По артикулу найдено и отмечено: 2. Позиции: лист 1, №2; лист 1, №3."
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            text="/findarticle 2 COVANI SMS25-BHLM3",
        )

        bot_module.handle_findarticle(message)

        self.assertEqual(calls, [("COVANI SMS25-BHLM3", 2)])
        self.assertEqual(bot_module.active_chat_id, 77)

    def test_plain_article_text_uses_local_search_without_gemini(self):
        bot_module = load_bot_module()
        command_bot = FakeDocumentBot(b"")
        bot_module.bot = command_bot
        calls = []
        bot_module.tool_find_article = lambda article, quantity=1: (
            calls.append((article, quantity))
            or "✅ Ваша марка найдена, её порядковый номер: 5."
        )
        bot_module.get_chat_session = lambda chat_id: self.fail(
            "Артикул должен обрабатываться локально без Gemini"
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            text="COVANI SMS25-BHLM3",
        )

        bot_module.handle_text(message)

        self.assertEqual(calls, [("COVANI SMS25-BHLM3", 1)])
        self.assertEqual(bot_module.active_chat_id, 77)
        self.assertEqual(command_bot.replies, [])

    def test_documents_command_lists_documents_with_keyboard(self):
        bot_module = load_bot_module()

        class FakeUiBot(FakePhotoBot):
            def __init__(self):
                super().__init__(b"")
                self.messages = []

            def send_message(self, chat_id, text, **kwargs):
                self.replies.append((text, kwargs))
                return SimpleNamespace(message_id=1)

        ui_bot = FakeUiBot()
        bot_module.bot = ui_bot
        bot_module.marks_store = SimpleNamespace(
            list_documents=lambda chat_id: [
                {
                    "id": 5,
                    "file_name": "new.pdf",
                    "product_name": "Босоножки ЭВАЛИ",
                    "created_at": "2026-10-31 10:00:00",
                    "total": 721,
                    "available": 600,
                    "selected": 121,
                    "is_current": True,
                },
                {
                    "id": 3,
                    "file_name": "old.pdf",
                    "product_name": "Сабо COVANI",
                    "created_at": "2026-10-20 10:00:00",
                    "total": 244,
                    "available": 200,
                    "selected": 44,
                    "is_current": False,
                },
            ]
        )
        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/documents")

        bot_module.handle_documents(message)

        self.assertEqual(len(ui_bot.messages), 0)  # send_message записывает в replies
        self.assertEqual(len(ui_bot.replies), 1)
        text, kwargs = ui_bot.replies[0]
        self.assertIn("Босоножки ЭВАЛИ", text)
        self.assertIn("текущий", text)
        self.assertIn("Сабо COVANI", text)
        self.assertEqual(kwargs.get("parse_mode"), "HTML")
        keyboard = kwargs.get("reply_markup")
        self.assertIsNotNone(keyboard)
        callbacks = [
            button.callback_data
            for row in keyboard.keyboard
            for button in row
        ]
        self.assertEqual(callbacks, ["doc:5", "doc:3"])

    def test_usedoc_command_switches_current_document(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        switched = []
        bot_module.marks_store = SimpleNamespace(
            list_documents=lambda chat_id: [
                {"id": 5, "file_name": "new.pdf", "product_name": "ЭВАЛИ",
                 "created_at": "", "total": 1, "available": 1, "selected": 0,
                 "is_current": False},
            ],
            set_current_document=lambda chat_id, doc_id: switched.append(doc_id) or True,
        )
        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/usedoc 1")

        bot_module.handle_usedoc(message)

        self.assertEqual(switched, [5])
        self.assertIn("Текущий документ", reply_bot.replies[0][0])

    def test_mark_selection_callback_claims_specific_mark(self):
        bot_module = load_bot_module()

        class FakeCallbackBot(FakeSendingBot):
            def __init__(self):
                super().__init__()
                self.answered = []
                self.edited = []

            def answer_callback_query(self, call_id, text=None):
                self.answered.append((call_id, text))

            def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None):
                self.edited.append((chat_id, message_id))

        callback_bot = FakeCallbackBot()
        bot_module.bot = callback_bot
        bot_module.active_chat_id = None
        bot_module.pending_mark_selection.clear()
        bot_module.pending_mark_selection[77] = [101, 102]
        claimed = []
        bot_module.marks_store = SimpleNamespace(
            grid_columns=8,
            claim_specific_marks=lambda chat_id, ids: (
                claimed.append(ids)
                or SimpleNamespace(
                    matched=1,
                    available=9,
                    pdf_bytes=b"%PDF-marked",
                    output_name="ready_marks_marked.pdf",
                    locations=(
                        SimpleNamespace(
                            article="181-7А",
                            page_number=2,
                            position_on_page=5,
                            display_name="Босоножки ЭВАЛИ 181-7А",
                        ),
                    ),
                )
            ),
        )
        call = SimpleNamespace(
            id="cb1",
            data="sel:1:102",
            message=SimpleNamespace(chat=SimpleNamespace(id=77), message_id=55),
        )

        bot_module.handle_mark_selection_callback(call)

        self.assertEqual(claimed, [[102]])
        self.assertNotIn(77, bot_module.pending_mark_selection)
        self.assertEqual(callback_bot.answered, [("cb1", None)])
        self.assertEqual(callback_bot.edited, [(77, 55)])
        self.assertEqual(len(callback_bot.documents), 1)
        self.assertIn("Отмечено марок: 1", callback_bot.documents[0][3])

    def test_findarticle_without_argument_asks_for_article_and_waits(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        bot_module.pending_command_input.clear()

        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/findarticle")
        bot_module.handle_findarticle(message)

        self.assertEqual(bot_module.pending_command_input.get(77), "findarticle")
        self.assertIn("артикул", reply_bot.replies[0][0].lower())
        self.assertEqual(reply_bot.replies[0][1].get("parse_mode"), "HTML")

        # Следующее текстовое сообщение идёт в поиск по артикулу
        calls = []
        bot_module.tool_find_article = lambda article, quantity=1: (
            calls.append((article, quantity)) or "✅ Марка найдена"
        )
        followup = SimpleNamespace(chat=SimpleNamespace(id=77), text="181-7А")
        bot_module.handle_text(followup)

        self.assertEqual(calls, [("181-7А", 1)])
        self.assertNotIn(77, bot_module.pending_command_input)

    def test_findmark_without_argument_asks_for_query_and_cancel_works(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        bot_module.pending_command_input.clear()

        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/findmark")
        bot_module.handle_findmark(message)

        self.assertEqual(bot_module.pending_command_input.get(77), "findmark")
        self.assertIn("название", reply_bot.replies[0][0].lower())

        cancel = SimpleNamespace(chat=SimpleNamespace(id=77), text="отмена")
        bot_module.handle_text(cancel)

        self.assertNotIn(77, bot_module.pending_command_input)
        self.assertIn("отменил", reply_bot.replies[-1][0].lower())

    def test_unmark_without_argument_asks_and_waits(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        bot_module.pending_command_input.clear()

        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/unmark")
        bot_module.handle_unmark(message)

        self.assertEqual(bot_module.pending_command_input.get(77), "unmark")
        self.assertIn("возвращаем", reply_bot.replies[0][0].lower())
        self.assertEqual(reply_bot.replies[0][1].get("parse_mode"), "HTML")

    def test_unmark_with_multiple_matches_shows_keyboard(self):
        bot_module = load_bot_module()

        class FakeUiBot(FakePhotoBot):
            def __init__(self):
                super().__init__(b"")

            def send_message(self, chat_id, text, **kwargs):
                self.replies.append((text, kwargs))
                return SimpleNamespace(message_id=1)

        ui_bot = FakeUiBot()
        bot_module.bot = ui_bot
        bot_module.pending_unmark_selection.clear()
        bot_module.marks_store = SimpleNamespace(
            find_selected_by_article=lambda chat_id, query: [
                SimpleNamespace(id=101, article="181-7А", page_number=2,
                                position_on_page=5, display_name="Босоножки ЭВАЛИ 181-7А р.37",
                                document_id=3),
                SimpleNamespace(id=102, article="181-7А", page_number=2,
                                position_on_page=38, display_name="Босоножки ЭВАЛИ 181-7А р.39",
                                document_id=3),
            ],
            list_documents=lambda chat_id: [
                {"id": 3, "file_name": "OBEDINENO.pdf", "product_name": "OBEDINENO",
                 "created_at": "", "total": 1, "available": 1, "selected": 0,
                 "is_current": True},
            ],
        )
        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/unmark 181-7А")

        bot_module.handle_unmark(message)

        self.assertEqual(bot_module.pending_unmark_selection.get(77), [101, 102])
        text, kwargs = ui_bot.replies[0]
        self.assertIn("найдено", text.lower())
        self.assertIn("181-7А", text)
        keyboard = kwargs.get("reply_markup")
        self.assertIsNotNone(keyboard)
        callbacks = [b.callback_data for row in keyboard.keyboard for b in row]
        self.assertIn("unm:1:101", callbacks)
        self.assertIn("unm:all:0", callbacks)
        self.assertIn("unm:cancel:0", callbacks)

    def test_plain_article_classifier_requires_letters_and_digits(self):
        bot_module = load_bot_module()

        self.assertEqual(
            bot_module.extract_plain_article("COVANI SMS25-BHLM3"),
            "COVANI SMS25-BHLM3",
        )
        self.assertEqual(
            bot_module.extract_plain_article("SMS25-BHLM3"),
            "SMS25-BHLM3",
        )
        self.assertEqual(bot_module.extract_plain_article("HELLO-WORLD"), "")
        self.assertEqual(bot_module.extract_plain_article("123-456"), "")
        self.assertEqual(bot_module.extract_plain_article("181-7А"), "181-7А")
        self.assertEqual(bot_module.extract_plain_article("36-37"), "")

    def test_gemini_persona_keeps_dry_humor_style(self):
        bot_module = load_bot_module()
        created_configs = []
        bot_module.ai_client = SimpleNamespace(
            chats=SimpleNamespace(
                create=lambda model, config: (
                    created_configs.append(config) or SimpleNamespace()
                )
            )
        )

        bot_module.get_chat_session(4242)

        self.assertEqual(len(created_configs), 1)
        persona = created_configs[0].system_instruction
        self.assertIn("Сеня", persona)
        self.assertIn("шут", persona.lower())
        self.assertIn("Камеди", persona)

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

    def test_voice_with_article_routes_to_article_search(self):
        bot_module = load_bot_module()
        voice_bot = FakePhotoBot(b"voice-ogg-bytes")
        bot_module.bot = voice_bot
        bot_module.transcribe_voice = lambda data: "COVANI SMS25-BHLM3"
        calls = []
        bot_module.tool_find_article = lambda article, quantity=1: (
            calls.append((article, quantity))
            or "✅ Ваша марка найдена, её порядковый номер: 5."
        )
        bot_module.get_chat_session = lambda chat_id: self.fail(
            "Голос с артикулом должен идти в локальный поиск без Gemini-диалога"
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            content_type="voice",
            voice=SimpleNamespace(file_id="voice-id", file_size=100),
        )

        bot_module.handle_voice(message)

        self.assertEqual(calls, [("COVANI SMS25-BHLM3", 1)])
        self.assertEqual(bot_module.active_chat_id, 77)

    def test_voice_with_product_name_routes_to_find_marks(self):
        bot_module = load_bot_module()
        voice_bot = FakePhotoBot(b"voice-ogg-bytes")
        bot_module.bot = voice_bot
        bot_module.transcribe_voice = lambda data: "Босоножки COVANI DLT серые"
        calls = []
        bot_module.tool_find_marks = lambda query, quantity=1: (
            calls.append((query, quantity))
            or "✅ Найдено и отмечено: 1. Свободно: 719."
        )
        bot_module.get_chat_session = lambda chat_id: self.fail(
            "Голос с названием должен идти в локальный поиск без Gemini-диалога"
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            content_type="voice",
            voice=SimpleNamespace(file_id="voice-id", file_size=100),
        )

        bot_module.handle_voice(message)

        self.assertEqual(calls, [("Босоножки COVANI DLT серые", 1)])

    def test_voice_fuzzy_match_offers_variants_without_selecting(self):
        bot_module = load_bot_module()
        voice_bot = FakePhotoBot(b"voice-ogg-bytes")
        bot_module.bot = voice_bot
        bot_module.transcribe_voice = lambda data: "Босоножки COVANI DLT серые"
        calls = []

        def fake_find_marks(query, quantity=1):
            calls.append((query, quantity))
            return f"Свободные марки с названием '{query}' не найдены."

        bot_module.tool_find_marks = fake_find_marks
        bot_module.marks_store = SimpleNamespace(
            suggest_similar=lambda chat_id, query, limit=5: [
                "Босоножки COVANI DLT- S26-WLM3-BM002-2 коричневый (41)",
                "Босоножки COVANI DLT- S26-WLM3-BM003-2 бежевый кожа (36)",
            ]
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            content_type="voice",
            voice=SimpleNamespace(file_id="voice-id", file_size=100),
        )

        bot_module.handle_voice(message)

        self.assertEqual(calls, [("Босоножки COVANI DLT серые", 1)])
        self.assertEqual(len(voice_bot.replies), 1)
        reply = voice_bot.replies[0][0]
        self.assertIn("не наш", reply.lower())
        self.assertIn("коричневый", reply)
        self.assertIn("бежевый", reply)
        self.assertIn("уточни", reply.lower())
        self.assertIn(77, bot_module.pending_suggestions)

    def test_cancel_text_clears_pending_suggestion(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        bot_module.pending_suggestions[77] = ["Вариант А", "Вариант Б"]
        bot_module.get_chat_session = lambda chat_id: self.fail(
            "Отмена не должна уходить в Gemini-диалог"
        )
        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="отмена")

        bot_module.handle_text(message)

        self.assertNotIn(77, bot_module.pending_suggestions)
        self.assertEqual(len(reply_bot.replies), 1)
        self.assertIn("отменил", reply_bot.replies[0][0].lower())

    def test_cancel_text_without_pending_reports_nothing_to_cancel(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        bot_module.pending_suggestions.pop(77, None)
        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="Отмена")

        bot_module.handle_text(message)

        self.assertEqual(len(reply_bot.replies), 1)
        self.assertIn("нечего отменять", reply_bot.replies[0][0].lower())

    def test_successful_mark_search_clears_pending_suggestion(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        bot_module.pending_suggestions[77] = ["Вариант А"]
        bot_module.tool_find_article = lambda article, quantity=1: (
            "✅ Ваша марка найдена, её порядковый номер: 5."
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77), text="COVANI SMS25-BHLM3"
        )

        bot_module.handle_text(message)

        self.assertNotIn(77, bot_module.pending_suggestions)

    def test_voice_chat_falls_back_to_gemini_dialog(self):
        bot_module = load_bot_module()
        voice_bot = FakePhotoBot(b"voice-ogg-bytes")
        bot_module.bot = voice_bot
        bot_module.transcribe_voice = lambda data: "расскажи анекдот про печать"
        sent_parts = []
        session = SimpleNamespace(
            send_message=lambda parts: (
                sent_parts.append(parts)
                or SimpleNamespace(text="Держи: принтер и сканер поспорили...")
            )
        )
        bot_module.get_chat_session = lambda chat_id: session
        bot_module.call_with_503_retry = lambda fn, arg: fn(arg)
        bot_module.update_stats = lambda resp: None
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            content_type="voice",
            voice=SimpleNamespace(file_id="voice-id", file_size=100),
        )

        bot_module.handle_voice(message)

        self.assertEqual(len(sent_parts), 1)
        self.assertEqual(sent_parts[0], "расскажи анекдот про печать")

    def test_voice_reports_empty_transcription(self):
        bot_module = load_bot_module()
        voice_bot = FakePhotoBot(b"voice-ogg-bytes")
        bot_module.bot = voice_bot
        bot_module.transcribe_voice = lambda data: ""
        bot_module.get_chat_session = lambda chat_id: self.fail(
            "Пустая транскрипция не должна уходить в Gemini"
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            content_type="voice",
            voice=SimpleNamespace(file_id="voice-id", file_size=100),
        )

        bot_module.handle_voice(message)

        self.assertEqual(len(voice_bot.replies), 1)
        self.assertIn("не расслышал", voice_bot.replies[0][0].lower())

    def test_unmark_command_releases_selected_marks_and_replies(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        calls = []
        bot_module.marks_store = SimpleNamespace(
            unselect_marks=lambda chat_id, query, quantity=1: (
                calls.append((chat_id, query, quantity)) or 1
            ),
            stats=lambda chat_id: SimpleNamespace(available=10),
            find_selected_by_article=lambda chat_id, query: [],
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            text="/unmark COVANI SMS25-BHLM3-015H",
        )

        bot_module.handle_unmark(message)

        self.assertEqual(calls, [(77, "COVANI SMS25-BHLM3-015H", 1)])
        self.assertEqual(len(reply_bot.replies), 1)
        self.assertIn("вернул", reply_bot.replies[0][0].lower())

    def test_unmark_command_parses_quantity(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        calls = []
        bot_module.marks_store = SimpleNamespace(
            unselect_marks=lambda chat_id, query, quantity=1: (
                calls.append((chat_id, query, quantity)) or quantity
            ),
            stats=lambda chat_id: SimpleNamespace(available=10),
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            text="/unmark 3 Босоножки COVANI",
        )

        bot_module.handle_unmark(message)

        self.assertEqual(calls, [(77, "Босоножки COVANI", 3)])

    def test_unmark_command_reports_when_nothing_released(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        bot_module.marks_store = SimpleNamespace(
            unselect_marks=lambda chat_id, query, quantity=1: 0,
            find_selected_by_article=lambda chat_id, query: [],
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            text="/unmark нет такой марки",
        )

        bot_module.handle_unmark(message)

        self.assertEqual(len(reply_bot.replies), 1)
        self.assertIn("не наш", reply_bot.replies[0][0].lower())

    def test_export_command_shows_format_keyboard(self):
        bot_module = load_bot_module()

        class FakeUiBot(FakePhotoBot):
            def __init__(self):
                super().__init__(b"")

            def send_message(self, chat_id, text, **kwargs):
                self.replies.append((text, kwargs))
                return SimpleNamespace(message_id=1)

        ui_bot = FakeUiBot()
        bot_module.bot = ui_bot
        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/export")

        bot_module.handle_export(message)

        self.assertEqual(len(ui_bot.replies), 1)
        text, kwargs = ui_bot.replies[0]
        self.assertIn("Выгрузка", text)
        self.assertEqual(kwargs.get("parse_mode"), "HTML")
        keyboard = kwargs.get("reply_markup")
        self.assertIsNotNone(keyboard)
        callbacks = [b.callback_data for row in keyboard.keyboard for b in row]
        self.assertIn("exp:;", callbacks)
        self.assertIn("exp:,", callbacks)

    def test_export_callback_sends_csv_with_chosen_delimiter(self):
        bot_module = load_bot_module()

        class FakeExportBot(FakeSendingBot):
            def __init__(self):
                super().__init__()
                self.answered = []
                self.edited = []
                self.sent_messages = []

            def answer_callback_query(self, call_id, text=None):
                self.answered.append(call_id)

            def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None):
                self.edited.append(message_id)

            def send_message(self, chat_id, text, **kwargs):
                pass

        export_bot = FakeExportBot()
        bot_module.bot = export_bot
        delimiters = []
        bot_module.marks_store = SimpleNamespace(
            export_selected_marks=lambda chat_id, delimiter=";": (
                delimiters.append(delimiter)
                or "Артикул,Название\nA,B\n".encode("utf-8-sig")
            )
        )
        call = SimpleNamespace(
            id="cb1",
            data="exp:,",
            message=SimpleNamespace(chat=SimpleNamespace(id=77), message_id=55),
        )

        bot_module.handle_export_callback(call)

        self.assertEqual(delimiters, [","])
        self.assertEqual(len(export_bot.documents), 1)
        chat_id, name, payload, caption, parse_mode = export_bot.documents[0]
        self.assertEqual(chat_id, 77)
        self.assertTrue(name.endswith(".csv"))
        self.assertTrue(payload.startswith(b"\xef\xbb\xbf"))
        self.assertIn("(,)", caption)

    def test_export_callback_reports_empty_selection(self):
        bot_module = load_bot_module()

        class FakeExportBot(FakeSendingBot):
            def __init__(self):
                super().__init__()
                self.sent_messages = []

            def answer_callback_query(self, call_id, text=None):
                pass

            def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None):
                pass

            def send_message(self, chat_id, text, **kwargs):
                self.sent_messages.append(text)

        export_bot = FakeExportBot()
        bot_module.bot = export_bot
        bot_module.marks_store = SimpleNamespace(
            export_selected_marks=lambda chat_id, delimiter=";": None
        )
        call = SimpleNamespace(
            id="cb2",
            data="exp:;",
            message=SimpleNamespace(chat=SimpleNamespace(id=77), message_id=55),
        )

        bot_module.handle_export_callback(call)

        self.assertEqual(len(export_bot.documents), 0)
        self.assertIn("нет выбранных", export_bot.sent_messages[0].lower())

    def test_resetmarks_command_asks_for_confirmation(self):
        bot_module = load_bot_module()

        class FakeUiBot(FakePhotoBot):
            def __init__(self):
                super().__init__(b"")

            def send_message(self, chat_id, text, **kwargs):
                self.replies.append((text, kwargs))
                return SimpleNamespace(message_id=1)

        ui_bot = FakeUiBot()
        bot_module.bot = ui_bot
        calls = []
        bot_module.marks_store = SimpleNamespace(
            reset_inventory=lambda chat_id: (calls.append(chat_id) or 5),
            stats=lambda chat_id: SimpleNamespace(
                total=10, available=5, selected=5
            ),
        )
        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/resetmarks")

        bot_module.handle_resetmarks(message)

        self.assertEqual(calls, [])  # сброс НЕ вызван
        self.assertEqual(len(ui_bot.replies), 1)
        text, kwargs = ui_bot.replies[0]
        self.assertIn("Сбросить ВСЕ", text)
        self.assertIn("5", text)
        self.assertEqual(kwargs.get("parse_mode"), "HTML")
        keyboard = kwargs.get("reply_markup")
        self.assertIsNotNone(keyboard)
        callbacks = [b.callback_data for row in keyboard.keyboard for b in row]
        self.assertIn("rst:yes", callbacks)
        self.assertIn("rst:no", callbacks)

    def test_resetmarks_callback_confirms_and_resets(self):
        bot_module = load_bot_module()

        class FakeResetBot(FakePhotoBot):
            def __init__(self):
                super().__init__(b"")

            def answer_callback_query(self, call_id, text=None):
                pass

            def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None):
                pass

            def send_message(self, chat_id, text, **kwargs):
                self.replies.append((text, kwargs))
                return SimpleNamespace(message_id=1)

        reset_bot = FakeResetBot()
        bot_module.bot = reset_bot
        calls = []
        bot_module.marks_store = SimpleNamespace(
            reset_inventory=lambda chat_id: (calls.append(chat_id) or 5),
            stats=lambda chat_id: SimpleNamespace(
                total=10, available=10, selected=0
            ),
        )
        call = SimpleNamespace(
            id="cb1",
            data="rst:yes",
            message=SimpleNamespace(chat=SimpleNamespace(id=77), message_id=55),
        )

        bot_module.handle_resetmarks_callback(call)

        self.assertEqual(calls, [77])
        self.assertIn("Сбросил", reset_bot.replies[0][0])
        self.assertIn("5", reset_bot.replies[0][0])

    def test_resetmarks_callback_cancel_keeps_selection(self):
        bot_module = load_bot_module()

        class FakeResetBot(FakePhotoBot):
            def __init__(self):
                super().__init__(b"")

            def answer_callback_query(self, call_id, text=None):
                pass

            def edit_message_reply_markup(self, chat_id, message_id, reply_markup=None):
                pass

            def send_message(self, chat_id, text, **kwargs):
                self.replies.append((text, kwargs))
                return SimpleNamespace(message_id=1)

        reset_bot = FakeResetBot()
        bot_module.bot = reset_bot
        calls = []
        bot_module.marks_store = SimpleNamespace(
            reset_inventory=lambda chat_id: calls.append(chat_id),
        )
        call = SimpleNamespace(
            id="cb2",
            data="rst:no",
            message=SimpleNamespace(chat=SimpleNamespace(id=77), message_id=55),
        )

        bot_module.handle_resetmarks_callback(call)

        self.assertEqual(calls, [])  # сброс НЕ вызван
        self.assertIn("Отменил", reset_bot.replies[0][0])

    def test_resetmarks_command_reports_empty_selection(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        calls = []
        bot_module.marks_store = SimpleNamespace(
            reset_inventory=lambda chat_id: calls.append(chat_id),
            stats=lambda chat_id: SimpleNamespace(
                total=10, available=10, selected=0
            ),
        )
        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/resetmarks")

        bot_module.handle_resetmarks(message)

        self.assertEqual(calls, [])
        self.assertEqual(len(reply_bot.replies), 1)
        self.assertIn("нет выбранных", reply_bot.replies[0][0].lower())

    def test_list_selected_marks_returns_table_text(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        bot_module.marks_store = SimpleNamespace(
            export_selected_marks=lambda chat_id: (
                "Артикул;Название;Лист;Позиция на листе;Выбрана;Короткий код\n"
                "A1;Ботинки;1;1;2026-10-27 10:00:00;CODE1\n"
                "A2;Туфли;2;5;2026-10-27 11:00:00;CODE2\n"
            ).encode("utf-8-sig")
        )
        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/list")

        bot_module.handle_list(message)

        self.assertEqual(len(reply_bot.replies), 1)
        reply = reply_bot.replies[0][0]
        self.assertIn("A1", reply)
        self.assertIn("A2", reply)
        self.assertIn("Ботинки", reply)
        self.assertIn("Туфли", reply)
        self.assertIn("1", reply)
        self.assertIn("2", reply)

    def test_list_selected_marks_reports_empty(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        bot_module.marks_store = SimpleNamespace(
            export_selected_marks=lambda chat_id: None
        )
        message = SimpleNamespace(chat=SimpleNamespace(id=77), text="/list")

        bot_module.handle_list(message)

        self.assertEqual(len(reply_bot.replies), 1)
        self.assertIn("нет выбранных", reply_bot.replies[0][0].lower())

    def test_findarticle_reports_already_selected_mark(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        bot_module.marks_store = SimpleNamespace(
            claim_current_by_article=lambda chat_id, article, quantity=1: SimpleNamespace(
                matched=0, available=718, locations=(),
            ),
            selected_marks_by_query=lambda chat_id, query: [
                SimpleNamespace(
                    article="COVANI DLT-S26-WLM3-BM002",
                    page_number=1,
                    position_on_page=49,
                    selected_at="2026-10-27 15:43:28",
                )
            ],
            find_available_by_article=lambda chat_id, article: [],
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            text="/findarticle COVANI DLT-S26-WLM3-BM002",
        )

        bot_module.handle_findarticle(message)

        self.assertEqual(len(reply_bot.replies), 1)
        reply = reply_bot.replies[0][0]
        self.assertIn("уже выбрана", reply.lower())
        self.assertIn("27.10.2026 18:43 МСК", reply)
        self.assertIn("49", reply)
        self.assertIn("1", reply)
        self.assertEqual(reply_bot.replies[0][1].get("parse_mode"), "HTML")

    def test_findarticle_reports_not_found_when_no_selected_match(self):
        bot_module = load_bot_module()
        reply_bot = FakePhotoBot(b"")
        bot_module.bot = reply_bot
        bot_module.marks_store = SimpleNamespace(
            claim_current_by_article=lambda chat_id, article, quantity=1: SimpleNamespace(
                matched=0, available=718, locations=(),
            ),
            selected_marks_by_query=lambda chat_id, query: [],
            find_available_by_article=lambda chat_id, article: [],
        )
        message = SimpleNamespace(
            chat=SimpleNamespace(id=77),
            text="/findarticle COVANI DLT-S26-WLM3-BM999",
        )

        bot_module.handle_findarticle(message)

        self.assertEqual(len(reply_bot.replies), 1)
        reply = reply_bot.replies[0][0]
        self.assertIn("не найдены", reply.lower())
        self.assertNotIn("уже выбрана", reply.lower())


if __name__ == "__main__":
    unittest.main()
