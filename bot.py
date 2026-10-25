import os, re, zipfile, rarfile, io, telebot, time
import pymupdf as fitz
from contextvars import ContextVar
from google import genai
from google.genai import types
import cz_processor
from marks_inventory import MarksInventory
from settings import (
    GEMINI_CHAT_MODEL,
    GEMINI_IMAGE_MODEL,
    MARKS_CELL_HEIGHT_PT,
    MARKS_CELL_WIDTH_PT,
    MARKS_DB_PATH,
    MARKS_GRID_COLUMNS,
    MARKS_GRID_ROWS,
    MARKS_HIGHLIGHT_OPACITY,
    MARKS_MAX_QUANTITY,
    MARKS_SCAN_DPI,
    MARKS_STORAGE_DIR,
    MAX_503_RETRIES,
    MAX_FILE_SIZE,
    RETRY_DELAY_SECONDS,
    TELEGRAM_LONG_POLLING_TIMEOUT,
    TELEGRAM_POLLING_TIMEOUT,
    TELEGRAM_TOKEN,
)

ai_client = genai.Client()

bot = telebot.TeleBot(TELEGRAM_TOKEN)
user_chats = {}
last_user_pdf = {} 
active_chat_id = None
_active_chat_context = ContextVar("active_chat_id", default=None)


def set_active_chat_id(chat_id):
    global active_chat_id
    active_chat_id = chat_id
    _active_chat_context.set(chat_id)


def get_active_chat_id():
    return _active_chat_context.get() or active_chat_id

bot_stats = {"requests": 0, "tokens": 0}

marks_store = MarksInventory(
    db_path=MARKS_DB_PATH,
    storage_dir=MARKS_STORAGE_DIR,
    scan_dpi=MARKS_SCAN_DPI,
    grid_columns=MARKS_GRID_COLUMNS,
    grid_rows=MARKS_GRID_ROWS,
    cell_width_pt=MARKS_CELL_WIDTH_PT,
    cell_height_pt=MARKS_CELL_HEIGHT_PT,
    highlight_opacity=MARKS_HIGHLIGHT_OPACITY,
    max_quantity=MARKS_MAX_QUANTITY,
)

def is_service_unavailable(error):
    response = getattr(error, "response", None)
    status_values = (
        getattr(error, "code", None),
        getattr(error, "status_code", None),
        getattr(error, "status", None),
        getattr(response, "status_code", None),
        getattr(response, "status", None),
    )
    if any(str(value).lower() in {"503", "unavailable", "service_unavailable"} for value in status_values if value is not None):
        return True

    error_text = str(error).lower()
    overload_markers = (
        "503",
        "service unavailable",
        "service_unavailable",
        "high demand",
        "overloaded",
        "server overload",
    )
    return any(marker in error_text for marker in overload_markers)

def call_with_503_retry(operation, *args, **kwargs):
    retries = 0
    while True:
        try:
            return operation(*args, **kwargs)
        except Exception as error:
            if not is_service_unavailable(error) or retries >= MAX_503_RETRIES:
                raise
            retries += 1
            time.sleep(RETRY_DELAY_SECONDS)

def update_stats(resp):
    global bot_stats
    bot_stats["requests"] += 1
    if hasattr(resp, 'usage_metadata') and resp.usage_metadata:
        bot_stats["tokens"] += getattr(resp.usage_metadata, 'total_token_count', 0)

def register_marks_document(chat_id, file_name, pdf_bytes, caption=""):
    if not file_name.lower().endswith(".pdf") or not marks_store.looks_like_marks_pdf(pdf_bytes):
        return None

    name_match = re.search(r"(?:название|товар)\s*[:=\-]\s*(.+)", caption or "", re.IGNORECASE)
    if name_match:
        product_name = name_match.group(1).strip()
    else:
        product_name = os.path.splitext(os.path.basename(file_name))[0]
        product_name = re.sub(r"(?i)(?:_?a3)?_?1?_?soderzhimoe(?:_all)?$", "", product_name)
        product_name = product_name.replace("_", " ").strip() or "Марки"

    result = marks_store.import_pdf(
        chat_id=chat_id,
        file_name=file_name,
        pdf_bytes=pdf_bytes,
        product_name=product_name,
    )
    prefix = "Этот документ уже был в базе." if result.duplicate else "База марок создана."
    return (
        f"✅ {prefix}\n"
        f"Марок: {result.total}. Свободно в наличии: {result.available}.\n"
        "Поиск: /findmark <Data Matrix или название>."
    )

def make_a3_layout(pdf_bytes, base_name="STROBS", cut_w_mm=50.0, cut_h_mm=40.0, gap_x_mm=5.0, gap_y_mm=5.0, inner_pad_mm=4.0, cols=0, rows=0):
    MM = 72.0 / 25.4
    A3_W, A3_H = 420.0 * MM, 297.0 * MM
    CUT_W, CUT_H = cut_w_mm * MM, cut_h_mm * MM
    GAP_X, GAP_Y = gap_x_mm * MM, gap_y_mm * MM
    STEP_W, STEP_H = CUT_W + GAP_X, CUT_H + GAP_Y
    INNER_PAD = inner_pad_mm * MM
    SAFE_W = max(10.0 * MM, CUT_W - 2 * INNER_PAD)
    SAFE_H = max(10.0 * MM, CUT_H - 2 * INNER_PAD)

    max_cols = max(1, int((400.0 + gap_x_mm) // (cut_w_mm + gap_x_mm)))
    max_rows = max(1, int((277.0 + gap_y_mm) // (cut_h_mm + gap_y_mm)))
    COLS = cols if (0 < cols <= max_cols) else max_cols
    ROWS = rows if (0 < rows <= max_rows) else max_rows
    PER_SHEET = COLS * ROWS

    TOTAL_W = COLS * CUT_W + (COLS - 1) * GAP_X
    TOTAL_H = ROWS * CUT_H + (ROWS - 1) * GAP_Y
    MARGIN_X, MARGIN_Y = (A3_W - TOTAL_W) / 2.0, (A3_H - TOTAL_H) / 2.0

    src = fitz.open("pdf", pdf_bytes)
    total_pages = len(src)
    if total_pages == 0:
        raise ValueError("PDF пустой!")

    CX0, CY0, CX1, CY1 = 6.0, 7.49, 157.38, 105.73
    CW, CH = CX1 - CX0, CY1 - CY0
    SCALE = min(SAFE_W / CW, SAFE_H / CH)

    doc_print, doc_cut = fitz.open(), fitz.open()
    num_sheets = (total_pages - 1) // PER_SHEET + 1

    for sheet in range(num_sheets):
        p_print = doc_print.new_page(width=A3_W, height=A3_H)
        p_cut = doc_cut.new_page(width=A3_W, height=A3_H)
        shape_cut = p_cut.new_shape()
        start_idx = sheet * PER_SHEET
        end_idx = min(start_idx + PER_SHEET, total_pages)

        for idx in range(start_idx, end_idx):
            i = idx - start_idx
            col, row = i % COLS, i // COLS
            cx0, cy0 = MARGIN_X + col * STEP_W, MARGIN_Y + row * STEP_H
            shape_cut.draw_rect(fitz.Rect(cx0, cy0, cx0 + CUT_W, cy0 + CUT_H))

            svg_b = src[idx].get_svg_image(text_as_path=True).encode("utf-8")
            svg_d = fitz.open("svg", svg_b)
            c_pdf = fitz.open("pdf", svg_d.convert_to_pdf())

            tx0 = cx0 + (CUT_W - CW * SCALE) / 2.0 - CX0 * SCALE
            ty0 = cy0 + (CUT_H - CH * SCALE) / 2.0 - (113.386 - CY1) * SCALE
            t_rect = fitz.Rect(tx0, ty0, tx0 + 170.079 * SCALE, ty0 + 113.386 * SCALE)
            p_print.show_pdf_page(t_rect, c_pdf, 0)
            c_pdf.close()
            svg_d.close()

        shape_cut.finish(color=(1, 0, 0), fill=None, width=0.25)
        shape_cut.commit()

    pb, cb = doc_print.tobytes(deflate=True), doc_cut.tobytes(deflate=True)
    doc_print.close(); doc_cut.close(); src.close()
    clean = os.path.splitext(base_name)[0]
    info = {"cols": COLS, "rows": ROWS, "per_sheet": PER_SHEET, "sheets": num_sheets, "mx": MARGIN_X / MM, "my": MARGIN_Y / MM}
    return pb, cb, f"{clean}_A3_1_soderzhimoe.pdf", f"{clean}_A3_2_rez.pdf", info

def handle_ai_error(e, chat_id, message, thinking_msg=None):
    err = str(e).lower()
    if "429" in err and "resource_exhausted" in err:
        text = "⏳ **Упс, перегрев!**\nМы уперлись в лимит токенов. Подожди минуту и попробуй снова!"
    else:
        text = f"Ошибка:\n`{err}`"
    try:
        if thinking_msg: bot.edit_message_text(chat_id=chat_id, message_id=thinking_msg.message_id, text=text, parse_mode="Markdown")
        else: bot.reply_to(message, text, parse_mode="Markdown")
    except: pass

def tool_spusk_a3(cut_w_mm: float = 50.0, cut_h_mm: float = 40.0, gap_x_mm: float = 5.0, gap_y_mm: float = 5.0, inner_pad_mm: float = 4.0, cols: int = 0, rows: int = 0) -> str:
    chat_id = get_active_chat_id()
    if not chat_id or chat_id not in last_user_pdf or not last_user_pdf[chat_id]:
        return "ОШИБКА: Нет файлов!"
    fname, pdf_bytes = last_user_pdf[chat_id][0]
    try:
        bot.send_message(chat_id, f"⚙️ Раскладываю '{fname}'...")
        pb, cb, pn, cn, inf = make_a3_layout(pdf_bytes, fname, cut_w_mm, cut_h_mm, gap_x_mm, gap_y_mm, inner_pad_mm, cols, rows)
        f1, f2 = io.BytesIO(pb), io.BytesIO(cb)
        f1.name, f2.name = pn, cn
        bot.send_document(chat_id, f1, caption=f"✅ Печать: {pn}")
        bot.send_document(chat_id, f2, caption=f"✂️ Резка: {cn}")
        return "Готово!"
    except Exception as e:
        return f"Ошибка: {e}"

def tool_process_cz_archive(num_shift_x_mm: float = 0.0, num_shift_y_mm: float = 0.0, gap_x_mm: float = 2.0, gap_y_mm: float = 1.8) -> str:
    """Инструмент для обработки архивов Честного Знака с параметрами сдвига нумерации и зазоров."""
    chat_id = get_active_chat_id()
    if not chat_id or chat_id not in last_user_pdf or not last_user_pdf[chat_id]:
        return "ОШИБКА: Архив с марками не найден!"
    
    extracted_pdfs = last_user_pdf[chat_id]
    bot.send_message(chat_id, f"⚙️ Собираю макет ЧЗ ({len(extracted_pdfs)} файлов).\nЗазоры: X={gap_x_mm}мм, Y={gap_y_mm}мм.\nСдвиг номера: X={num_shift_x_mm}мм, Y={num_shift_y_mm}мм вверх.")
    
    try:
        sod_bytes, rez_bytes, total = cz_processor.process_cz_archive(
            extracted_pdfs, num_shift_x_mm, num_shift_y_mm, gap_x_mm, gap_y_mm
        )
        if sod_bytes:
            f1, f2 = io.BytesIO(sod_bytes), io.BytesIO(rez_bytes)
            f1.name = "STROBS_A3_1_soderzhimoe_ALL.pdf"
            f2.name = "STROBS_A3_2_rez_ALL.pdf"
            bot.send_document(chat_id, f1, caption=f"✅ Готово! Склеил {total} этикеток.")
            bot.send_document(chat_id, f2, caption="✂️ Файл резки CMYK.")
            return f"Успешно склеено {total} марок."
        else:
            return "Ошибка: не удалось распознать этикетки."
    except Exception as e:
        return f"Ошибка обработки ЧЗ: {e}"

def tool_find_marks(query: str, quantity: int = 1) -> str:
    """Находит свободные марки по Data Matrix или названию и помечает их зелёным."""
    chat_id = get_active_chat_id()
    if not chat_id:
        return "ОШИБКА: Не удалось определить чат."

    result = marks_store.claim(chat_id, query, quantity)
    if result.matched == 0:
        return f"Марки по запросу '{query}' не найдены. Свободно: {result.available}."

    marked_pdf = io.BytesIO(result.pdf_bytes)
    marked_pdf.name = result.output_name
    caption = f"✅ Найдено и отмечено: {result.matched}. Свободно: {result.available}."
    bot.send_document(chat_id, marked_pdf, caption=caption)
    return caption


def tool_marks_stats() -> str:
    """Показывает остаток свободных и уже выбранных марок."""
    chat_id = get_active_chat_id()
    if not chat_id:
        return "ОШИБКА: Не удалось определить чат."
    stats = marks_store.stats(chat_id)
    return (
        f"Всего марок: {stats.total}. "
        f"Выбрано: {stats.selected}. "
        f"Свободно в наличии: {stats.available}."
    )


def get_chat_session(chat_id):
    if chat_id not in user_chats:
        persona = (
            "Твое имя Сеня. Ты эксперт-полиграфист. Инструмент `tool_spusk_a3` вызывай для обычных PDF макетов. "
            "Инструмент `tool_process_cz_archive` вызывай, когда пользователь загружает архив (или много PDF файлов) "
            "Честного Знака и просит разложить его, изменить зазоры (gap_x_mm, gap_y_mm) или сдвинуть нумерацию "
            "(num_shift_x_mm, num_shift_y_mm). ВАЖНО: 'сдвинуть нумерацию вверх на 2мм' означает num_shift_y_mm = 2.0. "
            "'сдвинуть вправо' означает num_shift_x_mm > 0. По умолчанию gap_x_mm=2.0, gap_y_mm=1.8. "
            "Если пользователь просит найти, выбрать или отметить марку по названию либо Data Matrix, вызывай "
            "tool_find_marks. Для вопроса об остатках марок вызывай tool_marks_stats."
        )
        user_chats[chat_id] = ai_client.chats.create(
            model=GEMINI_CHAT_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=persona,
                tools=[tool_spusk_a3, tool_process_cz_archive, tool_find_marks, tool_marks_stats],
            )
        )
    return user_chats[chat_id]

# --- ОБРАБОТЧИКИ КОМАНД ---

@bot.message_handler(commands=['start', 'reset'])
def send_welcome(message):
    user_chats.pop(message.chat.id, None)
    bot.reply_to(message, "Контекст сброшен! Кидай архив ЧЗ и пиши, если нужно сдвинуть номера или зазоры.")

@bot.message_handler(commands=['stats'])
def handle_stats(message):
    text = f"📊 Запросов: `{bot_stats['requests']}`\nТокенов: `{bot_stats['tokens']}`"
    bot.reply_to(message, text, parse_mode="Markdown")

@bot.message_handler(commands=['spusk'])
def handle_spusk(message):
    chat_id = message.chat.id
    set_active_chat_id(chat_id)
    if chat_id not in last_user_pdf:
        bot.reply_to(message, "Сначала закинь PDF или архив!")
        return
    tool_spusk_a3()

@bot.message_handler(commands=['findmark', 'марка'])
def handle_findmark(message):
    set_active_chat_id(message.chat.id)
    payload = message.text.split(maxsplit=1)[1].strip() if len(message.text.split(maxsplit=1)) > 1 else ""
    if not payload:
        bot.reply_to(message, "Использование: /findmark [количество] <Data Matrix или название>")
        return

    quantity = 1
    query = payload
    first, separator, remainder = payload.partition(" ")
    if separator and first.isdigit():
        quantity = max(1, min(int(first), MARKS_MAX_QUANTITY))
        query = remainder.strip()
    if not query:
        bot.reply_to(message, "После количества укажи Data Matrix или название.")
        return

    response = tool_find_marks(query, quantity)
    if not response.startswith("✅"):
        bot.reply_to(message, response)


@bot.message_handler(commands=['markstats', 'остаток'])
def handle_markstats(message):
    set_active_chat_id(message.chat.id)
    bot.reply_to(message, tool_marks_stats())


# НОВАЯ КОМАНДА: РИСОВАНИЕ КАРТИНОК (ДОЛЖНА БЫТЬ ВЫШЕ TEXT!)
@bot.message_handler(commands=['draw', 'рисуй'])
def handle_draw(message):
    prompt = message.text.replace('/draw', '').replace('/рисуй', '').strip()
    if not prompt:
        bot.reply_to(message, "Напиши, что нарисовать. Например: `/draw Кот в костюме космонавта`", parse_mode="Markdown")
        return

    thinking_msg = bot.reply_to(message, "<blockquote>🎨 Развожу краски и грунтую холст...</blockquote>", parse_mode="HTML")
    try:
        # Обращаемся к модели генерации картинок Imagen 3
        result = call_with_503_retry(
            ai_client.models.generate_images,
            model=GEMINI_IMAGE_MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio="1:1" # Можно менять на "16:9" или "9:16"
            )
        )

        # Вытаскиваем байты картинки
        image_bytes = result.generated_images[0].image.image_bytes
        photo_stream = io.BytesIO(image_bytes)

        # Отправляем в чат
        bot.send_photo(message.chat.id, photo_stream, caption=f"🎨 Готово!\nПо запросу: {prompt}")
        bot.delete_message(chat_id=message.chat.id, message_id=thinking_msg.message_id)

    except Exception as e:
        err = str(e).lower()
        if "safety" in err or "policy" in err:
            text = "❌ Гугл заблокировал запрос (возможно, описание нарушает правила безопасности: насилие, 18+ и т.д.)."
        else:
            text = f"❌ Ошибка генерации:\n`{e}`"
        bot.edit_message_text(chat_id=message.chat.id, message_id=thinking_msg.message_id, text=text, parse_mode="Markdown")

# --- ОБРАБОТЧИКИ ФАЙЛОВ И ТЕКСТА ---

@bot.message_handler(content_types=['photo', 'video'])
def handle_media(message):
    chat_id = message.chat.id
    set_active_chat_id(chat_id)
    thinking_msg = None
    media = message.photo[-1] if message.content_type == 'photo' else message.video
    if media.file_size and media.file_size > MAX_FILE_SIZE:
        bot.reply_to(message, "⚠️ Файл тяжелее 20 МБ, Телега не даст скачать!")
        return
    try:
        data = bot.download_file(bot.get_file(media.file_id).file_path)
        mime = "image/jpeg" if message.content_type == 'photo' else "video/mp4"
        prompt = message.caption or "Изучи файл."
        thinking_msg = bot.reply_to(message, "<blockquote>⚙️ Мозгую варианты...</blockquote>", parse_mode="HTML")
        resp = call_with_503_retry(
            get_chat_session(chat_id).send_message,
            [types.Part.from_bytes(data=data, mime_type=mime), prompt],
        )
        update_stats(resp)
        if resp.text: bot.edit_message_text(chat_id=chat_id, message_id=thinking_msg.message_id, text=resp.text, parse_mode="Markdown")
    except Exception as e:
        handle_ai_error(e, chat_id, message, thinking_msg)

@bot.message_handler(content_types=['document'])
def handle_doc(message):
    chat_id = message.chat.id
    set_active_chat_id(chat_id)
    thinking_msg = None
    doc = message.document
    orig = doc.file_name or "file.pdf"
    fname = orig.lower()
    cap = message.caption or "Сделай стандартную раскладку."
    
    if doc.file_size and doc.file_size > MAX_FILE_SIZE:
        bot.reply_to(message, f"⚠️ Файл больше 20 МБ!")
        return
        
    try:
        data = bot.download_file(bot.get_file(doc.file_id).file_path)

        marks_response = register_marks_document(
            chat_id=chat_id,
            file_name=orig,
            pdf_bytes=data,
            caption=message.caption or "",
        )
        if marks_response:
            last_user_pdf[chat_id] = [(orig, data)]
            bot.reply_to(message, marks_response)
            return

        parts = []
        extracted_pdfs = []
        
        if fname.endswith(('.zip', '.rar')):
            arch = zipfile.ZipFile(io.BytesIO(data)) if fname.endswith('.zip') else rarfile.RarFile(io.BytesIO(data))
            with arch as a:
                for n in a.namelist():
                    if n.startswith('__MACOSX/') or n.endswith('/'): continue
                    if n.lower().endswith('.pdf'):
                        pd = a.read(n)
                        extracted_pdfs.append((os.path.basename(n), pd))
        elif fname.endswith('.pdf'):
            extracted_pdfs.append((orig, data))
            parts.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
        
        if not extracted_pdfs:
            bot.reply_to(message, "В архиве нет PDF файлов.")
            return
            
        last_user_pdf[chat_id] = extracted_pdfs
        
        # МАГИЯ: Передаем нейросети только ТЕКСТ (без тяжелых файлов)
        if len(extracted_pdfs) > 1 or 'честный знак' in cap.lower() or 'марки' in cap.lower():
            prompt = f"[СИСТЕМНОЕ: Пользователь скинул архив с {len(extracted_pdfs)} марками. Сообщение: '{cap}'. Если просит сдвинуть номера или зазоры, вызови tool_process_cz_archive с нужными параметрами, переведя их в миллиметры. Если нет - вызови со стандартными.]"
            thinking_msg = bot.reply_to(message, "<blockquote>⚙️ Читаю настройки раскладки ЧЗ...</blockquote>", parse_mode="HTML")
            resp = call_with_503_retry(
                get_chat_session(chat_id).send_message,
                prompt,
            )
            update_stats(resp)
            if resp.text: bot.edit_message_text(chat_id=chat_id, message_id=thinking_msg.message_id, text=resp.text, parse_mode="Markdown")
            return
            
        # Для обычных единичных файлов
        thinking_msg = bot.reply_to(message, "<blockquote>⚙️ Изучаю макеты...</blockquote>", parse_mode="HTML")
        parts.append(cap)
        resp = call_with_503_retry(
            get_chat_session(chat_id).send_message,
            parts,
        )
        update_stats(resp)
        if resp.text: bot.edit_message_text(chat_id=chat_id, message_id=thinking_msg.message_id, text=resp.text, parse_mode="Markdown")
    except Exception as e:
        handle_ai_error(e, chat_id, message, thinking_msg)

@bot.message_handler(content_types=['text'])
def handle_text(message):
    chat_id = message.chat.id
    set_active_chat_id(chat_id)
    try:
        thinking_msg = bot.reply_to(message, "<blockquote>⚙️ Думаю...</blockquote>", parse_mode="HTML")
        resp = call_with_503_retry(
            get_chat_session(chat_id).send_message,
            message.text,
        )
        update_stats(resp)
        if resp.text: bot.edit_message_text(chat_id=chat_id, message_id=thinking_msg.message_id, text=resp.text, parse_mode="Markdown")
    except Exception as e:
        handle_ai_error(e, chat_id, message, None)

print("Бот СЕНЯ (С умным пониманием команд, генерацией ИИ-картинок и ЧЗ) запущен!")
bot.infinity_polling(
    timeout=TELEGRAM_POLLING_TIMEOUT,
    long_polling_timeout=TELEGRAM_LONG_POLLING_TIMEOUT,
)
