import zipfile
import io
import telebot
from google import genai
from google.genai import types
from settings import GEMINI_CHAT_MODEL, TELEGRAM_TOKEN

# --- НАСТРОЙКА КЛЮЧЕЙ И API ---
ai_client = genai.Client()
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# --- ПАМЯТЬ БОТА ---
user_chats = {}

def get_chat_session(chat_id):
    if chat_id not in user_chats:
        print(f"-> Создана новая сессия памяти для {chat_id}")
        
        persona = (
            "Твое имя Сеня. Ты общаешься максимально неформально, как старый друг-айтишник. "
            "Ты можешь немного дерзить, подкалывать собеседника, использовать легкий сарказм "
            "и айтишный сленг. Никогда не будь слишком вежливым или официальным, общайся на 'ты', "
            "но при этом всегда отвечай на суть вопроса и помогай пользователю."
        )
        
        user_chats[chat_id] = ai_client.chats.create(
            model=GEMINI_CHAT_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=persona
            )
        )
    return user_chats[chat_id]

print("Бот СЕНЯ (с поддержкой ZIP) запущен. Напишите ему в Telegram!")

# --- 1. КОМАНДЫ /start И /reset ---
@bot.message_handler(commands=['start', 'reset'])
def send_welcome(message):
    chat_id = message.chat.id
    if chat_id in user_chats:
        del user_chats[chat_id]
        
    print(f"-> История очищена для {chat_id}")
    bot.reply_to(message, "Ну чё, контекст сброшен. Валяй по новой! Кидай текст, фотку, PDF или ZIP-архив с марками.")

# --- 2. ОБРАБОТКА ФОТОГРАФИЙ ---
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    chat_id = message.chat.id
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_bytes = bot.download_file(file_info.file_path)
        
        user_prompt = message.caption if message.caption else "Чё тут нарисовано?"
        
        file_part = types.Part.from_bytes(
            data=downloaded_bytes,
            mime_type="image/jpeg"
        )
        
        chat_session = get_chat_session(chat_id)
        response = chat_session.send_message([file_part, user_prompt])
        bot.reply_to(message, response.text)
    except Exception as e:
        print(f"!! Ошибка photo: {e}")
        bot.reply_to(message, f"Упс, картинка сломалась. Ошибка: {e}")

# --- 3. ОБРАБОТКА ДОКУМЕНТОВ (PDF, Картинки, ZIP) ---
@bot.message_handler(content_types=['document'])
def handle_document(message):
    chat_id = message.chat.id
    file_name = message.document.file_name.lower() if message.document.file_name else ""
    caption = message.caption if message.caption else "Разбери этот файл и подскажи, что с ним делать."

    bot.reply_to(message, "Секунду, отправляю файл на анализ...")

    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_bytes = bot.download_file(file_info.file_path)
        chat_session = get_chat_session(chat_id)

        # 1. Если это ZIP-архив
        if file_name.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(downloaded_bytes)) as z:
                parts = []
                for name in z.namelist():
                    name_lower = name.lower()
                    if name_lower.endswith(('.pdf', '.png', '.jpg', '.jpeg', '.webp')):
                        file_data = z.read(name)
                        mime = "application/pdf" if name_lower.endswith('.pdf') else "image/jpeg"
                        parts.append(types.Part.from_bytes(data=file_data, mime_type=mime))
                
                if parts:
                    parts.append(caption)
                    response = chat_session.send_message(parts)
                    bot.reply_to(message, response.text)
                else:
                    bot.reply_to(message, "В архиве нет поддерживаемых файлов (PDF или картинок).")

        # 2. Если это PDF
        elif file_name.endswith('.pdf'):
            file_part = types.Part.from_bytes(data=downloaded_bytes, mime_type="application/pdf")
            response = chat_session.send_message([file_part, caption])
            bot.reply_to(message, response.text)
            
        # 3. Если это картинка файлом
        elif file_name.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            file_part = types.Part.from_bytes(data=downloaded_bytes, mime_type="image/jpeg")
            response = chat_session.send_message([file_part, caption])
            bot.reply_to(message, response.text)
        else:
            bot.reply_to(message, "Я пока умею работать только с PDF, картинками и ZIP-архивами.")

    except Exception as e:
        print(f"!! Ошибка при обработке файла {file_name}: {e}")
        bot.reply_to(message, f"Не смог переварить твой файл. Ошибка: {e}")

# --- 4. ОБРАБОТКА ТЕКСТА ---
@bot.message_handler(content_types=['text'])
def handle_text(message):
    chat_id = message.chat.id
    try:
        chat_session = get_chat_session(chat_id)
        response = chat_session.send_message(message.text)
        bot.reply_to(message, response.text)
    except Exception as e:
        print(f"!! Ошибка text: {e}")
        bot.reply_to(message, f"Чёт всё упало: {e}")

bot.infinity_polling(timeout=20, long_polling_timeout=10)
