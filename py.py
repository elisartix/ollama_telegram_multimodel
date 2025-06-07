import os
import signal
import atexit
import threading
import queue
import base64
import asyncio
import json
import logging
from io import BytesIO
from dotenv import load_dotenv
from configobj import ConfigObj
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand, WebAppInfo
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
import httpx

# Threading lock for thread-safety
DATA_LOCK = threading.Lock()

# 🔑 Load .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

BOT_TOKEN = os.getenv('BOT_TOKEN')
if not BOT_TOKEN:
    logging.error("BOT_TOKEN environment variable is not set.")
    raise ValueError("BOT_TOKEN environment variable is not set.")

# ⚙️ Ollama Settings
OLLAMA_URL = os.getenv('OLLAMA_URL', "http://localhost:11434/api/generate")
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))

# Constants for real-time message editing
EDIT_INTERVAL = float(os.getenv('EDIT_INTERVAL', 1.0))
MIN_CHARS_TO_EDIT = int(os.getenv('MIN_CHARS_TO_EDIT', 10))
CURSOR = "_"

# 🗂 Full list of available models
AVAILABLE_MODELS = {
    "mistral:latest": "Mistral (основная)",
    "gemma3:4b-it-fp16": "Gemma3 4B",
    "gemma3:12b-it-q4_K_M": "Gemma3 12B (квант.)",
    "gemma3:1b": "Gemma3 1B",
    "llava:13b": "LLaVA 13B",
    "gemma3:12b": "Gemma3 12B",
    "qwen2.5-coder:14b": "Qwen Coder 14B",
    "deepseek-coder-v2:latest": "DeepSeek Coder",
    "deepseek-r1:8b": "DeepSeek R1 8B",
    "openchat:latest": "OpenChat",
    "llama3.1:latest": "Llama 3.1"
}

# 🗂 Multimodal models
MULTIMODAL_MODELS = ["llava:13b"]

# 🗂 Public models and system prompt (loaded from settings file)
PUBLIC_MODELS = {}
SYSTEM_PROMPT = ""

# 🏷 Bot name
DEFAULT_BOT_NAME = os.getenv('DEFAULT_BOT_NAME', "elix.ai")

# 📂 Configuration files
WHITELIST_FILE = 'whitelist.txt'
ADMINS_FILE = 'admins.txt'
SETTINGS_FILE = 'settings.txt'
LOCK_FILE = 'bot.lock'

# 🗂 Data storage (in-memory)
chat_history = {}
approved_users = set()
auth_queue = queue.Queue()
user_settings = {}

# 🗂 Whitelisted and admin users (loaded from files)
whitelisted_chat_ids = set()
whitelisted_usernames = set()
admin_chat_ids = set()
admin_usernames = set()

# ⚙️ Default model ID (loaded from settings file)
DEFAULT_MODEL_ID_FROM_SETTINGS = None

# 🆕 Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 🆕 Commands for the menu
COMMANDS = [
    BotCommand("start", "🚀 Начать работу / Информация"),
    BotCommand("models", "🧠 Модели (Общий список)"),
    BotCommand("reset", "🧹 Очистить контекст"),
    BotCommand("admin", "🛠️ Панель администратора (Все модели)")
]

# --- UTILITY FUNCTIONS ---

def acquire_lock():
    """Ensure only one bot instance runs using a lock file."""
    if os.path.exists(LOCK_FILE):
        logger.error(f"Lock file '{LOCK_FILE}' exists. Another instance might be running.")
        raise RuntimeError("Bot is already running.")
    try:
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
        logger.info(f"Acquired lock with PID {os.getpid()}")
        return True
    except OSError as e:
        logger.error(f"Failed to create lock file: {e}")
        raise RuntimeError("Failed to create lock file.")

def release_lock():
    """Release the lock file."""
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
            logger.info(f"Lock file '{LOCK_FILE}' removed.")
    except OSError as e:
        logger.warning(f"Failed to remove lock file: {e}")

def cleanup():
    """Cleanup function to ensure lock file is removed on exit."""
    logger.info("Running cleanup...")
    release_lock()

atexit.register(cleanup)

def signal_handler(sig, frame):
    """Handle signals (Ctrl+C, process termination)."""
    logger.info(f"Received signal {sig}. Shutting down...")
    cleanup()
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def is_admin_user(chat_id, username):
    """Checks if a user is an admin."""
    username_lower = username.lower() if username else None
    return (chat_id in admin_chat_ids) or (username_lower and username_lower in admin_usernames)

def load_list_from_file(filename):
    """Loads chat_ids and usernames from a comma-separated file."""
    loaded_chat_ids = set()
    loaded_usernames = set()
    try:
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as f:
                pass
            logger.warning(f"File '{filename}' not found, created an empty one.")
            return loaded_chat_ids, loaded_usernames

        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                for item in content.split(','):
                    item = item.strip()
                    if not item:
                        continue
                    try:
                        loaded_chat_ids.add(int(item))
                    except ValueError:
                        loaded_usernames.add(item.lower())
        logger.info(f"Loaded from '{filename}': {len(loaded_chat_ids)} chat_ids and {len(loaded_usernames)} usernames.")
    except Exception as e:
        logger.error(f"Error reading file '{filename}': {e}")
    return loaded_chat_ids, loaded_usernames

def load_settings(filename):
    """Loads settings, public models, and system prompt from a ConfigObj file."""
    global SYSTEM_PROMPT, DEFAULT_MODEL_ID_FROM_SETTINGS, PUBLIC_MODELS
    resolved_default_model_id = None
    public_models = {}
    default_model_name_from_settings = None
    system_prompt_file = None

    try:
        config = ConfigObj(filename, encoding='utf-8')
        if 'settings' in config:
            settings_section = config['settings']
            default_model_name_from_settings = settings_section.get('default_model', '').strip()
            system_prompt_file = settings_section.get('system_prompt_file', '').strip()
        else:
            logger.warning(f"Section [settings] not found in '{filename}'.")

        if 'public_models' in config:
            for model_id, public_name in config['public_models'].items():
                model_id = model_id.strip()
                public_name = public_name.strip()
                if model_id in AVAILABLE_MODELS:
                    public_models[model_id] = public_name
                else:
                    logger.warning(f"Invalid model_id '{model_id}' in [public_models].")
        else:
            logger.warning(f"Section [public_models] not found in '{filename}'.")

        if system_prompt_file:
            full_system_prompt_path = os.path.join(os.path.dirname(filename), system_prompt_file)
            try:
                with open(full_system_prompt_path, 'r', encoding='utf-8') as f:
                    SYSTEM_PROMPT = f.read().strip()
                logger.info(f"System prompt loaded from '{full_system_prompt_path}'.")
            except FileNotFoundError:
                logger.error(f"System prompt file '{full_system_prompt_path}' not found.")
                SYSTEM_PROMPT = "You are {bot_name}, created by Elisey Gorbunov in 2025."
        else:
            logger.warning(f"Key 'system_prompt_file' not found. Using default system prompt.")
            SYSTEM_PROMPT = "You are {bot_name}, created by Elisey Gorbunov in 2025."

        if default_model_name_from_settings:
            if default_model_name_from_settings in AVAILABLE_MODELS:
                resolved_default_model_id = default_model_name_from_settings
            elif default_model_name_from_settings in public_models.values():
                for mid, name in public_models.items():
                    if name == default_model_name_from_settings:
                        resolved_default_model_id = mid
                        break
            else:
                logger.warning(f"Default model '{default_model_name_from_settings}' not found.")

        if not resolved_default_model_id and public_models:
            resolved_default_model_id = list(public_models.keys())[0]
            logger.info(f"Using first public model '{resolved_default_model_id}' as default.")
        elif not resolved_default_model_id and AVAILABLE_MODELS:
            resolved_default_model_id = list(AVAILABLE_MODELS.keys())[0]
            logger.info(f"Using first available model '{resolved_default_model_id}' as default.")
        else:
            logger.error("No available models to set a default.")

        PUBLIC_MODELS = public_models
        DEFAULT_MODEL_ID_FROM_SETTINGS = resolved_default_model_id
        logger.info(f"Settings loaded: default_model_id='{resolved_default_model_id}', public_models={len(public_models)}.")
    except Exception as e:
        logger.error(f"Error parsing settings file '{filename}': {e}")
    return resolved_default_model_id, public_models

def get_user_settings(chat_id):
    """Gets user settings, initializing with defaults if necessary."""
    with DATA_LOCK:
        if chat_id not in user_settings:
            default_model_id = DEFAULT_MODEL_ID_FROM_SETTINGS or list(AVAILABLE_MODELS.keys())[0]
            user_settings[chat_id] = {"model": default_model_id, "bot_name": DEFAULT_BOT_NAME}
            logger.info(f"Initialized settings for user {chat_id} with model '{default_model_id}'.")
        elif user_settings[chat_id]["model"] not in AVAILABLE_MODELS:
            logger.warning(f"User {chat_id}'s model '{user_settings[chat_id]['model']}' is no longer available.")
            fallback_model_id = list(AVAILABLE_MODELS.keys())[0]
            user_settings[chat_id]["model"] = fallback_model_id
            logger.info(f"Resetting user {chat_id} to '{fallback_model_id}'.")
        return user_settings[chat_id]

def build_models_keyboard(chat_id, models_dict):
    """Creates an inline keyboard for model selection."""
    keyboard = []
    row = []
    current_model = get_user_settings(chat_id)["model"]
    sorted_models = sorted(models_dict.items(), key=lambda x: x[1])
    for i, (model_id, model_name) in enumerate(sorted_models, 1):
        button_text = f"{model_name} ✅" if model_id == current_model else model_name
        row.append(InlineKeyboardButton(button_text, callback_data=f"model_{model_id}"))
        if i % 2 == 0:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    return InlineKeyboardMarkup(keyboard)

def get_full_prompt(chat_id, history):
    """Generates the full prompt including system prompt and chat history."""
    settings = get_user_settings(chat_id)
    prompt_with_name = SYSTEM_PROMPT.format(bot_name=settings.get("bot_name", DEFAULT_BOT_NAME))
    history_text = "\n".join(history)
    return f"{prompt_with_name}\n\n{history_text}\nAI:"

async def stream_ollama_response(chat_id, history, image_data_base64=None):
    """Sends a request to Ollama with streaming response. Yields text chunks."""
    settings = get_user_settings(chat_id)
    model_to_use = settings["model"]

    if model_to_use not in AVAILABLE_MODELS:
        logger.error(f"Unknown model '{model_to_use}' for chat {chat_id}.")
        yield "Ошибка: Выбранная модель недоступна."
        return

    payload = {"model": model_to_use, "stream": True}
    if image_data_base64:
        user_prompt = history[-1].split("с подписью: \"", 1)[-1].rstrip("\"") if history and "[Фото" in history[-1] else "Опиши изображение"
        payload["prompt"] = user_prompt
        payload["images"] = [image_data_base64]
        logger.info(f"Multimodal request: chat {chat_id}, model {model_to_use}, prompt: '{user_prompt[:50]}...'")
    else:
        payload["prompt"] = get_full_prompt(chat_id, history)
        logger.info(f"Text request: chat {chat_id}, model {model_to_use}, prompt: '{payload['prompt'][:50]}...'")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=30.0)) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    decoded_chunk = chunk.decode('utf-8')
                    for line in decoded_chunk.splitlines():
                        if line.strip():
                            data = json.loads(line)
                            if 'response' in data:
                                yield data['response']
                            elif 'error' in data:
                                logger.error(f"Ollama error for chat {chat_id}: {data['error']}")
                                yield f"Ошибка Ollama: {data['error']}"
                                return
    except httpx.RequestError as e:
        logger.error(f"Ollama API error for chat {chat_id}: {e}")
        yield f"Ошибка при запросе к модели: {e}"

# --- TELEGRAM HANDLERS ---

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Logs errors caused by Updates."""
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text("Произошла внутренняя ошибка. Пожалуйста, попробуйте позже.")
        except Exception:
            logger.error("Failed to send error message to user.")

async def post_init_callback(application):
    """Sets the bot's command menu."""
    try:
        await application.bot.set_my_commands(COMMANDS)
        logger.info("Bot commands set successfully.")
    except Exception as e:
        logger.error(f"Failed to set bot commands: {e}")

async def handle_web_app_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles data received from the Mini App."""
    chat_id = update.effective_chat.id
    with DATA_LOCK:
        if chat_id not in approved_users:
            await update.message.reply_text("Вы не авторизованы для использования этой функции.")
            return
    data = update.message.web_app_data.data
    logger.info(f"Received data from Mini App for user {update.effective_user.id}: {data}")
    await update.message.reply_text(f"Получены данные от Mini App: {data}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles photo messages and sends them to Ollama."""
    chat_id = update.effective_chat.id
    with DATA_LOCK:
        if chat_id not in approved_users:
            await update.message.reply_text("Вы не авторизованы для отправки фото.")
            return

    settings = get_user_settings(chat_id)
    current_model = settings["model"]
    if current_model not in MULTIMODAL_MODELS:
        await update.message.reply_text(f"Модель '{AVAILABLE_MODELS.get(current_model)}' не поддерживает изображения. Выберите другую модель.")
        return

    await update.message.chat.send_action("upload_photo")
    user_prompt = update.message.caption or ""
    response_message = None

    try:
        file_id = update.message.photo[-1].file_id
        telegram_file = await context.bot.get_file(file_id)
        file_content = BytesIO()
        await telegram_file.download_to_memory(file_content)
        file_content.seek(0)
        file_content_bytes = file_content.read()

        if len(file_content_bytes) > 20 * 1024 * 1024:
            await update.message.reply_text("Изображение слишком большое. Максимальный размер: 20 МБ.")
            return

        image_data_base64 = base64.b64encode(file_content_bytes).decode('utf-8')

        with DATA_LOCK:
            if chat_id not in chat_history:
                chat_history[chat_id] = []
            if user_prompt:
                caption_part = f' с подписью: "{user_prompt}"'
            else:
                caption_part = ""
            history_entry = f"User: [Фото{caption_part}]"
            chat_history[chat_id].append(history_entry)
            if len(chat_history[chat_id]) > 40:
                chat_history[chat_id] = chat_history[chat_id][-40:]

        response_message = await update.message.reply_text("Думаю...")
        full_response = ""
        last_edit_time = asyncio.get_running_loop().time()
        last_edited_length = 0

        async for chunk in stream_ollama_response(chat_id, chat_history[chat_id], image_data_base64):
            full_response += chunk
            current_time = asyncio.get_running_loop().time()
            if response_message and (current_time - last_edit_time > EDIT_INTERVAL) and \
               (len(full_response) - last_edited_length >= MIN_CHARS_TO_EDIT):
                try:
                    await response_message.edit_text(full_response + CURSOR)
                    last_edit_time = current_time
                    last_edited_length = len(full_response)
                except Exception:
                    response_message = None

        if full_response:
            if response_message:
                await response_message.edit_text(full_response)
            else:
                await update.message.reply_text(full_response)
        else:
            final_error_message = "Не удалось получить ответ от модели."
            if response_message:
                await response_message.edit_text(final_error_message)
            else:
                await update.message.reply_text(final_error_message)

        with DATA_LOCK:
            chat_history[chat_id].append(f"AI: {full_response}")
        logger.info(f"Photo response for chat {chat_id}: {full_response[:100]}...")

    except Exception as e:
        logger.error(f"Error during photo handling for chat {chat_id}: {e}")
        error_msg = "Произошла ошибка при обработке фото. Пожалуйста, попробуйте позже."
        if response_message:
            await response_message.edit_text(error_msg)
        else:
            await update.message.reply_text(error_msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles text messages and sends them to Ollama with streaming."""
    chat_id = update.effective_chat.id
    with DATA_LOCK:
        if chat_id not in approved_users:
            await update.message.reply_text("Вы не авторизованы для отправки сообщений.")
            return

    user_input = update.message.text
    logger.info(f"User {update.effective_user.id} in chat {chat_id}: {user_input}")

    with DATA_LOCK:
        if chat_id not in chat_history:
            chat_history[chat_id] = []
        chat_history[chat_id].append(f"User: {user_input}")
        if len(chat_history[chat_id]) > 40:
            chat_history[chat_id] = chat_history[chat_id][-40:]

    response_message = None
    try:
        response_message = await update.message.reply_text("Думаю...")
        full_response = ""
        last_edit_time = asyncio.get_running_loop().time()
        last_edited_length = 0

        async for chunk in stream_ollama_response(chat_id, chat_history[chat_id]):
            full_response += chunk
            current_time = asyncio.get_running_loop().time()
            if response_message and (current_time - last_edit_time > EDIT_INTERVAL) and \
               (len(full_response) - last_edited_length >= MIN_CHARS_TO_EDIT):
                try:
                    await response_message.edit_text(full_response + CURSOR)
                    last_edit_time = current_time
                    last_edited_length = len(full_response)
                except Exception:
                    response_message = None

        if full_response:
            if response_message:
                await response_message.edit_text(full_response)
            else:
                await update.message.reply_text(full_response)
        else:
            final_error_message = "Не удалось получить ответ от модели."
            if response_message:
                await response_message.edit_text(final_error_message)
            else:
                await update.message.reply_text(final_error_message)

        with DATA_LOCK:
            chat_history[chat_id].append(f"AI: {full_response}")
        logger.info(f"Bot response for chat {chat_id}: {full_response[:100]}...")

    except Exception as e:
        logger.error(f"Error during message handling for chat {chat_id}: {e}")
        error_msg = "Произошла ошибка при обработке сообщения. Пожалуйста, попробуйте позже."
        if response_message:
            await response_message.edit_text(error_msg)
        else:
            await update.message.reply_text(error_msg)

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Resets the chat context for the user."""
    chat_id = update.effective_chat.id
    with DATA_LOCK:
        if chat_id not in approved_users:
            await update.message.reply_text("Вы не авторизованы для сброса контекста.")
            return
        if chat_id in chat_history:
            del chat_history[chat_id]
            logger.info(f"Chat history reset for user {update.effective_user.id} in chat {chat_id}")
            await update.message.reply_text("🧹 Контекст чата сброшен.")
        else:
            await update.message.reply_text("🧹 Контекст чата уже пуст.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command."""
    chat_id = update.effective_chat.id
    user = update.effective_user
    username = user.username
    user_display = user.mention_html() if username else user.first_name

    keyboard = [[InlineKeyboardButton("Веб-чат", web_app=WebAppInfo(url="https://elix.loca.lt/"))]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    with DATA_LOCK:
        if chat_id in approved_users:
            await update.message.reply_html(
                f"Привет, {user_display}! Я — {DEFAULT_BOT_NAME}. Вы готовы к работе!\n"
                "Отправьте текст или фото. Используйте /models или /admin.",
                reply_markup=reply_markup
            )
            return

        is_whitelisted = (chat_id in whitelisted_chat_ids) or (username and username.lower() in whitelisted_usernames)
        if is_whitelisted:
            approved_users.add(chat_id)
            logger.info(f"Auto-approved user: Chat ID: {chat_id}, User: {user_display}")
            await update.message.reply_html(
                f"Привет, {user_display}! Я — {DEFAULT_BOT_NAME}. Доступ предоставлен.\n"
                "Отправьте текст или фото. Используйте /models или /admin.",
                reply_markup=reply_markup
            )
            return

    logger.info(f"Authorization request: Chat ID: {chat_id}, User: {user_display}")
    auth_queue.put((chat_id, user_display, username))
    await update.message.reply_text("Ваш запрос на доступ отправлен. Ожидайте одобрения.")

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /models command to show public models."""
    chat_id = update.effective_chat.id
    with DATA_LOCK:
        if chat_id not in approved_users:
            await update.message.reply_text("Вы не авторизованы для выбора моделей.")
            return
    if not PUBLIC_MODELS:
        await update.message.reply_text("Публичные модели недоступны.")
        return
    keyboard = build_models_keyboard(chat_id, PUBLIC_MODELS)
    current_model_name = PUBLIC_MODELS.get(get_user_settings(chat_id)["model"], "Неизвестно")
    await update.message.reply_text(f"Текущая модель: {current_model_name}\nВыберите модель:", reply_markup=keyboard)
    logger.info(f"User {update.effective_user.id} viewed public models.")

async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /admin command to show all available models."""
    chat_id = update.effective_chat.id
    if not is_admin_user(chat_id, update.effective_user.username):
        await update.message.reply_text("Эта команда доступна только администраторам.")
        return
    keyboard = build_models_keyboard(chat_id, AVAILABLE_MODELS)
    current_model_name = AVAILABLE_MODELS.get(get_user_settings(chat_id)["model"], "Неизвестно")
    await update.message.reply_text(f"🛡️ Панель администратора\nТекущая модель: {current_model_name}\nВыберите модель:", reply_markup=keyboard)
    logger.info(f"Admin user {update.effective_user.id} opened admin panel.")

async def model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles inline button presses for model selection."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    with DATA_LOCK:
        if chat_id not in approved_users:
            await query.edit_message_text("Вы не авторизованы.")
            return
    model_id = query.data.replace("model_", "")
    allowed_models = AVAILABLE_MODELS if is_admin_user(chat_id, query.from_user.username) else PUBLIC_MODELS
    if model_id not in allowed_models:
        logger.warning(f"User {query.from_user.id} attempted to select disallowed model '{model_id}'.")
        current_model_name = PUBLIC_MODELS.get(get_user_settings(chat_id)["model"], AVAILABLE_MODELS.get(get_user_settings(chat_id)["model"]))
        await query.edit_message_text(f"Недостаточно прав для модели '{AVAILABLE_MODELS.get(model_id)}'.\nТекущая: {current_model_name}\nВыберите:", reply_markup=build_models_keyboard(chat_id, allowed_models))
        return
    with DATA_LOCK:
        user_settings[chat_id]["model"] = model_id
        if chat_id in chat_history:
            del chat_history[chat_id]
    logger.info(f"User {query.from_user.id} changed model to '{model_id}'.")
    displayed_model_name = PUBLIC_MODELS.get(model_id, AVAILABLE_MODELS.get(model_id))
    message_text = f"Модель изменена на: {displayed_model_name}\n{'🛡️ Выберите модель:' if is_admin_user(chat_id, query.from_user.username) else 'Выберите модель:'}"
    await query.edit_message_text(message_text, reply_markup=build_models_keyboard(chat_id, allowed_models))

def handle_auth_requests():
    """Handles authorization requests from the queue in a separate thread."""
    logger.info("Authorization handler thread started.")
    while True:
        try:
            chat_id, user_display, username = auth_queue.get()
            print(f"\n--- Authorization Request ---")
            print(f"Chat ID: {chat_id}")
            print(f"User: {user_display} (@{username})")
            print("Approve access? (y/n): ")
            response = input().strip().lower()
            if response == 'y':
                with DATA_LOCK:
                    approved_users.add(chat_id)
                logger.info(f"Approved user: Chat ID: {chat_id}, User: {user_display}")
                print(f"User {user_display} approved.")
            else:
                logger.info(f"Denied user: Chat ID: {chat_id}, User: {user_display}")
                print(f"User {user_display} denied.")
            auth_queue.task_done()
            print("-----------------------------")
        except Exception as e:
            logger.error(f"Error in authorization handler thread: {e}")
            auth_queue.task_done()

# --- MAIN FUNCTION ---

def main():
    """Main function to start the bot."""
    global whitelisted_chat_ids, whitelisted_usernames, admin_chat_ids, admin_usernames
    try:
        acquire_lock()
        whitelisted_chat_ids, whitelisted_usernames = load_list_from_file(WHITELIST_FILE)
        admin_chat_ids, admin_usernames = load_list_from_file(ADMINS_FILE)
        DEFAULT_MODEL_ID_FROM_SETTINGS, PUBLIC_MODELS = load_settings(SETTINGS_FILE)
        with DATA_LOCK:
            approved_users.update(whitelisted_chat_ids)

        if not AVAILABLE_MODELS:
            logger.critical("No available models configured. Exiting.")
            return

        app = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init_callback).build()
        app.add_error_handler(error_handler)
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("models", models_command))
        app.add_handler(CommandHandler("admin", admin_command))
        app.add_handler(CommandHandler("reset", reset))
        app.add_handler(CallbackQueryHandler(model_selection, pattern="^model_"))
        app.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        app.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, handle_web_app_data))

        logger.info("Bot is starting polling...")
        auth_thread = threading.Thread(target=handle_auth_requests, daemon=True)
        auth_thread.start()
        app.run_polling(drop_pending_updates=True)
    except Exception as e:
        logger.critical(f"Unhandled error during bot startup: {e}", exc_info=True)
    finally:
        release_lock()
        logger.info("Bot stopped.")

if __name__ == "__main__":
    main()
