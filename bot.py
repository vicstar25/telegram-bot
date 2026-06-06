import os
import random
import re
from pathlib import Path
from telegram import Update
from telegram.error import NetworkError
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, ApplicationHandlerStop
from telegram.ext import MessageHandler, filters
import joblib
import database
# Keep the token out of source control. Set TELEGRAM_BOT_TOKEN in your shell.
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

WELCOME_TEXT = (
    "Hello, group! I'm here to share positive quotes and respond to simple greetings.\n"
    "Type /help to see the available commands."
)

HELP_TEXT = (
    "Here are the commands I understand:\n"
    "/start - Start the bot\n"
    "/help - Show this help message\n"
    "/about - Learn more about this bot\n"
    "/quote - Receive a motivational quote\n"
    "/warn - Warn a user (admin only, reply to a message)\n"
    "/stats - Show moderation stats (admin only)\n"
    "I only reply to simple group-friendly messages."
)

QUOTE_LIST = [
    "Believe you can and you're halfway there.",
    "Every moment is a fresh beginning.",
    "Small steps every day lead to big changes.",
    "The only limit is your mind.",
]

GREETING_RESPONSES = [
    "Hi everyone!",
    "Hello!",
    "Hey, welcome!",
]

ADMIN_STATUSES = {"administrator", "creator"}
SPAM_MODEL_PATH = Path("models/spam_message_model.joblib")
TOXIC_MODEL_PATH = Path("models/toxic_message_model.joblib")
SPAM_THRESHOLD = float(os.environ.get("SPAM_THRESHOLD", "0.85"))
TOXIC_THRESHOLD = float(os.environ.get("TOXIC_THRESHOLD", "0.85"))
SPAM_MODEL = None
TOXIC_MODEL = None
SAFE_SHORT_MESSAGES = {
    "good afternoon",
    "good evening",
    "good morning",
    "hello",
    "hello everyone",
    "hello sir",
    "hello sir how are you",
    "hey",
    "hi",
    "thank you",
    "thanks",
}

def get_stats(context: ContextTypes.DEFAULT_TYPE) -> dict:
    stats = context.application.bot_data.setdefault(
        "stats",
        {
            "messages_seen": 0,
            "model_deleted": 0,
            "spam_deleted": 0,
            "toxic_deleted": 0,
            "warnings_sent": 0,
            "manual_warnings": 0,
        },
    )
    defaults = {
        "messages_seen": 0,
        "model_deleted": 0,
        "spam_deleted": 0,
        "toxic_deleted": 0,
        "warnings_sent": 0,
        "manual_warnings": 0,
    }
    for key, value in defaults.items():
        stats.setdefault(key, value)
    return stats

def mention_user(user) -> str:
    if user.username:
        return f"@{user.username}"
    return user.full_name or "this user"

async def is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    chat = update.effective_chat
    if not user or not chat:
        return False

    if chat.type == "private":
        return True

    member = await context.bot.get_chat_member(chat.id, user.id)
    return member.status in ADMIN_STATUSES

async def require_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if await is_admin(update, context):
        return True

    await update.message.reply_text("Admin-only command.")
    return False

def load_model(path: Path, cache_name: str):
    try:
        model = joblib.load(path)
        print(f"Loaded {cache_name} model from {path}")
        return model
    except Exception as error:
        print(f"Could not load {cache_name} model: {error}")
        return None

def load_spam_model():
    global SPAM_MODEL

    if SPAM_MODEL is not None:
        return SPAM_MODEL

    if not SPAM_MODEL_PATH.exists():
        return None

    SPAM_MODEL = load_model(SPAM_MODEL_PATH, "spam")
    return SPAM_MODEL

def load_toxic_model():
    global TOXIC_MODEL

    if TOXIC_MODEL is not None:
        return TOXIC_MODEL

    if not TOXIC_MODEL_PATH.exists():
        return None

    TOXIC_MODEL = load_model(TOXIC_MODEL_PATH, "toxic")
    return TOXIC_MODEL

def model_score(artifact, text: str) -> float:
    if not artifact or not text.strip():
        return 0.0

    model = artifact["model"] if isinstance(artifact, dict) else artifact
    return float(model.predict_proba([text])[0][1])

def artifact_threshold(artifact, env_threshold: float) -> float:
    if env_threshold > 0:
        return env_threshold

    if isinstance(artifact, dict):
        return float(artifact.get("threshold", 0.5))

    return 0.5

def normalize_message(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def is_obviously_safe_message(text: str) -> bool:
    normalized = normalize_message(text)
    if normalized in SAFE_SHORT_MESSAGES:
        return True

    words = normalized.split()
    if len(words) <= 4 and words and words[0] in {"hi", "hello", "hey"}:
        return not any(word in normalized for word in ["http", "www", "t.me", "@"])

    return False

def spam_score(text: str) -> float:
    return model_score(load_spam_model(), text)

def toxic_score(text: str) -> float:
    return model_score(load_toxic_model(), text)

def is_spam_message(text: str) -> bool:
    artifact = load_spam_model()
    if not artifact or not text.strip():
        return False

    return model_score(artifact, text) >= artifact_threshold(artifact, SPAM_THRESHOLD)

def is_toxic_message(text: str) -> bool:
    artifact = load_toxic_model()
    if not artifact or not text.strip():
        return False

    return model_score(artifact, text) >= artifact_threshold(artifact, TOXIC_THRESHOLD)

def classify_message(text: str) -> dict[str, tuple[float, float]]:
    labels = {}

    if is_obviously_safe_message(text):
        return labels

    spam_artifact = load_spam_model()
    if spam_artifact and text.strip():
        score = model_score(spam_artifact, text)
        threshold = artifact_threshold(spam_artifact, SPAM_THRESHOLD)
        if score >= threshold:
            labels["spam"] = (score, threshold)

    toxic_artifact = load_toxic_model()
    if toxic_artifact and text.strip():
        score = model_score(toxic_artifact, text)
        threshold = artifact_threshold(toxic_artifact, TOXIC_THRESHOLD)
        if score >= threshold:
            labels["toxic"] = (score, threshold)

    return labels

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(WELCOME_TEXT)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "I'm a simple Telegram bot built with python-telegram-bot. "
        "I keep replies short and group-friendly."
    )

async def quote(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(random.choice(QUOTE_LIST))

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("pong")

async def warn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await require_admin(update, context):
        return

    target_message = update.message.reply_to_message
    if not target_message or not target_message.from_user:
        await update.message.reply_text("Reply to a user's message with /warn.")
        return

    reason = " ".join(context.args).strip() or "Please follow the group rules."
    target_user = target_message.from_user
    stats = get_stats(context)
    stats["warnings_sent"] += 1
    stats["manual_warnings"] += 1

    await update.message.reply_text(
        f"Warning for {mention_user(target_user)}: {reason}"
    )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await require_admin(update, context):
        return

    current_stats = get_stats(context)
    await update.message.reply_text(
        "Bot stats:\n"
        f"Messages seen: {current_stats['messages_seen']}\n"
        f"Model deleted: {current_stats.get('model_deleted', 0)}\n"
        f"Spam deleted: {current_stats.get('spam_deleted', 0)}\n"
        f"Toxic deleted: {current_stats.get('toxic_deleted', 0)}\n"
        f"Warnings sent: {current_stats['warnings_sent']}\n"
        f"Manual warnings: {current_stats['manual_warnings']}"
    )

def get_bot_response(user_text: str) -> str:
    text = (user_text or "").strip().lower()

    if any(word in text for word in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
        return random.choice(GREETING_RESPONSES)

    if "thank" in text:
        return "You're welcome!"

    if any(word in text for word in ["quote", "motivate", "inspire"]):
        return random.choice(QUOTE_LIST)

    return ""

async def handle_bot_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text or ""
    response = get_bot_response(user_text)
    if response:
        await update.message.reply_text(response)

async def moderate_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    get_stats(context)["messages_seen"] += 1
    user_text = update.message.text or update.message.caption or ""
    labels = classify_message(user_text)
    if not labels:
        return

    stats = get_stats(context)
    user = update.message.from_user

    try:
        await update.message.delete()
        stats["model_deleted"] += 1
        if "spam" in labels:
            stats["spam_deleted"] += 1
        if "toxic" in labels:
            stats["toxic_deleted"] += 1
    except Exception as error:
        print(f"Could not delete model-flagged message: {error}")

    stats["warnings_sent"] += 1
    reason = " and ".join(sorted(labels))
    score_details = ", ".join(
        f"{label}={score:.3f}/{threshold:.3f}"
        for label, (score, threshold) in sorted(labels.items())
    )
    print(f"Flagged message from {mention_user(user)} ({score_details}): {user_text!r}")
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Warning for {mention_user(user)}: {reason} messages are not allowed here.",
    )
    raise ApplicationHandlerStop

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Unknown command. Try /help."
    )
# Function to handle log messages
async def log_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        user = update.message.from_user
        chat = update.message.chat
        text = update.message.text
        print("\n--- New Message ---")
        print(f"user: {user.username}")
        print(f"chat ID: {chat.id}")
        print(f"message: {text}")

async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    error = context.error
    if isinstance(error, NetworkError):
        print(f"Network error while connecting to Telegram: {error}")
        print("Check your internet connection, DNS, VPN/proxy, or firewall, then restart the bot.")
        return

    print(f"Bot error: {error}")

async def handle_message(update, context):
    user = update.message.from_user
    username = user.username or user.full_name
    text = update.message.text

    # call function from database.py
    database.save_message(username, text)

    print(f"{username}: {text}")

    response = get_bot_response(text)
    if response:
        await update.message.reply_text(response)

async def show_messages(update, context):
    messages = database.get_messages()
    if not messages:
        await update.message.reply_text("No messages found.")
        return

    response = "Saved Messages:\n"
    for username, message, timestamp in messages:
        response += f"{timestamp} - {username}: {message}\n"

    await update.message.reply_text(response)
    
async def post_init(app):
    bot = await app.bot.get_me()
    print(f"Connected as @{bot.username} (id: {bot.id})")

# Main function
def main():
    if not TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN before starting the bot.")

    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(CommandHandler("quote", quote))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("warn", warn))
    app.add_handler(CommandHandler("stats", stats))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, moderate_message), group=0)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message), group=1)
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    app.add_handler(MessageHandler(filters.ALL, log_message), group=2)
    app.add_error_handler(handle_error)
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
