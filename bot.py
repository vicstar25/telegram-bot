import os
import random
import re
from pathlib import Path
from dotenv import load_dotenv
import argparse
from telegram import Update
from telegram.error import NetworkError
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, ApplicationHandlerStop
from telegram.ext import MessageHandler, filters
import joblib
import database
# Load environment from a .env file if present, and allow overriding via CLI.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
parser = argparse.ArgumentParser(
    description="Telegram bot runner. Use --token or set TELEGRAM_BOT_TOKEN in .env or environment.",
    add_help=False,
)
parser.add_argument("--token", "-t", help="Telegram bot token")
parser.add_argument("--help", "-h", action="help", help="Show this help message and exit")
args, _ = parser.parse_known_args()
raw_token = args.token or os.environ.get("TELEGRAM_BOT_TOKEN")
TOKEN = raw_token.strip() if raw_token else None
if TOKEN in {"your-telegram-bot-token-here", "your_bot_token_here", "YOUR_TOKEN"}:
    TOKEN = None
TOKEN_SOURCE = "CLI" if args.token else ("ENV/.env" if raw_token else None)

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
# Set to 0.0 to use the threshold found during model training (recommended).
# Or set to a value like 0.5-0.95 to override with a fixed threshold.
SPAM_THRESHOLD = float(os.environ.get("SPAM_THRESHOLD", "0.0"))
TOXIC_THRESHOLD = float(os.environ.get("TOXIC_THRESHOLD", "0.0"))
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
    "who come be this one",
    "who come be this one?",
    "who come be this",
    "who be this one",
    "who this one be",
    "how far",
    "how far everyone",
    "how you dey",
    "how you dey?",
    "how you dey do",
    "how body",
    "how your day",
    "how your day dey",
    "how your day dey go",
    "how your day going",
    "how your day dey go?",
    "how your day dey?",
    "i dey",
    "i dey fine",
    "i dey o",
    "i dey ooo",
    "no wahala",
    "no wahala o",
    "my brother",
    "my guy",
    "my sister",
    "abeg",
    "abeg o",
    "bros",
    "bros how far",
    "sis",
    "soji",
    "welcome o",
    "welcome ooo",
    "na you",
    "na so",
    "na wa o",
    "chai",
    "lol",
    "lmao",
    "smh",
    "ok o",
    "ok ooo",
    "ok na",
    "alright",
    "okay",
    "fine",
    "fine o",
    "yes o",
    "no o",
    "see you",
    "see you later",
    "take care",
    "have a nice day",
    "have a good one",
    "bless you",
    "take care o",
    "come go",
    "make we go",
    "let's go",
    "make we move",
    "i come",
    "i don come",
    "i don come o",
    "who send you",
    "who send you?",
    "wetin",
    "wetin dey",
    "wetin dey happen",
    "wetin dey sup",
    "wetin dey happen?",
    "wetin sup",
    "wetin dey sup?",
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

# Spammy words that should not appear in messages bypassing the model
_SPAMMY_WORDS = {
    "buy", "sell", "cheap", "price", "offer", "deal", "discount",
    "promotion", "limited", "urgent", "cash", "money", "free",
    "win", "winner", "won", "prize", "claim", "click", "link",
    "subscribe", "follow", "share", "invest", "bitcoin", "crypto",
    "loan", "credit", "score", "trader", "trading", "signal",
    "profit", "guaranteed", "earn", "income", "payment", "paid",
}


def normalize_message(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _has_spammy_content(words: set) -> bool:
    """Check if message contains obvious spam keywords."""
    return bool(words & _SPAMMY_WORDS)


def _has_url(text: str) -> bool:
    """Check if message contains a URL or mention."""
    return any(token in text for token in ("http", "www.", "t.me", "@"))


def is_obviously_safe_message(text: str) -> bool:
    normalized = normalize_message(text)
    words = set(normalized.split())

    # Always trust exact matches from the safe-list
    if normalized in SAFE_SHORT_MESSAGES:
        return True

    # If there's any spammy content, always check with the model
    if _has_spammy_content(words):
        return False

    # Short casual messages without spam content are safe
    word_count = len(words)
    if word_count <= 4 and not _has_url(text):
        return True

    # Pidgin / casual conversational patterns
    pidgin_markers = {"dey", "na", "abeg", "wetin", "o", "ooo", "bros", "how far", "wahala"}
    if word_count <= 6 and not _has_url(text):
        if any(marker in words for marker in pidgin_markers):
            return True

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
        elif score >= 0.3:
            # Log near-misses so we can tune the threshold
            print(f"    [DEBUG] Near-miss spam: score={score:.4f}, threshold={threshold:.4f}, text={text!r}")

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

    # Handle "conflict" errors (another instance is running)
    error_text = str(error).lower()
    if "conflict" in error_text or "terminated by other getupdates" in error_text:
        print(f"⚠️  Conflict error (another bot instance is running): {error}")
        print("   The bot will keep retrying until the old session times out (~30s).")
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

    # Close any stale polling sessions from previous instances
    # This prevents "Conflict: terminated by other getUpdates request" errors
    try:
        await app.bot.close()
        await app.bot.initialize()
        print("Cleared old polling sessions.")
    except Exception as e:
        print(f"Session cleanup (non-critical): {e}")

    # Preload ML models at startup so the first message isn't delayed
    print("Preloading spam model...")
    load_spam_model()
    print("Preloading toxic model...")
    load_toxic_model()
    print("Models loaded. Bot is ready.")

# Main function
def main():
    if not TOKEN:
        print("ERROR: No Telegram bot token was provided.")
        if TOKEN_SOURCE:
            print(f"Token source detected: {TOKEN_SOURCE}")
            print("But the value is a placeholder or invalid token string.")
        print("Use one of these options:")
        print("  1) python bot.py --token YOUR_TOKEN")
        print(f"  2) edit {BASE_DIR / '.env'} and set TELEGRAM_BOT_TOKEN=YOUR_TOKEN")
        print()
        parser.print_help()
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
    # Avoid Windows platform signal detection issues when running in some environments.
    app.run_polling(stop_signals=())

if __name__ == "__main__":
    main()
