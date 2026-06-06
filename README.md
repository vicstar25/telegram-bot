# Telegram Moderation Bot

A Python Telegram bot that saves group messages to SQLite, responds to simple friendly prompts, and moderates spam or toxic messages with trained scikit-learn models.

## Features

- Saves normal text messages to `message.db`
- Ignores normal messages without replying
- Replies to simple greetings, thanks, and quote requests
- Deletes messages classified as spam or toxic
- Sends warning messages after moderation actions
- Provides admin commands for manual warnings and stats

## Project Files

- `bot.py` - Main Telegram bot application
- `database.py` - SQLite message storage helpers
- `train_spam_model.py` - Trains the spam classifier
- `train_model.py` - Trains the toxic-message classifier
- `models/` - Saved model files
- `Data/` - Training datasets
- `message.db` - Local SQLite database, generated at runtime

## Setup

Create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set your Telegram bot token as an environment variable:

```powershell
$env:TELEGRAM_BOT_TOKEN="your_bot_token_here"
```

Start the bot:

```powershell
python bot.py
```

For message deletion to work in a Telegram group, add the bot as an admin and give it permission to delete messages.

## Configuration

The moderation thresholds can be tuned with environment variables:

```powershell
$env:SPAM_THRESHOLD="0.85"
$env:TOXIC_THRESHOLD="0.85"
```

Higher values reduce false positives. Lower values catch more messages but may flag normal messages.

## Commands

- `/start` - Start the bot
- `/help` - Show help
- `/about` - Show bot information
- `/quote` - Send a motivational quote
- `/warn` - Warn a user, admin only, used by replying to a message
- `/stats` - Show moderation stats, admin only

## Keeping Secrets Safe

Do not hardcode your Telegram API token in `bot.py` or any file that will be pushed to GitHub. Use `TELEGRAM_BOT_TOKEN` instead.

If your token was already committed or shown publicly, create a new token with BotFather before pushing the project. Removing it from the latest file is not enough if it exists in Git history.

Before pushing, check for secrets:

```powershell
git status
git diff
git log --all -S "TELEGRAM_BOT_TOKEN"
```

If the real token appears anywhere in Git history, rotate it with BotFather.
