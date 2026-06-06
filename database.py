import sqlite3
from datetime import datetime

#connect to the database
conn = sqlite3.connect('message.db', check_same_thread=False)
cursor = conn.cursor()

# create table 
cursor.execute('''CREATE TABLE IF NOT EXISTS messages(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    message TEXT,
                    timestamp TEXT)''')
conn.commit()

# create 
def save_message(username, message ):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO messages (username, message, timestamp) VALUES (?, ?, ?)",
                   (username, message, timestamp))
    conn.commit()

# read 
def get_messages():
    cursor.execute("SELECT username, message, timestamp FROM messages")
    return cursor.fetchall()

# delete
def delete_message(message_id):
    cursor.execute("DELETE FROM messages WHERE id = ?", (message_id,))
    conn.commit()