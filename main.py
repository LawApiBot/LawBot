import os
from typing import Dict, Optional
import logging
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from datetime import datetime
import io
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import docx

# Добавьте в начало с остальными настройками
SUPPORTED_FILE_TYPES = {
    'text/plain': 'txt',
    'application/pdf': 'pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'image/png': 'image',
    'image/jpeg': 'image'
}

MAX_FILE_SIZE = 512 * 1024 * 1024  # 15 MB


# Добавьте новые функции обработки файлов
async def download_telegram_file(file_id: str):
    async with httpx.AsyncClient() as client:
        # Получаем информацию о файле
        file_info = await client.get(
            f"{TELEGRAM_API_URL}/getFile?file_id={file_id}"
        )
        file_path = file_info.json()['result']['file_path']

        # Скачиваем файл
        file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        file_response = await client.get(file_url)
        return file_response.content


def extract_text_from_file(content: bytes, file_type: str):
    try:
        # Обработка текстовых файлов
        if file_type == 'txt':
            return content.decode('utf-8')

        # Обработка PDF
        elif file_type == 'pdf':
            text = ""
            pdf = PdfReader(io.BytesIO(content))
            for page in pdf.pages:
                text += page.extract_text()
            return text

        # Обработка Word документов
        elif file_type == 'docx':
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([para.text for para in doc.paragraphs])

        # Обработка изображений с OCR
        elif file_type == 'image':
            image = Image.open(io.BytesIO(content))
            return pytesseract.image_to_string(image)

        else:
            return None
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Telegram OpenAI Assistant Bot")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
DATABASE_URL = os.getenv("DATABASE_URL")

# Global shared thread ID
SHARED_THREAD_ID = None

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize the database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create users table to store user information
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            chat_id BIGINT PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create messages table to store message history
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chat_id) REFERENCES users (chat_id)
        )
        ''')
        
        # Create shared_thread table to store the global thread ID
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS shared_thread (
            id SERIAL PRIMARY KEY,
            thread_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()

# Initialize database on startup
init_db()

def get_or_create_shared_thread():
    """Get the shared thread ID or create a new one if it doesn't exist"""
    global SHARED_THREAD_ID
    
    # If we already have the thread ID in memory, return it
    if SHARED_THREAD_ID:
        return SHARED_THREAD_ID
    
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Try to get existing thread from database
        cursor.execute("SELECT thread_id FROM shared_thread ORDER BY created_at DESC LIMIT 1")
        thread_record = cursor.fetchone()
        
        if thread_record:
            SHARED_THREAD_ID = thread_record["thread_id"]
            logger.info(f"Using existing shared thread: {SHARED_THREAD_ID}")
            return SHARED_THREAD_ID
        
        # Create new thread with OpenAI
        thread = client.beta.threads.create()
        SHARED_THREAD_ID = thread.id
        
        # Store the new thread ID
        cursor.execute(
            "INSERT INTO shared_thread (thread_id) VALUES (%s)",
            (SHARED_THREAD_ID,)
        )
        conn.commit()
        
        logger.info(f"Created new shared thread: {SHARED_THREAD_ID}")
        return SHARED_THREAD_ID

# Telegram API base URL
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

class WebhookInfo(BaseModel):
    url: str

@app.get("/")
async def root():
    return {"status": "running", "message": "Telegram OpenAI Assistant Bot is running"}

@app.post("/set-webhook")
async def set_webhook(webhook_info: WebhookInfo):
    """Set the Telegram webhook URL"""
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=500, detail="Telegram bot token not configured")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TELEGRAM_API_URL}/setWebhook",
            json={"url": webhook_info.url}
        )
        result = response.json()
    
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=f"Failed to set webhook: {result}")
    
    return {"status": "success", "message": "Webhook set successfully", "result": result}

async def send_telegram_message(chat_id: int, text: str):
    """Send a message to a Telegram chat"""
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )

async def send_chat_action(chat_id: int, action: str = "typing"):
    """Send a chat action to a Telegram chat (e.g., typing)"""
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{TELEGRAM_API_URL}/sendChatAction",
            json={"chat_id": chat_id, "action": action}
        )

def register_user(chat_id: int, username: str = None, first_name: str = None, last_name: str = None):
    """Register a user or update their last interaction time"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if user exists
        cursor.execute("SELECT chat_id FROM users WHERE chat_id = %s", (chat_id,))
        user = cursor.fetchone()
        
        if user:
            # Update last interaction time
            cursor.execute(
                "UPDATE users SET last_interaction = NOW(), username = %s, first_name = %s, last_name = %s WHERE chat_id = %s",
                (username, first_name, last_name, chat_id)
            )
        else:
            # Insert new user
            cursor.execute(
                """
                INSERT INTO users (chat_id, username, first_name, last_name)
                VALUES (%s, %s, %s, %s)
                """,
                (chat_id, username, first_name, last_name)
            )
        
        conn.commit()

def store_message(chat_id: int, role: str, content: str, username: str = None, first_name: str = None, last_name: str = None):
    """Store a message in the database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (chat_id, username, first_name, last_name, role, content) VALUES (%s, %s, %s, %s, %s, %s)",
            (chat_id, username, first_name, last_name, role, content)
        )
        conn.commit()

def get_user_display_name(username: str = None, first_name: str = None, last_name: str = None):
    """Get a display name for the user"""
    if username:
        return f"@{username}"
    elif first_name and last_name:
        return f"{first_name} {last_name}"
    elif first_name:
        return first_name
    else:
        return "Anonymous User"

async def process_message(
        chat_id: int,
        text: Optional[str] = None,
        file_content: Optional[str] = None,
        username: str = None,
        first_name: str = None,
        last_name: str = None
):
    try:
        register_user(chat_id, username, first_name, last_name)

        # Определяем тип контента
        content = text if text else file_content
        content_type = "text" if text else "file"

        # Сохраняем сообщение с указанием типа
        store_message(chat_id, "user", content, username, first_name, last_name)

        thread_id = get_or_create_shared_thread()
        await send_chat_action(chat_id, "typing")

        user_display = get_user_display_name(username, first_name, last_name)
        formatted_message = f"{user_display} [{content_type}]: {content}"

        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=formatted_message
        )

        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        )

        # Poll for the run to complete
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
        
        # Wait for the run to complete (in a real app, use a more sophisticated approach)
        import time
        while run_status.status not in ["completed", "failed", "cancelled", "expired"]:
            time.sleep(1)
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
        
        if run_status.status != "completed":
            logger.error(f"Run failed with status: {run_status.status}")
            await send_telegram_message(chat_id, "Sorry, I encountered an error processing your request.")
            return
        
        # Get the assistant's messages
        messages = client.beta.threads.messages.list(
            thread_id=thread_id
        )
        
        # Find the most recent assistant message
        assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
        if not assistant_messages:
            await send_telegram_message(chat_id, "Sorry, I couldn't generate a response.")
            return
        
        # Get the content of the most recent message
        latest_message = assistant_messages[0]
        response_text = ""
        
        for content_part in latest_message.content:
            if content_part.type == "text":
                response_text += content_part.text.value
        
        # Store the assistant's response
        store_message(chat_id, "assistant", response_text)
        
        # Send the assistant's response back to the user
        await send_telegram_message(chat_id, response_text)
        
    except Exception as e:
        logger.exception(f"Error processing message: {e}")
        await send_telegram_message(chat_id, "Sorry, an error occurred while processing your message.")


@app.post("/telegram-webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        update = await request.json()

        if "message" not in update:
            return JSONResponse(content={"status": "No message in update"})

        message = update["message"]
        chat_id = message["chat"]["id"]
        chat = message["chat"]
        username = chat.get("username")
        first_name = chat.get("first_name")
        last_name = chat.get("last_name")

        # Обработка текстовых сообщений
        if "text" in message:
            text = message["text"]
            background_tasks.add_task(
                process_message,
                chat_id,
                text,
                None,
                username,
                first_name,
                last_name
            )

        # Обработка файлов
        elif "document" in message or "photo" in message:
            # Получаем информацию о файле
            if "document" in message:
                file_info = message["document"]
                mime_type = file_info.get("mime_type")
            else:
                file_info = message["photo"][-1]  # Берем самую большую фотографию
                mime_type = "image/jpeg"

            # Проверяем поддержку формата
            if mime_type not in SUPPORTED_FILE_TYPES:
                await send_telegram_message(chat_id, "❌ Unsupported file type")
                return JSONResponse(content={"status": "unsupported_file_type"})

            # Проверяем размер файла
            if file_info.get("file_size", 0) > MAX_FILE_SIZE:
                await send_telegram_message(chat_id, "❌ File is too large (max 15MB)")
                return JSONResponse(content={"status": "file_too_large"})

            # Скачиваем файл
            file_content = await download_telegram_file(file_info["file_id"])

            # Извлекаем текст
            file_type = SUPPORTED_FILE_TYPES[mime_type]
            extracted_text = extract_text_from_file(file_content, file_type)

            if not extracted_text:
                await send_telegram_message(chat_id, "❌ Could not extract text from file")
                return JSONResponse(content={"status": "extraction_error"})

            # Обрабатываем извлеченный текст
            background_tasks.add_task(
                process_message,
                chat_id,
                None,
                extracted_text,
                username,
                first_name,
                last_name
            )

        else:
            await send_telegram_message(chat_id, "❌ I can only process text and files")
            return JSONResponse(content={"status": "unsupported_content"})

        return JSONResponse(content={"status": "processing"})

    except Exception as e:
        logger.exception(f"Error handling webhook: {e}")
        return JSONResponse(
            content={"status": "error", "detail": str(e)},
            status_code=500
        )
@app.get("/users")
async def list_users():
    """List all users (admin endpoint)"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT chat_id, username, first_name, last_name, 
                   created_at, last_interaction 
            FROM users
            ORDER BY last_interaction DESC
        """)
        users = cursor.fetchall()
        
        return {"users": list(users)}

@app.get("/messages")
async def get_all_messages(limit: int = 100):
    """Get recent message history for all users (admin endpoint)"""
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT m.id, m.chat_id, m.username, m.first_name, m.last_name, 
                   m.role, m.content, m.timestamp
            FROM messages m
            ORDER BY m.timestamp DESC
            LIMIT %s
        """, (limit,))
        messages = cursor.fetchall()
        
        return {"messages": list(messages)}

@app.get("/thread")
async def get_thread_info():
    """Get information about the shared thread (admin endpoint)"""
    thread_id = get_or_create_shared_thread()
    
    # Get messages from OpenAI
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    
    # Format messages for response
    formatted_messages = []
    for msg in messages.data:
        content_text = ""
        for content_part in msg.content:
            if content_part.type == "text":
                content_text += content_part.text.value
        
        formatted_messages.append({
            "id": msg.id,
            "role": msg.role,
            "content": content_text,
            "created_at": msg.created_at
        })
    
    return {
        "thread_id": thread_id,
        "messages": formatted_messages
    }

@app.post("/reset-thread")
async def reset_thread():
    """Reset the shared thread (admin endpoint)"""
    global SHARED_THREAD_ID
    
    # Create a new thread
    thread = client.beta.threads.create()
    new_thread_id = thread.id
    
    # Update the database
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO shared_thread (thread_id) VALUES (%s)",
            (new_thread_id,)
        )
        conn.commit()
    
    # Update the global variable
    SHARED_THREAD_ID = new_thread_id
    
    return {"status": "success", "message": "Thread reset successfully", "new_thread_id": new_thread_id}

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)