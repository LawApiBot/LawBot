import os
from typing import Dict, Optional
import logging
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv() 

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

# In-memory storage for user threads (in production, use a database)
user_threads: Dict[int, str] = {}

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

async def process_message(chat_id: int, text: str):
    """Process a message from a Telegram user"""
    try:
        # Get or create a thread for this user
        thread_id = user_threads.get(chat_id)
        if not thread_id:
            thread = client.beta.threads.create()
            thread_id = thread.id
            user_threads[chat_id] = thread_id
            logger.info(f"Created new thread {thread_id} for user {chat_id}")
        
        # Send typing indicator
        await send_chat_action(chat_id, "typing")
        
        # Add the user's message to the thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=text
        )
        
        # Run the Assistant on the thread
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
        
        # Send the assistant's response back to the user
        await send_telegram_message(chat_id, response_text)
        
    except Exception as e:
        logger.exception(f"Error processing message: {e}")
        await send_telegram_message(chat_id, "Sorry, an error occurred while processing your message.")

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming webhook requests from Telegram"""
    try:
        update = await request.json()
        
        # Check if this is a message update
        if "message" not in update:
            return JSONResponse(content={"status": "No message in update"})
        
        message = update["message"]
        chat_id = message["chat"]["id"]
        
        # Check if the message contains text
        if "text" not in message:
            await send_telegram_message(chat_id, "I can only process text messages.")
            return JSONResponse(content={"status": "No text in message"})
        
        text = message["text"]
        
        # Process the message in the background
        background_tasks.add_task(process_message, chat_id, text)
        
        return JSONResponse(content={"status": "processing"})
        
    except Exception as e:
        logger.exception(f"Error handling webhook: {e}")
        return JSONResponse(
            content={"status": "error", "detail": str(e)},
            status_code=500
        )

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)