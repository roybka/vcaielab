import logging
from collections import defaultdict
from typing import Dict, List, Optional
import asyncio
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from telegram.ext import filters, MessageHandler
from openai import OpenAI
from phi.agent import Agent
from phi.model.openai import OpenAIChat
# from phi.model.google import GoogleChat
from phi.tools.duckduckgo import DuckDuckGo

# from phi.embedder.google import GoogleEmbedder
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase,PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.model.message import Message


def gchat(client, sysmsg:Message, messages:List[Message]):
    if isinstance(sysmsg, list) and isinstance(sysmsg[0], dict):
        sysmsg = [Message(role=msg["role"], content=msg["content"]) for msg in sysmsg]
    
    # client=Gemini(model="gemini-2.0-flash")
    out=client.invoke(messages=sysmsg+messages)
    return out.candidates[0].content.parts[0].text


def chat(client, sysmsg, messages):
    # send a chat completion request to openai, return the response
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=sysmsg + messages
    )
    return completion.choices[0].message.content

# Bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Hello!"
    )




# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def get_agent_response(agent, sysmsg, messages):
    # Convert system messages to Message objects if they're in dict format
    if isinstance(sysmsg, list) and isinstance(sysmsg[0], dict):
        sysmsg = [Message(role=msg["role"], content=msg["content"]) for msg in sysmsg]
    
    # Combine system messages with conversation messages
    all_messages = sysmsg + messages
    out = agent.run(messages=all_messages)
    
    return out

# Message history storage
class MessageHistory:
    def __init__(self, max_messages: int = 50):
        self.max_messages = max_messages
        self.history: Dict[int, List[Message]] = defaultdict(list)

    def add_message(self, chat_id: int, role: str, speaker: str, content: str):
        if len(self.history[chat_id]) >= self.max_messages:
            self.history[chat_id].pop(0)
        self.history[chat_id].append(Message(role=role, content=f"{speaker}: {content}"))

    def get_history(self, chat_id: int) -> List[Message]:
        return self.history[chat_id]

    def get_last_n_messages(self, chat_id: int, n: int = 2) -> List[Message]:
        return self.history[chat_id][-n:] if self.history[chat_id] else []
