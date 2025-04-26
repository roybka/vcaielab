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
from phi.model.google.gemini import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.message import Message

from phi.embedder.google import GeminiEmbedder
from phi.embedder.openai import OpenAIEmbedder
from phi.knowledge.pdf import PDFUrlKnowledgeBase,PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from utils import gchat, get_agent_response, MessageHistory, start



bot_token = '7888726289:AAG0F9b-PHfBtYNgMW1G-WUc_D2cticoKogfdafsd'
knowledge_base_path = '' # path to a folder containing the knowledge base pdfs (empty = no knowledge base)
bot_name = 'rachel'

bot_sys_msg = [{"role": "system", "content": """You now act as Rachel Botsman, author of 
What's Mine Is Yours: The Rise of Collaborative Consumption
Try to make the conversation flow. you can ask questions, and try to bring an interesting, eye-opening opinion. try to keep your answers short and concise.
When you answer, give the plain text, without formatting or your name before the text.
You are in the middle of a conversation with several people, here's what they (and you) wrote previously:"""}]


# this is the knowledge base for the bot
if knowledge_base_path:
    knowledge_base = PDFKnowledgeBase(
    path=knowledge_base_path,
    # Use LanceDB as the vector database
    vector_db=LanceDb(
        table_name="give_some_name",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=GeminiEmbedder(),
    ),
    )
    knowledge_base.load()



bots: Dict[str, Bot] = {}
agents: Dict[str,Agent]={}
agent = Agent(
    name=bot_name,
    model=Gemini(model="gemini-2.0-flash"),
    tools=[DuckDuckGo()], # This agent can use search, see more about tools here: https://docs.phidata.com/tools/introduction
    instructions=["Act as Rachel Botsman"],
    show_tool_calls=False,
    knowledge=None if not knowledge_base_path else knowledge_base,
    markdown=True,
)
message_history = MessageHistory()


async def handle_bot_message(update: Update, context: ContextTypes.DEFAULT_TYPE,caller='user'):
    global bot_messages_cnt
    chat_id = update.effective_chat.id
    user_message = update.message.text
    user = update.message.from_user.first_name
    # Add user message to history
    message_history.add_message(chat_id, "user", user, user_message)
    messages = message_history.get_history(chat_id)
    response=await get_agent_response(agent,bot_sys_msg,messages)
    bot_message = response.content
    out = await bots['rachel'].send_message(
            chat_id=chat_id,
            text=bot_message
        )
    return response


def create_bot_application(token: str):
    app = ApplicationBuilder().token(token).build() # telegram app builder
    # Add start command handler
    app.add_handler(CommandHandler('start', start)) # Add start command handler
    # Add message handler - the function that will handle the messages
    handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_bot_message) 
    app.add_handler(handler)
    return app


async def main():
    # Create and store all bot applications
    app = create_bot_application(bot_token)
    bots[bot_name] = app.bot
    logging.info('done init')

    try:
        # Initialize and start all applications
        await asyncio.gather(app.initialize())
        await asyncio.gather(app.start())

        # Run all applications
        await asyncio.gather(app.updater.start_polling())

        # Keep the script running
        stop = asyncio.Event()
        await stop.wait()

    except Exception as e:
        logging.error(f"Error running bots: {e}")

    finally:
        # Properly shut down all applications
        await asyncio.gather(app.stop())
        await asyncio.gather(app.shutdown())


if __name__ == '__main__':
    asyncio.run(main())

