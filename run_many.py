import logging
from collections import defaultdict
from typing import Dict, List, Optional
import asyncio
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from telegram.ext import filters, MessageHandler
from phi.agent import Agent
from phi.model.google.gemini import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from phi.embedder.google import GeminiEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.model.message import Message

from utils import gchat, get_agent_response, MessageHistory, start

# Configuration
bot1_name = 'Rachel'
bot2_name = 'Nick'
moderator_name = 'Modi'

TOKENS = {
    bot1_name: "",
    bot2_name: "",
    moderator_name: "",
}



bot1_sys_msg = [{"role": "system", "content": """You now act as Rachel Botsman, author of 
Key Work: What's Mine Is Yours: The Rise of Collaborative Consumption
Try to make the conversation flow. you can ask questions, and try to bring an interesting, eye-opening opinion. try to keep your answers short and concise.
When you answer, give the plain text, without formatting or your name before the text.
You are in the middle of a conversation with several people, here's what they (and you) wrote previously:"""}]

bot2_sys_msg = [{"role": "system", "content": """You now act as Nick Srnicek, author of 
Platform Capitalism. 
You are in the middle of a conversation with several people, 
Try to make the conversation flow. you can ask questions, and try to bring an interesting, eye-opening opinion. try to keep your answers short and concise.
When you answer, give the plain text, without formatting or your name before the text.  
here's what they (and you) wrote previously:"""}]

mod_sys_msg = [
    {"role": "system", "content": """You are a conversation moderator. Your goal is to read previous messages, and decide who will speak next. The conversation has both humans and bots. 
    you can select bots from the following list ['Rachel','Nick']. 
    Rules:
	If someone is addressed by name, he/she should speak. If the user was conversing with one of the agents, and now answers without explicitly stating who is he\she addressing, assume he\she still speaks to the same agent. 
	If nobody should speak (if a human is addressed for example), return a string 'no one'
	Your answer must strictly be one of the list above or 'no one'. no text before or after. 
You are in the middle of a conversation with several people, here's what they (and you) wrote previously:"""}]


mod_sys_msg2=Message(role="user", content="""\n\nREMEMBER: DO NOT CONTINUE THIS CONVERSATION. JUST SAY WHO NEEDS TO SPEAK NEXT according to the rules above""")
sys_msgs = {bot1_name: bot1_sys_msg, bot2_name: bot2_sys_msg}

bot_messages_cnt = 0

# Initialize knowledge bases
knowledge_base1 = PDFKnowledgeBase(
    path='/home/roy/Documents/lab_data/c',
    vector_db=LanceDb(
        table_name="collaborative_C",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=GeminiEmbedder(),
    ),
)

knowledge_base2 = PDFKnowledgeBase(
    path='/home/roy/Documents/lab_data/platform.pdf',
    vector_db=LanceDb(
        table_name="recipes",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=GeminiEmbedder(),
    ),
)

knowledge_base1.load()
knowledge_base2.load()

# Initialize bots and agents
bots: Dict[str, Bot] = {}
agents: Dict[str, Agent] = {}

agent1 = Agent(
    name=bot1_name,
    model=Gemini(model="gemini-2.0-flash"),
    tools=[DuckDuckGo()],
    instructions=["Act as Rachel Botsman"],
    show_tool_calls=False,
    knowledge=knowledge_base1,
    markdown=True,
)

agent2 = Agent(
    name=bot2_name,
    model=Gemini(model="gemini-2.0-flash"),
    tools=[DuckDuckGo()],
    instructions=["act as Nick Srnicek"],
    show_tool_calls=False,
    knowledge=knowledge_base2,
    markdown=True,
)

# Initialize message history
message_history = MessageHistory()
client=Gemini(model="gemini-2.0-flash")

async def empty_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # do nothing
    pass  

async def handle_mod_message(update: Update, context: ContextTypes.DEFAULT_TYPE, caller='user'):
    global bot_messages_cnt
    chat_id = update.effective_chat.id
    if caller == 'user':
        bot_messages_cnt = 0
        user_message = update.message.text
        user = update.message.from_user.first_name
        message_history.add_message(chat_id, "user", user, user_message)
    
    messages = message_history.get_history(chat_id)
    who_should_speak = gchat(client, mod_sys_msg, messages + [mod_sys_msg2])
    who_should_speak = who_should_speak.strip()
    if bot_messages_cnt > 6:
        logging.info('bot messages num exceeded')
        who_should_speak = 'no one'
    
    logging.info(f'who_should_speak:{who_should_speak}')
    logging.info(context)
    
    try:
        if who_should_speak == 'Rachel':
            response = await handle_bot_message(messages, context, chat_id, bot1_name)
            logging.info(f"Returned from handle_bot_message for {bot1_name}")
        elif who_should_speak == 'Nick':
            response = await handle_bot_message(messages, context, chat_id, bot2_name)
            logging.info(f"Returned from handle_bot_message for {bot2_name}")
        else:
            logging.info(f"No matching bot, returning for {who_should_speak}")
            return
            
        logging.info("Adding response to history")
        message_history.add_message(chat_id, "assistant", who_should_speak, response.content)
        logging.info("=== Mod Message Handler Complete ===\n")
        await handle_mod_message(update, context, caller='bot')
    except Exception as e:
        logging.error(f"Error in handle_mod_message: {e}")
        import traceback
        logging.error(traceback.format_exc())

async def handle_bot_message(messages, context, chat_id, bot):
    global bot_messages_cnt
    logging.info(f"Starting handler with chat_id: {chat_id}")
    logging.info(f"Context bot username: {context.bot.username if context and context.bot else 'No context bot'}")
    logging.info(f"Bots[bot] username: {bots[bot].username if bot in bots else 'No such bot'}")

    sys_msg = sys_msgs[bot]
    agent = agents[bot]
    response = await get_agent_response(agent, sys_msg, messages)
    
    logging.info(response.content)
    logging.info(response)
    logging.info(str({bots[bot]}))
    bot_info = await bots[bot].get_me()
    logging.info(f"Bot info: {bot_info}")
    bot_messages_cnt += 1
    
    try:
        out = await bots[bot].send_message(
            chat_id=chat_id,
            text=response.content
        )
        logging.info('sent')
        logging.info(str({bots[bot]}))
        bot_info = await bots[bot].get_me()
        logging.info(f"Bot info: {bot_info}")
        logging.info(out)
    except Exception as e:
        print(f"Error sending message: {e}")
        print(f"chat_id: {chat_id}")
        print(f"bot object exists: {bot in bots}")
        print(f"response length: {len(response)}")
    return response

def create_bot_application(token: str, bot_number: int):
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler('start', start))
    
    if bot_number == 1:
        handler = MessageHandler(filters.TEXT & (~filters.COMMAND), empty_handle_message)
    elif bot_number == 2:
        handler = MessageHandler(filters.TEXT & (~filters.COMMAND), empty_handle_message)
    elif bot_number == 3:
        handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_mod_message)

    app.add_handler(handler)
    return app

async def main():
    applications = []
    logging.info(len(TOKENS))
    agents[bot1_name] = agent1
    agents[bot2_name] = agent2
    
    for i, (bot_name, token) in enumerate(TOKENS.items(), 1):
        app = create_bot_application(token, i)
        applications.append(app)
        logging.info(f"Starting {bot_name}")
        bots[bot_name] = app.bot

    logging.info('done init')
    logging.info(str(bots))
    
    try:
        await asyncio.gather(*(app.initialize() for app in applications))
        await asyncio.gather(*(app.start() for app in applications))
        await asyncio.gather(*(app.updater.start_polling() for app in applications))
        
        stop = asyncio.Event()
        await stop.wait()

    except Exception as e:
        logging.error(f"Error running bots: {e}")

    finally:
        await asyncio.gather(*(app.stop() for app in applications))
        await asyncio.gather(*(app.shutdown() for app in applications))

if __name__ == '__main__':
    asyncio.run(main()) 