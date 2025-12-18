import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import OllamaAgent
from src.tools import tools


@cl.on_chat_start
async def start():
    # Initialize agent with tools
    agent = OllamaAgent(model="gpt-oss:20b", tools=tools, temperature=0.3)

    # Store agent in session
    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    chat_history = cl.user_session.get("chat_history", [])

    # Stream the agent's response
    msg = cl.Message(content="")
    await msg.send()

    full_response = ""
    async for chunk in agent.astream(message.content, chat_history):
        if chunk:
            full_response += chunk
            await msg.stream_token(chunk)

    await msg.update()

    # Update chat history with proper message objects
    chat_history.append(HumanMessage(content=message.content))
    chat_history.append(AIMessage(content=full_response))
    cl.user_session.set("chat_history", chat_history)
