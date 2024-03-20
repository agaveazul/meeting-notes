from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from dotenv import load_dotenv
# from tools.sql import run_query_tool, list_tables, describe_tables_tool
from langchain_community.chat_models import ChatOpenAI
from typing import Any, List, Tuple
from langchain.chains import ConversationChain

# tables = list_tables()

def run_llm(query: str, msgs: StreamlitChatMessageHistory, chat: ChatOpenAI, history: List[Tuple[str,Any]]=[]) -> Any:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=(
                """
                You are an AI that summarizes meeting notes and provides action items.
                When you receive a transcript of a conversation, you provide the following: 
                List of attendees: all the attendees who were speaking in the transcript
                Key Takeaways: a high-level summary of the main topics discussed
                Action Items: a list of the actions necessary coming out of the meeting with the relevant owner
                Meeting Summary: a detailed description of the key takeaways
                Do not make up any information. If you do not know the answer, then leave the content blank.
                """
                )),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            # MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )
    memory = ConversationBufferMemory(memory_key="history", return_messages=True, chat_memory=msgs)

    conversation = ConversationChain(
        llm=chat,
        verbose=True,
        memory=memory,
        prompt=prompt
    )

    return conversation.predict(input=query)

    # tools = []

    # agent = OpenAIFunctionsAgent(
    #     llm=chat,
    #     prompt=prompt,
    #     tools=tools
    # )

    # agent_executor = AgentExecutor(
    #     agent=agent,
    #     verbose=True,
    #     tools=tools,
    #     memory=memory
    # )

    # return agent_executor(query)