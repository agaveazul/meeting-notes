from main import run_llm
import streamlit as st
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models import ChatOpenAI

st.header("Meeting Notes & Action Items")

openai_api_key = st.sidebar.text_input(
    label="OpenAI API Key",
    type="password",
)

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

chat = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-4"
)

if 'prompt' not in st.session_state:
    st.session_state['prompt'] = ''
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []

prompt = st.text_input("What would you like to summarize today?", key='prompt', value=st.session_state['prompt'])

msgs = StreamlitChatMessageHistory()

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("What would you like to summarize today?")
    st.session_state.steps = {}

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, history=st.session_state["history"], msgs=msgs, chat=chat
        )
        print(generated_response)
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(generated_response)
        st.session_state["history"].append(HumanMessage(content=prompt))
        st.session_state["history"].append(SystemMessage(content=generated_response))

if st.session_state["chat_answers_history"]:
    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            # Render intermediate steps if any were saved
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                    st.write(step[0].log)
                    st.write(step[1])
            st.write(msg.content)

