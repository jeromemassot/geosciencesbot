from utils.retriever_reader_tools import init_guardrails

import google.generativeai as palm
import pinecone
import openai

from PIL import Image

import streamlit as st


@st.cache_resource()
def cached_init_guardrails():
    return init_guardrails()

@st.cache_resource()
def cached_setup():
    """
    Setup the API keys
    """
    palm.configure(api_key=st.secrets["PALM_API_KEY"])
    pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment=st.secrets["PINECONE_ENV_ZONE"])
    openai.api_key = st.secrets["OPENAI_API_KEY"]

# setup the environment
cached_setup()

# setup the guardrails
rails = cached_init_guardrails()

# User Interface

st.title("💬 Chatbot") 
st.image(Image.open("./decorations/tech_stack.jpg"), width=300)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant", 
            "content": "How can I help you? I know about well drilling, mineral resources in the US."
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = rails.generate(prompt=prompt)
    if type(response) == tuple:
        response = response[0]
    msg = {'role': 'assistant', 'content': response}
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg["content"])

