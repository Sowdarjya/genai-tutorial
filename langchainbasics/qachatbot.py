import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
import os

st.set_page_config(page_title="Groq Chatbot", page_icon="💣")

st.title("Groq Chatbot")
st.markdown("This is a simple chatbot application using Groq's LLM.")

with st.sidebar:
    st.title("Settings")
    groq_api_key = st.text_input(
        "Groq API Key", type="password", help="Enter your Groq API key here.")

    model_name = st.selectbox(
        "Select Model",
        ["llama3-8b-8192", "gemma2-9b-it"],
        index=0,
        help="Choose the model you want to use for the chatbot."
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def get_chain(api_key, model_name):
    if not api_key:
        return None

    llm = ChatGroq(groq_api_key=api_key, model_name=model_name,
                   temperature=0.7, streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant powered by Groq. Answer questions clearly and concisely."),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    return chain


chain = get_chain(groq_api_key, model_name)

if not chain:
    st.error("Please enter your Groq API key to start.")

else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if question := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            full_response = ""
            message_placeholder = st.empty()

            try:
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "...")

                message_placeholder.markdown(full_response)

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")


st.markdown("---")
st.markdown("### 💡 Try these examples:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain Groq's LPU technology")
with col2:
    st.markdown("- How do I learn programming?")
    st.markdown("- Write a haiku about AI")

# Footer
st.markdown("---")
st.markdown("Built with LangChain & Groq | Experience the speed! ⚡")
