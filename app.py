import streamlit as st
from src.retriever import Retriever
from src.generator import Generator
import os

st.title("Amlgo Labs RAG Chatbot")
st.write("Ask questions about the provided document (e.g., Terms & Conditions).")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Chatbot Info")
    chunk_count = len(os.listdir('chunks')) if os.path.exists(r'chunks\chunks.pkl') else 0
    st.write(f"**Model**: Mistral-7B-Instruct-v0.3 (Ollama)")
    st.write(f"**Indexed Chunks**: {chunk_count}")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

retriever = Retriever(r"vectordb\faiss_index.bin", r"chunks\chunks.pkl")
generator = Generator()

user_input = st.text_input("Your question:", key="user_input")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    retrieved_chunks, _ = retriever.retrieve(user_input)
    response_container = st.empty()
    with st.spinner("Generating response..."):
        response, sources = generator.generate(user_input, retrieved_chunks)
        response_container.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.messages.append({"role": "sources", "content": sources})

for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"**You**: {message['content']}")
    elif message["role"] == "assistant":
        st.write(f"**Bot**: {message['content']}")
    elif message["role"] == "sources":
        with st.expander("Source Chunks"):
            for idx, chunk in enumerate(message["content"], 1):
                st.write(f"**Chunk {idx}**: {chunk}")