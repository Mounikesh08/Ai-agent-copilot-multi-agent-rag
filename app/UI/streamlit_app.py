import sys
import os
from pathlib import Path

import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI Agent Copilot", layout="wide")

st.title("AI Agent Copilot")
st.caption("Upload PDFs, ask questions, and get grounded answers with sources.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Upload Documents")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Upload and Process"):
            with st.spinner("Uploading and indexing document..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }

                try:
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"{uploaded_file.name} uploaded successfully.")
                        st.json(result.get("indexing_result", {}))
                    else:
                        st.error(f"Upload failed: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to backend. Make sure FastAPI is running.")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")

st.header("Ask Questions")

question = st.text_input("Enter your question")
top_k = st.slider("Number of retrieved chunks", 1, 5, 3)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving answer..."):
            payload = {
                "question": question,
                "top_k": top_k
            }

            try:
                response = requests.post(f"{API_BASE_URL}/query", json=payload)

                if response.status_code == 200:
                    result = response.json()

                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": result.get("answer", "No answer returned."),
                            "sources": result.get("sources", []),
                        }
                    )
                else:
                    st.error(f"Query failed: {response.text}")
                    st.write("Status code:", response.status_code)
                    st.write("Raw response:", response.text)

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Make sure FastAPI is running.")

for chat in reversed(st.session_state.chat_history):
    st.subheader("Question")
    st.write(chat["question"])

    st.subheader("Answer")
    st.write(chat["answer"])

    st.subheader("Sources")

    if chat["sources"]:
        for i, src in enumerate(chat["sources"], start=1):
            st.markdown(
                f"- **Source {i}**: `{src.get('source')}` | Page `{src.get('page_number')}` | Chunk `{src.get('chunk_id')}`"
            )
    else:
        st.info("No sources returned.")

    st.markdown("---")