import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"


st.set_page_config(page_title="AI Agent Copilot", layout="wide")
st.title("AI Agent Copilot")
st.write("Upload a PDF, ask questions, and get grounded answers with sources.")


# -------------------------
# Upload Section
# -------------------------
st.header("Upload Document")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Upload and Index PDF"):
        with st.spinner("Uploading and indexing document..."):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
            }

            try:
                response = requests.post(f"{API_BASE_URL}/upload", files=files)

                if response.status_code == 200:
                    result = response.json()
                    st.success("File uploaded and indexed successfully.")
                    st.json(result)
                else:
                    st.error(f"Upload failed: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Make sure the API server is running.")


# -------------------------
# Query Section
# -------------------------
st.header("Ask a Question")

question = st.text_input("Enter your question")
top_k = st.slider("Number of retrieved chunks", min_value=1, max_value=5, value=3)

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

                    st.subheader("Answer")
                    st.write(result.get("answer", "No answer returned."))

                    st.subheader("Sources")
                    sources = result.get("sources", [])

                    if sources:
                        for i, src in enumerate(sources, start=1):
                            st.markdown(
                                f"**Source {i}:** File: `{src.get('source')}` | "
                                f"Page: `{src.get('page_number')}` | "
                                f"Chunk ID: `{src.get('chunk_id')}`"
                            )
                    else:
                        st.info("No sources returned.")

                else:
                    st.error(f"Query failed: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend. Make sure the API server is running.")