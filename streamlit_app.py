import streamlit as st
from app import (
    load_or_create_vector_db,
    retrieve_chunks,
    generate_answer
)

# Page Config
st.set_page_config(
    page_title="Government Scheme Assistant",
    page_icon="🤖",
    layout="wide"
)

# Header
st.title("🤖 Government Scheme Assistant")
st.caption("AI Powered RAG Chatbot using FAISS + Gemini")

# Load DB only once
@st.cache_resource
def load_database():
    return load_or_create_vector_db()

index, chunks, chunk_sources = load_database()

# Sidebar
with st.sidebar:

    st.header("📊 System Info")

    st.metric(
        "Total Chunks",
        len(chunks)
    )

    st.metric(
        "Sources",
        len(set(chunk_sources))
    )

    st.success("FAISS Loaded")
    st.success("Gemini Connected")

# User Question
question = st.text_input(
    "Ask a Question",
    placeholder="Example: What is PMMY?"
)

# Search Button
if st.button("🔍 Get Answer"):

    if question.strip():

        with st.spinner("Searching documents..."):

            retrieved_text, used_sources = retrieve_chunks(
                question,
                index,
                chunks,
                chunk_sources
            )

            answer = generate_answer(
                question,
                retrieved_text
            )

        # Answer Section
        st.subheader("📌 Answer")
        st.write(answer)

        # Sources Section
        st.subheader("📚 Sources")

        for source in used_sources:
            st.success(source)

    else:
        st.warning("Please enter a question.")