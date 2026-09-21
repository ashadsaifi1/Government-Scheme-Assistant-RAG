import streamlit as st

from app import (
    load_or_create_vector_db,
    retrieve_chunks,
    generate_answer
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Government Scheme Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 Government Scheme Assistant")
st.caption("AI-powered RAG chatbot using FAISS + Gemini")


# =========================================================
# LOAD DATABASE
# =========================================================

index, chunks, chunk_sources = load_or_create_vector_db()


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

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
    st.success("Gemini Ready")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# DISPLAY PREVIOUS CHAT
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        # Show sources only for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 Sources"):

                for source in message["sources"]:

                    st.write(f"• {source}")


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask about a government scheme..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # Show user message
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🔍 Searching documents..."):

            retrieved_text, used_sources = retrieve_chunks(
                question,
                index,
                chunks,
                chunk_sources,
                chat_history=st.session_state.messages
            )

        with st.spinner("🤖 Generating answer..."):

            answer = generate_answer(
                question,
                retrieved_text,
                chat_history=st.session_state.messages
            )

        st.markdown(answer)

        # -------------------------------------------------
        # Sources
        # -------------------------------------------------

        if used_sources:

            with st.expander("📚 Sources"):

                for source in used_sources:

                    st.write(f"• {source}")

    # -----------------------------------------------------
    # Save assistant response
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": list(used_sources)
        }
    )