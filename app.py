import os
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai


# =========================================================
# 1. LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    print("Loading Embedding Model...")
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# =========================================================
# 2. LOAD VECTOR DATABASE
# =========================================================

@st.cache_resource
def load_or_create_vector_db():

    print("Loading Vector Database...")

    index = faiss.read_index(
        "vector_db/faiss_index.bin"
    )

    chunks = np.load(
        "vector_db/chunks.npy",
        allow_pickle=True
    ).tolist()

    chunk_sources = np.load(
        "vector_db/chunk_sources.npy",
        allow_pickle=True
    ).tolist()

    print("Vector Database Loaded!")
    print("Vectors Stored:", index.ntotal)

    return index, chunks, chunk_sources


# =========================================================
# 3. CREATE CONTEXTUAL QUERY
# =========================================================

def create_contextual_query(question, chat_history):

    """
    Converts a follow-up question into a more complete
    question using previous conversation.
    """

    if not chat_history:
        return question

    # Take recent conversation only
    recent_history = chat_history[-6:]

    history_text = ""

    for message in recent_history:

        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            history_text += f"User: {content}\n"

        elif role == "assistant":
            history_text += f"Assistant: {content}\n"

    contextual_query = f"""
Conversation history:

{history_text}

Current user question:
{question}

Rewrite the current question as a standalone search query.

Rules:
- Keep the original meaning.
- Use previous conversation only when necessary.
- Resolve words such as "it", "this", "that", "its", etc.
- Do not answer the question.
- Return only the rewritten search query.

Standalone search query:
"""

    try:

        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return question

        genai.configure(
            api_key=api_key
        )

        query_model = genai.GenerativeModel(
            "gemini-3.1-flash-lite"
        )

        response = query_model.generate_content(
            contextual_query,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 100
            },
            request_options={
                "timeout": 60
            }
        )

        if response.text:
            return response.text.strip()

        return question

    except Exception:

        # If query rewriting fails,
        # use the original question.
        return question


# =========================================================
# 4. RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(
    question,
    index,
    chunks,
    chunk_sources,
    chat_history=None
):

    # Create contextual search query
    search_query = create_contextual_query(
        question,
        chat_history or []
    )

    print("Search Query:", search_query)

    # Convert question into embedding
    question_embedding = model.encode(
        [search_query],
        convert_to_numpy=True
    ).astype(np.float32)

    # Search top 10 chunks
    distances, indices = index.search(
        question_embedding,
        k=10
    )

    selected_chunks = []
    used_sources = set()

    # Select top 5 chunks
    for idx in indices[0]:

        if idx < 0:
            continue

        chunk_text = chunks[idx]

        selected_chunks.append(
            (idx, chunk_text)
        )

        used_sources.add(
            chunk_sources[idx]
        )

        if len(selected_chunks) == 5:
            break

    # Combine chunks
    retrieved_text = "\n\n".join(
        chunk_text
        for _, chunk_text in selected_chunks
    )

    return retrieved_text, used_sources


# =========================================================
# 5. GENERATE CHATBOT ANSWER
# =========================================================

def generate_answer(
    question,
    retrieved_text,
    chat_history=None
):

    load_dotenv()

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        return (
            "GEMINI_API_KEY is not configured."
        )

    # Configure Gemini
    genai.configure(
        api_key=api_key
    )

    # Gemini 3.1 Flash-Lite
    model_gemini = genai.GenerativeModel(
        "gemini-3.1-flash-lite"
    )

    # -----------------------------------------------------
    # Conversation history
    # -----------------------------------------------------

    history_text = ""

    if chat_history:

        recent_history = chat_history[-8:]

        for message in recent_history:

            role = message.get("role", "")
            content = message.get("content", "")

            if role == "user":

                history_text += (
                    f"User: {content}\n"
                )

            elif role == "assistant":

                history_text += (
                    f"Assistant: {content}\n"
                )

    # -----------------------------------------------------
    # Limit retrieved context
    # -----------------------------------------------------

    context = retrieved_text[:15000]

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are a Government Scheme Assistant.

You help users understand government schemes using
information from uploaded government scheme documents.

Your answer must be based ONLY on the provided context.

IMPORTANT RULES:

1. Understand the user's exact question.
2. Answer the question directly.
3. Use conversation history to understand follow-up questions.
4. Do not give a generic list of schemes unless the user
   specifically asks for a list.
5. Do not say:
   "Based on the provided context"
   or
   "According to the context".
6. Do not mention:
   - RAG
   - FAISS
   - embeddings
   - vectors
   - chunks
   - retrieval
   - internal processing
7. Do not use outside knowledge.
8. Do not invent missing information.
9. If the requested information is not available in the
   provided context, reply:

Information not found in uploaded documents.

10. Keep the answer clear and useful.
11. Use headings and bullet points when appropriate.
12. For a specific scheme, provide the available details such as:
    - Scheme name
    - Objective
    - Benefits
    - Eligibility
    - Financial assistance
    - Loan/scholarship amount
    - Application process
    - Important conditions
13. Do not mention information that is not present in the context.

CONVERSATION HISTORY:

{history_text}

DOCUMENT CONTEXT:

{context}

CURRENT USER QUESTION:

{question}

Now answer the user's question directly.
"""

    try:

        response = model_gemini.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 700
            },
            request_options={
                "timeout": 120
            }
        )

        if response.text:

            return response.text.strip()

        return (
            "Information not found in uploaded documents."
        )

    except Exception as e:

        return (
            f"Error generating answer: {str(e)}"
        )


# =========================================================
# 6. TERMINAL TEST
# =========================================================

def main():

    index, chunks, chunk_sources = (
        load_or_create_vector_db()
    )

    question = input(
        "\nAsk your question: "
    )

    retrieved_text, used_sources = retrieve_chunks(
        question,
        index,
        chunks,
        chunk_sources,
        chat_history=[]
    )

    answer = generate_answer(
        question,
        retrieved_text,
        chat_history=[]
    )

    print("\n==============================")
    print("===== FINAL ANSWER =====")
    print("==============================\n")

    print(answer)

    print("\n==============================")
    print("===== SOURCES USED =====")
    print("==============================\n")

    for source in used_sources:
        print(source)


# =========================================================
# 7. RUN
# =========================================================

if __name__ == "__main__":
    main()