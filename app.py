# =========================
# IMPORT LIBRARIES
# =========================

import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import google.generativeai as genai


# =========================
# CONSTANTS
# =========================

VECTOR_DB_PATH = "vector_db/faiss_index.bin"
CHUNKS_PATH = "vector_db/chunks.npy"
SOURCES_PATH = "vector_db/chunk_sources.npy"

PDF_FOLDER = "data"


# =========================
# TEXT SPLITTER
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# =========================
# LOAD EMBEDDING MODEL
# =========================

print("Loading Embedding Model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================
# CREATE OR LOAD VECTOR DB
# =========================

def load_or_create_vector_db():

    '''# Load existing vector database
    if (
        os.path.exists(VECTOR_DB_PATH)
        and os.path.exists(CHUNKS_PATH)
        and os.path.exists(SOURCES_PATH)
    ):

        print("\nLoading Existing Vector Database...")

        index = faiss.read_index(
            VECTOR_DB_PATH
        )

        chunks = np.load(
            CHUNKS_PATH,
            allow_pickle=True
        ).tolist()

        chunk_sources = np.load(
            SOURCES_PATH,
            allow_pickle=True
        ).tolist()

        print("Vector Database Loaded!")
        print("Vectors Stored:", index.ntotal)

        return index, chunks, chunk_sources'''

    # Create new vector database
    print("\nCreating New Vector Database...")

    chunks = []
    chunk_sources = []

    for file in os.listdir(PDF_FOLDER):

        if file.endswith(".pdf"):

            print(f"Reading: {file}")

            pdf_path = os.path.join(
                PDF_FOLDER,
                file
            )

            reader = PdfReader(pdf_path)

            pdf_text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    pdf_text += page_text + "\n"

            pdf_chunks = splitter.split_text(
                pdf_text
            )

            for chunk in pdf_chunks:

                chunks.append(chunk)
                chunk_sources.append(file)

    print(f"\nTotal Chunks: {len(chunks)}")

    embeddings = model.encode(
        chunks
    )

    print(
        "Embedding Shape:",
        embeddings.shape
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        np.array(
            embeddings,
            dtype=np.float32
        )
    )

    print(
        "Vectors Stored:",
        index.ntotal
    )

   ''' # Save FAISS
    faiss.write_index(
        index,
        VECTOR_DB_PATH
    )

    # Save chunks
    np.save(
        CHUNKS_PATH,
        np.array(
            chunks,
            dtype=object
        )
    )

    # Save sources
    np.save(
        SOURCES_PATH,
        np.array(
            chunk_sources,
            dtype=object
        )
    )

    print("\nVector Database Saved!")'''

    return index, chunks, chunk_sources
# =========================
# RETRIEVE RELEVANT CHUNKS
# =========================

def retrieve_chunks(
    question,
    index,
    chunks,
    chunk_sources
):

    question_embedding = model.encode(
        [question]
    )

    # Pehle zyada chunks retrieve karo
    distances, indices = index.search(
        np.array(
            question_embedding,
            dtype=np.float32
        ),
        k=15
    )

    retrieved_text = ""
    used_sources = set()

    # Question ke keywords nikalo
    keywords = [
        word.lower()
        for word in question.split()
        if len(word) > 3
    ]

    selected_chunks = []

    # Hybrid Search
    for idx in indices[0]:

        chunk_text = chunks[idx]

        chunk_lower = chunk_text.lower()

        keyword_match = any(
            keyword in chunk_lower
            for keyword in keywords
        )

        if keyword_match:
            selected_chunks.append(
                (idx, chunk_text)
            )

    # Agar keyword match na mile
    if len(selected_chunks) == 0:

        for idx in indices[0][:5]:

            selected_chunks.append(
                (idx, chunks[idx])
            )

    # Sirf top 5 chunks Gemini ko bhejo
    selected_chunks = selected_chunks[:5]

    for idx, chunk_text in selected_chunks:

        retrieved_text += chunk_text
        retrieved_text += "\n\n"

        used_sources.add(
            chunk_sources[idx]
        )

    return retrieved_text, used_sources


# =========================
# GEMINI ANSWER GENERATION
# =========================

def generate_answer(
    question,
    retrieved_text
):

    load_dotenv()

    genai.configure(
        api_key=os.getenv(
            "GEMINI_API_KEY"
        )
    )

    model_gemini = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    prompt = f"""
You are a Government Scheme Assistant.

Answer ONLY using the provided context.

Rules:
1. Do not use outside knowledge.
2. If answer is missing, reply exactly:
   Information not found in uploaded documents.
3. Keep the answer factual and clear.
4. If enough information is available, explain in 3-5 sentences.
5. Include objective, benefits, eligibility, scholarship amount or other important details if present in the context.
6. Do not mention chunks, vectors, embeddings, retrieval process or internal processing.

Context:
{retrieved_text}

Question:
{question}
"""
    response = model_gemini.generate_content(
        prompt
    )

    return response.text


# =========================
# MAIN FUNCTION
# =========================

def main():

    index, chunks, chunk_sources = (
        load_or_create_vector_db()
    )

    question = input(
        "\nAsk your question: "
    )

    retrieved_text, used_sources = (
        retrieve_chunks(
            question,
            index,
            chunks,
            chunk_sources
        )
    )

    answer = generate_answer(
        question,
        retrieved_text
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


# =========================
# PROGRAM START
# =========================

if __name__ == "__main__":
    main()