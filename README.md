# Government Scheme Assistant (RAG Chatbot)

An AI-powered Government Scheme Assistant built using Python, FAISS, Sentence Transformers, and Google Gemini API.

## Features

* Multi-PDF document search
* Retrieval-Augmented Generation (RAG)
* Semantic Search using Sentence Transformers
* FAISS Vector Database
* Source Tracking
* Hybrid Search
* Gemini 2.5 Flash Integration

## Tech Stack

* Python
* FAISS
* Sentence Transformers
* Google Gemini API
* LangChain Text Splitter
* NumPy
* PyPDF

## Project Workflow

1. Read Government Scheme PDFs
2. Split documents into chunks
3. Convert chunks into embeddings
4. Store embeddings in FAISS
5. Retrieve relevant chunks
6. Generate answers using Gemini

## Example Query

Question:
What is PMMY?

Answer:
Pradhan Mantri MUDRA Yojana (PMMY) was launched on 8 April 2015 to provide loans up to ₹20 lakh to micro enterprises engaged in manufacturing, trading and services sectors.

Source:
pmmy.pdf

## Future Improvements

* Streamlit UI
* Chat History
* PDF Upload Support
* Deployment on Cloud
