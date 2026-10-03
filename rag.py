import os
import re
import io

import PyPDF2
import nltk
import numpy as np
import faiss

from nltk import sent_tokenize
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv


# =========================================================
# NLTK
# =========================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

# Local development: reads sa.env if it exists
load_dotenv(os.path.join(BASE_DIR, "sa.env"))

api_key = os.getenv("GEMINI_API_KEY")

# Streamlit Cloud: fall back to Secrets
if not api_key:
    try:
        import streamlit as st

        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    raise ValueError(
        "Gemini API key not found. Set GEMINI_API_KEY in sa.env (local) "
        "or in Streamlit Secrets (cloud)."
    )


client = genai.Client(api_key=api_key)

# Model name: env var -> Streamlit secret -> default
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

if not GEMINI_MODEL:
    try:
        import streamlit as st

        GEMINI_MODEL = st.secrets.get("GEMINI_MODEL")
    except Exception:
        GEMINI_MODEL = None

if not GEMINI_MODEL:
    GEMINI_MODEL = "gemini-2.5-flash-lite"


# =========================================================
# RAG SETTINGS
# =========================================================

CHUNK_SIZE = 6
CHUNK_OVERLAP = 2

FAISS_TOP_K = 4
KEYWORD_TOP_K = 3
MAX_CONTEXT_CHUNKS = 6

MAX_OUTPUT_TOKENS = 1024


# =========================================================
# EMBEDDING MODEL
# =========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# =========================================================
# GLOBAL DOCUMENT DATA
# =========================================================

# Combined index (rebuilt from `documents` whenever it changes)
index = None
chunks = []
chunk_metadata = []
keyword_index = {}

# Per-file storage: filename -> {"chunks": [...], "embeddings": ndarray}
documents = {}

current_document_name = None


# =========================================================
# BASIC RESPONSES
# =========================================================

BASIC_RESPONSES = {
    "hi": "Hello! Upload a document and ask me a question about it.",
    "hello": "Hello! Upload a document and ask me a question about it.",
    "hey": "Hey! Upload a document and ask me a question about it.",
    "good morning": "Good morning! Upload a document and ask me a question.",
    "good afternoon": "Good afternoon! Upload a document and ask me a question.",
    "good evening": "Good evening! Upload a document and ask me a question.",
    "thanks": "You're welcome!",
    "thank you": "You're welcome!",
    "thanks a lot": "You're welcome!",
    "thank you so much": "You're welcome!",
    "bye": "Goodbye! Have a great day!",
    "goodbye": "Goodbye! Have a great day!",
}


# =========================================================
# STOP WORDS
# =========================================================

STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "of", "for", "in", "to",
    "and", "on", "how", "why", "can", "does", "do", "give", "me",
    "explain", "tell", "please", "about", "from", "with", "this",
    "that", "which", "where", "when", "who", "define", "definition",
    "describe", "meaning", "mean", "means", "whats", "s",
}


# =========================================================
# CLEAN QUERY
# =========================================================

def clean_query(query):

    words = re.findall(r"[a-zA-Z0-9-]+", query.lower())

    keywords = [
        word
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    ]

    if keywords:
        return " ".join(keywords)

    return query.strip()


# =========================================================
# EXTRACT TEXT FROM PDF
# =========================================================

def extract_pdf_text(pdf_bytes):

    pdf_file = io.BytesIO(pdf_bytes)

    reader = PyPDF2.PdfReader(pdf_file)

    text_parts = []

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            page_text = page.extract_text()

            if page_text:
                text_parts.append(page_text + "\n")

        except Exception as e:
            print(f"Error reading page {page_number}:", e)

    text = "\n".join(text_parts)

    # Clean text
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    if not text:
        raise ValueError("No readable text was found in the uploaded PDF.")

    return text


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(text):

    sentences = sent_tokenize(text)

    if not sentences:
        raise ValueError("No sentences were found in the document.")

    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)

    document_chunks = []

    for i in range(0, len(sentences), step):

        chunk = " ".join(sentences[i:i + CHUNK_SIZE])

        if chunk.strip():
            document_chunks.append(chunk.strip())

    if not document_chunks:
        raise ValueError("No chunks were created from the document.")

    return document_chunks


# =========================================================
# BUILD KEYWORD INDEX
# =========================================================

def build_keyword_index(document_chunks):

    new_keyword_index = {}

    for chunk_number, chunk in enumerate(document_chunks):

        words = set(re.findall(r"\b[a-zA-Z0-9-]+\b", chunk.lower()))

        for word in words:

            if word not in STOP_WORDS and len(word) > 1:

                new_keyword_index.setdefault(word, []).append(chunk_number)

    return new_keyword_index


# =========================================================
# REBUILD COMBINED INDEX FROM ALL DOCUMENTS
# =========================================================

def rebuild_index():

    global index
    global chunks
    global chunk_metadata
    global keyword_index
    global current_document_name

    if not documents:
        index = None
        chunks = []
        chunk_metadata = []
        keyword_index = {}
        current_document_name = None
        return

    all_chunks = []
    all_metadata = []
    all_embeddings = []

    for name, doc in documents.items():

        for text in doc["chunks"]:

            all_metadata.append({
                "chunk_id": len(all_chunks),
                "source": name,
            })

            all_chunks.append(text)

        all_embeddings.append(doc["embeddings"])

    embeddings = np.vstack(all_embeddings).astype("float32")

    new_index = faiss.IndexFlatIP(embeddings.shape[1])
    new_index.add(embeddings)

    index = new_index
    chunks = all_chunks
    chunk_metadata = all_metadata
    keyword_index = build_keyword_index(all_chunks)
    current_document_name = ", ".join(documents.keys())


# =========================================================
# PROCESS UPLOADED DOCUMENT (adds to existing documents)
# =========================================================

def process_uploaded_document(pdf_bytes, filename):

    print(f"Processing uploaded document: {filename}")

    text = extract_pdf_text(pdf_bytes)

    print("Extracted characters:", len(text))

    new_chunks = create_chunks(text)

    print("Total chunks:", len(new_chunks))

    print("Creating embeddings...")

    embeddings = embedding_model.encode(
        new_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    ).astype("float32")

    # Same filename = replace the old version
    documents[filename] = {
        "chunks": new_chunks,
        "embeddings": embeddings,
    }

    rebuild_index()

    print("Document processing completed.")

    return {
        "filename": filename,
        "chunks": len(new_chunks),
        "characters": len(text),
    }


# =========================================================
# REMOVE / RESET DOCUMENTS
# =========================================================

def remove_document(filename):

    documents.pop(filename, None)
    rebuild_index()


def reset_documents():

    documents.clear()
    rebuild_index()


# =========================================================
# DOCUMENT STATUS
# =========================================================

def document_loaded():

    return index is not None and len(chunks) > 0


# =========================================================
# CURRENT DOCUMENT(S)
# =========================================================

def get_document_name():

    return current_document_name


# =========================================================
# KEYWORD SEARCH
# =========================================================

def keyword_search(search_query):

    keywords = search_query.split()

    keyword_scores = {}

    # Word matching
    for keyword in keywords:

        for chunk_number in keyword_index.get(keyword, []):

            keyword_scores[chunk_number] = (
                keyword_scores.get(chunk_number, 0) + 1
            )

    # Exact phrase bonus
    if len(keywords) > 1:

        for chunk_number, chunk in enumerate(chunks):

            if search_query in chunk.lower():

                keyword_scores[chunk_number] = (
                    keyword_scores.get(chunk_number, 0) + 3
                )

    return sorted(
        keyword_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(user_query):

    if not document_loaded():
        return []

    search_query = clean_query(user_query)

    top_k = min(FAISS_TOP_K, len(chunks))

    # Vector search
    query_embeddings = embedding_model.encode(
        [search_query, user_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    scores, indices = index.search(query_embeddings, top_k)

    vector_results = []

    for rank in range(top_k):

        for row in range(indices.shape[0]):

            idx = int(indices[row][rank])

            if 0 <= idx < len(chunks) and idx not in vector_results:
                vector_results.append(idx)

    # Keyword search
    keyword_results = keyword_search(search_query)

    # Combine: keyword results first, then vector results
    final_indices = []

    for chunk_number, _ in keyword_results[:KEYWORD_TOP_K]:

        if chunk_number not in final_indices:
            final_indices.append(chunk_number)

    for chunk_number in vector_results:

        if chunk_number not in final_indices:
            final_indices.append(chunk_number)

    final_indices = final_indices[:MAX_CONTEXT_CHUNKS]

    retrieved_chunks = []

    for idx in final_indices:

        retrieved_chunks.append({
            "chunk_id": idx,
            "text": chunks[idx],
            "metadata": chunk_metadata[idx],
        })

    return retrieved_chunks


# =========================================================
# ASK QUESTION
# =========================================================

def ask_question(user_query):

    user_query = (user_query or "").strip()

    # Empty query
    if not user_query:
        return "Please enter a question."

    # Check document
    if not document_loaded():
        return "Please upload a PDF document before asking a question."

    # Basic responses
    basic_query = re.sub(r"\s+", " ", user_query.lower()).strip()
    basic_query = basic_query.rstrip("!.?")

    if basic_query in BASIC_RESPONSES:
        return BASIC_RESPONSES[basic_query]

    # Retrieve
    retrieved_chunks = retrieve_chunks(user_query)

    if not retrieved_chunks:
        return "No data available in the uploaded document."

    # Build context
    context_parts = []

    for item in retrieved_chunks:

        source = item["metadata"]["source"]
        chunk_id = item["chunk_id"]
        text = item["text"]

        context_parts.append(
            f"""
Source: {source}
Chunk: {chunk_id}

{text}
"""
        )

    context = "\n\n".join(context_parts)

    # Prompt
    prompt = f"""

You are a document question-answering assistant.

The user uploaded one or more documents and asked a question.

Answer ONLY using information contained in the
provided document context.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not assume information that is not present.
4. If the answer is present in the context, provide a
   clear and complete answer.
5. If the question asks for an explanation, explain it
   using only the document context.
6. If the question asks for multiple items, use bullet
   points.
7. Keep the answer concise but complete.
8. Do not mention information from documents that were
   not provided.
9. If the context does not contain enough information
   to answer the question, reply exactly:

No data available in the uploaded document.

Document Context:

{context}

User Question:

{user_query}

Answer:
"""

    # Gemini
    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                candidate_count=1,
            ),
        )

        if response.text and response.text.strip():
            return response.text.strip()

        try:
            print(
                "Empty Gemini response:",
                response.candidates[0].finish_reason,
            )
        except Exception:
            print("Empty Gemini response.")

        return "The model returned an empty response. Please try again."

    except Exception as e:

        print("Gemini error:", e)

        return "Unable to generate an answer right now."