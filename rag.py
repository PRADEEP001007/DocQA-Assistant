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

load_dotenv(os.path.join(BASE_DIR, "sa.env"))

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in sa.env")


client = genai.Client(api_key=api_key)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)


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

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# =========================================================
# GLOBAL DOCUMENT DATA
# =========================================================

index = None
chunks = []
chunk_metadata = []
keyword_index = {}

current_document_name = None


# =========================================================
# BASIC RESPONSES
# =========================================================

BASIC_RESPONSES = {

    "hi":
        "Hello! Upload a document and ask me a question about it.",

    "hello":
        "Hello! Upload a document and ask me a question about it.",

    "hey":
        "Hey! Upload a document and ask me a question about it.",

    "good morning":
        "Good morning! Upload a document and ask me a question.",

    "good afternoon":
        "Good afternoon! Upload a document and ask me a question.",

    "good evening":
        "Good evening! Upload a document and ask me a question.",

    "thanks":
        "You're welcome!",

    "thank you":
        "You're welcome!",

    "thanks a lot":
        "You're welcome!",

    "thank you so much":
        "You're welcome!",

    "bye":
        "Goodbye! Have a great day!",

    "goodbye":
        "Goodbye! Have a great day!"
}


# =========================================================
# STOP WORDS
# =========================================================

STOP_WORDS = {

    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "to",
    "and",
    "on",
    "how",
    "why",
    "can",
    "does",
    "do",
    "give",
    "me",
    "explain",
    "tell",
    "please",
    "about",
    "from",
    "with",
    "this",
    "that",
    "which",
    "where",
    "when",
    "who",
    "define",
    "definition",
    "describe",
    "meaning",
    "mean",
    "means",
    "whats",
    "s"
}


# =========================================================
# CLEAN QUERY
# =========================================================

def clean_query(query):

    words = re.findall(
        r"[a-zA-Z0-9-]+",
        query.lower()
    )

    keywords = [
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 1
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

                text_parts.append(
                    page_text + "\n"
                )

        except Exception as e:

            print(
                f"Error reading page {page_number}:",
                e
            )

    text = "\n".join(text_parts)

    # Clean text

    text = re.sub(
        r"\n+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.strip()

    if not text:

        raise ValueError(
            "No readable text was found in the uploaded PDF."
        )

    return text


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(text):

    sentences = sent_tokenize(text)

    if not sentences:

        raise ValueError(
            "No sentences were found in the document."
        )

    step = max(
        1,
        CHUNK_SIZE - CHUNK_OVERLAP
    )

    document_chunks = []

    for i in range(
        0,
        len(sentences),
        step
    ):

        chunk = " ".join(
            sentences[
                i:i + CHUNK_SIZE
            ]
        )

        if chunk.strip():

            document_chunks.append(
                chunk.strip()
            )

    if not document_chunks:

        raise ValueError(
            "No chunks were created from the document."
        )

    return document_chunks


# =========================================================
# BUILD KEYWORD INDEX
# =========================================================

def build_keyword_index(document_chunks):

    new_keyword_index = {}

    chunks_lower = [
        chunk.lower()
        for chunk in document_chunks
    ]

    for chunk_number, chunk in enumerate(
        chunks_lower
    ):

        words = set(
            re.findall(
                r"\b[a-zA-Z0-9-]+\b",
                chunk
            )
        )

        for word in words:

            if (
                word not in STOP_WORDS
                and len(word) > 1
            ):

                new_keyword_index.setdefault(
                    word,
                    []
                ).append(chunk_number)

    return new_keyword_index


# =========================================================
# PROCESS UPLOADED DOCUMENT
# =========================================================

def process_uploaded_document(
    pdf_bytes,
    filename
):

    global index
    global chunks
    global chunk_metadata
    global keyword_index
    global current_document_name

    print(
        f"Processing uploaded document: {filename}"
    )

    # -----------------------------------------------------
    # EXTRACT TEXT
    # -----------------------------------------------------

    text = extract_pdf_text(
        pdf_bytes
    )

    print(
        "Extracted characters:",
        len(text)
    )

    # -----------------------------------------------------
    # CREATE CHUNKS
    # -----------------------------------------------------

    new_chunks = create_chunks(
        text
    )

    print(
        "Total chunks:",
        len(new_chunks)
    )

    # -----------------------------------------------------
    # CREATE EMBEDDINGS
    # -----------------------------------------------------

    print("Creating embeddings...")

    embeddings = embedding_model.encode(
        new_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32
    )

    embeddings = embeddings.astype(
        "float32"
    )

    # -----------------------------------------------------
    # CREATE FAISS INDEX
    # -----------------------------------------------------

    dimension = embeddings.shape[1]

    new_index = faiss.IndexFlatIP(
        dimension
    )

    new_index.add(
        embeddings
    )

    # -----------------------------------------------------
    # METADATA
    # -----------------------------------------------------

    new_metadata = []

    for i in range(
        len(new_chunks)
    ):

        new_metadata.append({

            "chunk_id": i,

            "source": filename

        })

    # -----------------------------------------------------
    # KEYWORD INDEX
    # -----------------------------------------------------

    new_keyword_index = build_keyword_index(
        new_chunks
    )

    # -----------------------------------------------------
    # UPDATE GLOBAL DATA
    # -----------------------------------------------------

    index = new_index

    chunks = new_chunks

    chunk_metadata = new_metadata

    keyword_index = new_keyword_index

    current_document_name = filename

    print(
        "Document processing completed."
    )

    return {

        "filename": filename,

        "chunks": len(chunks),

        "characters": len(text)

    }


# =========================================================
# DOCUMENT STATUS
# =========================================================

def document_loaded():

    return (
        index is not None
        and len(chunks) > 0
    )


# =========================================================
# CURRENT DOCUMENT
# =========================================================

def get_document_name():

    return current_document_name


# =========================================================
# KEYWORD SEARCH
# =========================================================

def keyword_search(
    search_query
):

    keywords = search_query.split()

    keyword_scores = {}

    chunks_lower = [
        chunk.lower()
        for chunk in chunks
    ]

    # -----------------------------------------------------
    # WORD MATCHING
    # -----------------------------------------------------

    for keyword in keywords:

        for chunk_number in keyword_index.get(
            keyword,
            []
        ):

            keyword_scores[
                chunk_number
            ] = keyword_scores.get(
                chunk_number,
                0
            ) + 1

    # -----------------------------------------------------
    # EXACT PHRASE BONUS
    # -----------------------------------------------------

    if len(keywords) > 1:

        for chunk_number, chunk in enumerate(
            chunks_lower
        ):

            if search_query in chunk:

                keyword_scores[
                    chunk_number
                ] = keyword_scores.get(
                    chunk_number,
                    0
                ) + 3

    results = sorted(
        keyword_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return results


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(
    user_query
):

    if not document_loaded():

        return []

    search_query = clean_query(
        user_query
    )

    top_k = min(
        FAISS_TOP_K,
        len(chunks)
    )

    # -----------------------------------------------------
    # VECTOR SEARCH
    # -----------------------------------------------------

    query_embeddings = embedding_model.encode(

        [
            search_query,
            user_query
        ],

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False

    ).astype("float32")


    scores, indices = index.search(
        query_embeddings,
        top_k
    )


    vector_results = []

    for rank in range(top_k):

        for row in range(
            indices.shape[0]
        ):

            idx = int(
                indices[row][rank]
            )

            if (
                0 <= idx < len(chunks)
                and idx not in vector_results
            ):

                vector_results.append(
                    idx
                )


    # -----------------------------------------------------
    # KEYWORD SEARCH
    # -----------------------------------------------------

    keyword_results = keyword_search(
        search_query
    )


    # -----------------------------------------------------
    # COMBINE RESULTS
    # -----------------------------------------------------

    final_indices = []

    # Keyword results first

    for chunk_number, _ in keyword_results[
        :KEYWORD_TOP_K
    ]:

        if chunk_number not in final_indices:

            final_indices.append(
                chunk_number
            )


    # Vector results

    for chunk_number in vector_results:

        if chunk_number not in final_indices:

            final_indices.append(
                chunk_number
            )


    # Limit context

    final_indices = final_indices[
        :MAX_CONTEXT_CHUNKS
    ]


    retrieved_chunks = []

    for idx in final_indices:

        retrieved_chunks.append({

            "chunk_id": idx,

            "text": chunks[idx],

            "metadata": chunk_metadata[idx]

        })


    return retrieved_chunks


# =========================================================
# ASK QUESTION
# =========================================================

def ask_question(
    user_query
):

    user_query = (
        user_query or ""
    ).strip()


    # -----------------------------------------------------
    # EMPTY QUERY
    # -----------------------------------------------------

    if not user_query:

        return "Please enter a question."


    # -----------------------------------------------------
    # CHECK DOCUMENT
    # -----------------------------------------------------

    if not document_loaded():

        return (
            "Please upload a PDF document "
            "before asking a question."
        )


    # -----------------------------------------------------
    # BASIC RESPONSES
    # -----------------------------------------------------

    basic_query = re.sub(
        r"\s+",
        " ",
        user_query.lower()
    ).strip()

    basic_query = basic_query.rstrip(
        "!.?"
    )


    if basic_query in BASIC_RESPONSES:

        return BASIC_RESPONSES[
            basic_query
        ]


    # -----------------------------------------------------
    # RETRIEVE
    # -----------------------------------------------------

    retrieved_chunks = retrieve_chunks(
        user_query
    )


    if not retrieved_chunks:

        return (
            "No data available in the uploaded document."
        )


    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    context_parts = []

    for item in retrieved_chunks:

        source = item[
            "metadata"
        ][
            "source"
        ]

        chunk_id = item[
            "chunk_id"
        ]

        text = item[
            "text"
        ]

        context_parts.append(

            f"""
Source: {source}
Chunk: {chunk_id}

{text}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------

    prompt = f"""

You are a document question-answering assistant.

The user uploaded a document and asked a question.

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


    # -----------------------------------------------------
    # GEMINI
    # -----------------------------------------------------

    try:

        response = client.models.generate_content(

            model=GEMINI_MODEL,

            contents=prompt,

            config=types.GenerateContentConfig(

                temperature=0,

                max_output_tokens=MAX_OUTPUT_TOKENS,

                candidate_count=1

            )

        )


        if (
            response.text
            and response.text.strip()
        ):

            return response.text.strip()


        try:

            print(
                "Empty Gemini response:",
                response.candidates[
                    0
                ].finish_reason
            )

        except Exception:

            print(
                "Empty Gemini response."
            )


        return (
            "The model returned an empty response. "
            "Please try again."
        )


    except Exception as e:

        print(
            "Gemini error:",
            e
        )

        return (
            "Unable to generate an answer right now."
        )