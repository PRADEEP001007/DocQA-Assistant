import html

import streamlit as st

import rag as rag_module
from rag import (
    process_uploaded_document,
    ask_question,
    document_loaded,
    get_document_name,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="DocQA Assistant",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================================================
# SESSION STATE
# =========================================================
st.session_state.setdefault("messages", [])
st.session_state.setdefault("processed_ids", [])   # (name, size) of indexed files
st.session_state.setdefault("doc_names", [])        # names shown in the UI
st.session_state.setdefault("doc_store", {})        # name -> pdf bytes (for re-indexing)
st.session_state.setdefault("changing_doc", False)

# =========================================================
# CSS
# =========================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp { font-family: 'Inter', sans-serif; }

/* One background colour everywhere */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background:
        radial-gradient(900px 400px at 50% -10%, rgba(99,102,241,0.20), transparent 70%),
        #0a0a0f !important;
    background-attachment: fixed !important;
}

/* Hide default Streamlit chrome */
header[data-testid="stHeader"], footer, #MainMenu,
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

/* Container */
.block-container { max-width: 720px; padding-top: 2rem; padding-bottom: 7rem; }

/* Header */
.custom-header { text-align: center; padding: 20px 0 30px 0; }
.custom-header h1 {
    margin: 0; font-size: 42px; font-weight: 700; letter-spacing: -1.5px;
    background: linear-gradient(180deg, #ffffff, #a5a8c8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.custom-header p { margin: 8px 0 0 0; color: #8a8da8; font-size: 15px; }

/* Uploader */
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.025);
    border: 1px dashed rgba(255,255,255,0.15);
    border-radius: 14px;
}
[data-testid="stFileUploader"] section:hover { border-color: #818cf8; }

/* Document status card */
.document-status {
    background: rgba(129,140,248,0.10);
    border: 1px solid rgba(129,140,248,0.22);
    color: #c7c9ff; padding: 13px 16px; border-radius: 12px;
    font-size: 14px; }
.document-status .dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: #3dd6c6; margin-right: 10px;
}

.doc-title { font-weight: 600; margin-bottom: 8px; }
.doc-chip {
    display: inline-block; margin: 3px 6px 0 0; padding: 4px 10px;
    border-radius: 999px; font-size: 12px; color: #c7c9ff;
    background: rgba(129,140,248,0.12); border: 1px solid rgba(129,140,248,0.25);
}

.file-name {
    padding: 10px 14px; border-radius: 10px; font-size: 13px; color: #d7d9f5;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* Change-document button */
div[data-testid="stButton"] > button {
    background: transparent; color: #a5a8c8;
    border: 1px solid rgba(255,255,255,0.12); border-radius: 12px;
    padding: 10px 14px; font-size: 13px; width: 100%;
}
div[data-testid="stButton"] > button:hover {
    border-color: #818cf8; color: #ffffff; background: rgba(129,140,248,0.08);
}

/* Chat */
[data-testid="stChatMessage"] { background: transparent !important; padding: 8px 0; }
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none !important; }
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li { line-height: 1.7; font-size: 15px; color: #e5e7eb; }

.user-msg {
    background: #1b1b2b; border: 1px solid rgba(255,255,255,0.07);
    color: #f5f5f7; padding: 12px 16px; border-radius: 14px;
    display: inline-block; max-width: 90%; line-height: 1.5;
}

/* Bottom bar + chat input */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: transparent !important; }

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea { background: #14141f !important; }
[data-testid="stChatInput"] {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}
[data-testid="stChatInput"] > div { border: none !important; box-shadow: none !important; }
[data-testid="stChatInput"]:focus-within { border-color: #818cf8 !important; }
[data-testid="stChatInput"] textarea { color: #ffffff !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #8a8da8 !important; }

[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stSpinner"] { color: #a5a8c8; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def user_bubble(text: str) -> None:
    st.markdown(
        f'<div class="user-msg">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def remove_file(name: str):
    """Remove one document. Returns an error string, or None on success.

    Needs ONE of these in rag.py:
      - remove_document(name)  -> delete that file's chunks from the index
      - reset_documents()      -> clear the whole index (files are re-indexed here)
    """
    remaining = {n: b for n, b in st.session_state.doc_store.items() if n != name}
    remove_fn = getattr(rag_module, "remove_document", None)
    reset_fn = getattr(rag_module, "reset_documents", None)

    try:
        if remove_fn:
            remove_fn(name)
        elif reset_fn:
            reset_fn()
            for n, b in remaining.items():
                process_uploaded_document(b, n)
        else:
            return "rag.py needs a remove_document(name) or reset_documents() function."
    except Exception as e:
        return str(e)

    st.session_state.doc_store = remaining
    st.session_state.doc_names = [n for n in st.session_state.doc_names if n != name]
    st.session_state.processed_ids = [
        i for i in st.session_state.processed_ids if i[0] != name
    ]
    st.session_state.messages = []
    return None


has_docs = document_loaded() and bool(st.session_state.doc_names)

# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="custom-header">'
    "<h1>DocQA Assistant</h1>"
    "<p>Ask questions grounded strictly in your PDF documents.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# =========================================================
# UPLOAD (multiple PDFs; hidden once documents are loaded)
# =========================================================
if not has_docs or st.session_state.changing_doc:
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if (f.name, f.size) not in st.session_state.processed_ids
        ]
        failed = False

        if new_files:
            with st.spinner(f"Processing {len(new_files)} document(s)..."):
                for f in new_files:
                    try:
                        data = f.getvalue()
                        process_uploaded_document(data, f.name)
                        st.session_state.processed_ids.append((f.name, f.size))
                        st.session_state.doc_store[f.name] = data
                        if f.name not in st.session_state.doc_names:
                            st.session_state.doc_names.append(f.name)
                    except Exception as e:
                        failed = True
                        st.error(f"Error processing {f.name}: {e}")

            if not failed:
                st.session_state.messages = []
                st.session_state.changing_doc = False
                st.rerun()  # re-run so the uploader disappears

    if has_docs and st.session_state.changing_doc:
        if st.button("Cancel", key="cancel_add"):
            st.session_state.changing_doc = False
            st.rerun()

# =========================================================
# DOCUMENT STATUS + FILE LIST (replaces the uploader)
# =========================================================
if has_docs and not st.session_state.changing_doc:
    names = st.session_state.doc_names
    count = len(names)
    label = "1 document" if count == 1 else f"{count} documents"

    col_info, col_btn = st.columns([4, 1.3], vertical_alignment="center")

    with col_info:
        st.markdown(
            f'<div class="document-status"><span class="dot"></span>'
            f"<b>{label}</b>&nbsp;&nbsp;&bull;&nbsp;&nbsp;Ready</div>",
            unsafe_allow_html=True,
        )

    with col_btn:
        if st.button("Add files", key="add_files"):
            st.session_state.changing_doc = True
            st.rerun()

    remove_target = None
    for i, n in enumerate(names):
        c1, c2 = st.columns([4, 1.3], vertical_alignment="center")
        c1.markdown(
            f'<div class="file-name">{html.escape(n)}</div>',
            unsafe_allow_html=True,
        )
        if c2.button("Remove", key=f"rm_{i}"):
            remove_target = n

    if remove_target:
        err = remove_file(remove_target)
        if err:
            st.error(f"Could not remove {remove_target}: {err}")
        else:
            st.rerun()

    st.write("")

# =========================================================
# CHAT HISTORY
# =========================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            user_bubble(message["content"])
        else:
            st.markdown(message["content"])

# =========================================================
# CHAT INPUT
# =========================================================
prompt = st.chat_input("Ask a question about your document...")

if prompt:
    if not has_docs:
        st.warning("Please upload a PDF document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            user_bubble(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_question(prompt)
                except Exception as e:
                    answer = f"Sorry, an error occurred: {e}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})