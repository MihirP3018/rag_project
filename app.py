import streamlit as st
from main import process_pdf, ask_question

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title = "Book RAG Assistant",
    page_icon = "📚",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------- Custom CSS ----------------
st.markdown(
    """
    <style>
        #MainMenu, footer {visibility: hidden;}
        .block-container {padding-top: 2rem; max-width: 1000px;}

        .hero {
            padding: 1.6rem 2rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
            color: white;
            margin-bottom: 1.2rem;
        }
        .hero h1 {margin: 0; font-size: 2rem; color: white;}
        .hero p {margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1rem;}

        .book-badge {
            display: inline-block;
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            background: rgba(124, 58, 237, 0.12);
            border: 1px solid rgba(124, 58, 237, 0.35);
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }

        [data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
        }

        [data-testid="stMetric"] {
            background: rgba(124, 58, 237, 0.08);
            border-radius: 12px;
            padding: 0.6rem 0.9rem;
        }

        .stButton > button {border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html = True,
)

# ---------------- Session State ----------------
defaults = {
    "vectorstore": None,
    "messages": [],
    "book_name": None,
    "book_pages": 0,
    "book_chunks": 0,
    "pending_query": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 📖 Your Book")
    uploaded_file = st.file_uploader("Choose a PDF book", type = ["pdf"])
    process_button = st.button(
        "🔄 Process Book", use_container_width = True, type = "primary"
    )

    if st.session_state.book_name:
        st.divider()
        st.markdown("### 📌 Current Book")
        st.caption(st.session_state.book_name)
        col1, col2 = st.columns(2)
        col1.metric("Pages", st.session_state.book_pages)
        col2.metric("Chunks", st.session_state.book_chunks)

        if st.button("🗑️ Clear Chat", use_container_width = True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown("### ℹ️ How it works")
    st.caption(
        "1. Upload a PDF\n\n"
        "2. Click **Process Book**\n\n"
        "3. Ask questions in the chat"
    )

# ---------------- Header ----------------
st.markdown(
    """
    <div class="hero">
        <h1>📚 Book RAG Assistant</h1>
        <p>Upload a PDF book and ask questions. Answers come only from the book, with page sources.</p>
    </div>
    """,
    unsafe_allow_html = True,
)

# ---------------- Process Uploaded PDF ----------------
if process_button:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a PDF book first.")
    else:
        with st.spinner("📚 Processing your book... this can take a minute for large books."):
            pdf_bytes = uploaded_file.getbuffer()
            vectorstore, pages, chunks = process_pdf(pdf_bytes)

            st.session_state.vectorstore = vectorstore
            st.session_state.book_name = uploaded_file.name
            st.session_state.book_pages = pages
            st.session_state.book_chunks = chunks
            st.session_state.messages = []

        st.toast("Book processed successfully!", icon = "✅")
        st.rerun()

# ---------------- Main Area ----------------
if st.session_state.vectorstore is None:
    with st.container(border = True):
        st.markdown("### 👋 Welcome")
        st.write(
            "Use the sidebar to upload a PDF book and click **Process Book**. "
            "Once it is ready, you can start asking questions here."
        )
else:
    st.markdown(
        f'<div class="book-badge">📖 {st.session_state.book_name}</div>',
        unsafe_allow_html = True,
    )

    # Suggested questions (shown only before the first message)
    if not st.session_state.messages:
        st.markdown("##### 💡 Try asking")
        suggestions = [
            "Give me a summary of this book",
            "What are the main topics covered?",
            "Explain the key concepts in simple words",
        ]
        cols = st.columns(len(suggestions))
        for col, text in zip(cols, suggestions):
            if col.button(text, use_container_width = True):
                st.session_state.pending_query = text
                st.rerun()

# ---------------- Display Previous Chat Messages ----------------
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar = avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("📄 Sources: pages " + ", ".join(map(str, message["sources"])))

# ---------------- Chat Input ----------------
query = st.chat_input("Ask a question about the book...")

# A clicked suggestion acts like a typed question
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

if query:
    if st.session_state.vectorstore is None:
        st.warning("⚠️ Please upload and process a PDF before asking questions.")
    else:
        history = list(st.session_state.messages)  # capture BEFORE adding the new question

        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar = "🧑"):
            st.markdown(query)

        with st.chat_message("assistant", avatar = "🤖"):
            with st.spinner("🔍 Searching the book..."):
                try:
                    answer, pages = ask_question(
                        st.session_state.vectorstore, query, history
                    )
                except Exception as e:
                    answer, pages = f"⚠️ Something went wrong: {e}", []

            st.markdown(answer)
            if pages:
                st.caption("📄 Sources: pages " + ", ".join(map(str, pages)))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": pages}
        )