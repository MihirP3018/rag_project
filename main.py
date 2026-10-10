import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Load Embedding Model (cached so it loads only once)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

# Load Groq Model (cached)
@st.cache_resource
def load_model():
    return ChatGroq(model = "openai/gpt-oss-120b")

# Process Uploaded PDF
def process_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as temp_file:
        temp_file.write(pdf_bytes)
        pdf_path = temp_file.name

    try:
        docs = PyMuPDFLoader(pdf_path).load()
    finally:
        os.remove(pdf_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = load_embeddings()
    )

    return vectorstore, len(docs), len(chunks)

# Answer User Question
def ask_question(vectorstore, query, history = None):
    model = load_model()

    # Turn follow-ups like "explain that simply" into a standalone question
    search_query = query
    if history:
        recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        rewrite = model.invoke(
            "Rewrite the last question as a standalone question using the chat "
            "history. Return only the question.\n\n"
            f"Chat history:\n{recent}\n\nLast question: {query}"
        )
        search_query = rewrite.content.strip()

    retriever = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(search_query)

    context = "\n\n".join(
        f"[Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful AI assistant. Use ONLY the provided context to answer. "
         "If the answer is not in the context, say: "
         "\"I could not find the answer in the document.\" "
         "Do not use outside knowledge.\n\nContext:\n{context}"),
        ("human", "Question:\n{question}"),
    ])

    response = model.invoke(prompt.invoke({"context": context, "question": query}))
    pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
    return response.content, pages