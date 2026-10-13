
# PDF-Based RAG Chatbot

A Streamlit application that lets users upload a PDF, ask questions about its contents, and see the source pages used for answers.

## Features

- Upload and process PDF books.
- Extract PDF text and split it into overlapping chunks.
- Create embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
- Store vectors persistently in ChromaDB.
- Maintain a separate index for each PDF's content.
- Retrieve relevant chunks using Maximal Marginal Relevance (MMR).
- Generate context-grounded answers using a Groq-hosted language model.
- Support follow-up questions.
- Display source page numbers.
- Reuse the index when the same PDF is processed again.

## Technologies

- Python
- Streamlit
- LangChain
- PyMuPDF
- Hugging Face Sentence Transformers
- ChromaDB
- Groq API

## Setup on Windows

### 1. Open the project directory

Open this folder in VS Code and start a terminal in the project root.

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Never commit `.env` or share your API key.

### 5. Run the application

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal.

### 6. Use the chatbot

1. Upload a PDF.
2. Click **Process Book**.
3. Wait for processing to finish.
4. Ask questions about the document.
5. Review the displayed source pages.

## Optional: Index a PDF from the terminal

```powershell
python create_database.py "path/to/your.pdf"
```

The command-line script uses the same processing function as the Streamlit application.

## RAG pipeline

PDF → PyMuPDF text extraction → text chunks → Hugging Face embeddings → ChromaDB → MMR retrieval → Groq LLM → answer with source pages.

## How it works

1. The PDF loader extracts text and page metadata.
2. The text splitter divides the content into overlapping chunks.
3. The embedding model converts each chunk into a vector.
4. ChromaDB stores the vectors and associated text.
5. MMR retrieves relevant chunks for the user's question.
6. The Groq-hosted language model generates an answer from the retrieved context.
7. The application displays the answer and source page numbers.

## Limitations

- OCR is not included for image-only scanned PDFs.
- Prompt instructions cannot guarantee that every answer is correct.
- Vector indexes consume local disk space and can be regenerated.
- Dependency versions should be tested and locked for reproducible deployments.