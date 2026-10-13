"""
Optional command-line PDF indexer.

Usage:
    python create_database.py "path/to/your.pdf"

The Streamlit app automatically processes uploaded PDFs through main.py.
This script uses the same process_pdf() function.
"""

from pydoc import doc
import sys
from pathlib import Path
from main import process_pdf

def main():
    if len(sys.argv) != 2:
        print('Usafe: Python create_database.py "path/to/your.pdf"')
        raise SystemExit(1)

    pdf_path = Path(1)

    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}")
        raise SystemExit(1)

    try:
        (
            vectorstore,
            pages,
            chunks,
            document_id,
        ) = process_pdf(pdf_path.read_bytes())

    except Exception as exc:
        print(f"Could not index PDF: {exc}")
        raise SystemExit(1) from exc

    print("PDF indexed successfully.")
    print(f"Pages: {pages}")
    print(f"Chunks: {chunks}")
    print(f"Documnet ID: {document_id}")

if __name__ == "__main__":
    main()