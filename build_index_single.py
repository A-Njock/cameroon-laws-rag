"""
Build FAISS index from a SINGLE PDF — for testing only.
Usage:
    python build_index_single.py                   # uses default Labour Code
    python build_index_single.py "path/to/law.pdf" # use a specific file
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RAG import (
    RobustRAGSystem,
    extract_text_from_pdf,
    infer_law_reference,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DEFAULT_PDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "LOIS CAMEROUNAISES PDF",
    "LOI N° 092-007 DU 14 A0UT 1992 portant code du travail.pdf",
)


def build_single(pdf_path: str) -> None:
    index_path    = os.path.join(os.path.dirname(pdf_path) if False else os.getcwd(), "index_file.index")
    metadata_path = os.path.join(os.getcwd(), "index_file.meta.json")
    chunks_path   = os.path.join(os.getcwd(), "index_file.meta.chunks.json")

    print("=" * 60)
    print("Single-document index build (TEST MODE)")
    print("=" * 60)
    print(f"PDF:   {pdf_path}")
    print(f"Model: {EMBEDDING_MODEL}  (dim={EMBEDDING_DIM})")
    print("=" * 60 + "\n")

    # Extract text
    print("Extracting text from PDF...")
    doc_text = extract_text_from_pdf(pdf_path)
    print(f"  -> {len(doc_text):,} characters extracted\n")

    # Infer law reference
    law_ref = infer_law_reference(doc_text, pdf_path)
    print(f"Law reference: {law_ref}\n")

    # Chunk using RAG system (LLM-assisted with fallback)
    print("Chunking document...")
    rag = RobustRAGSystem(
        api_key="DEEPSEEK_ONLY",
        document_text=doc_text,
        embedding_model=EMBEDDING_MODEL,
        default_law=law_ref,
    )
    print(f"  -> {len(rag.chunks)} chunks extracted\n")

    # Encode with passage: prefix
    print("Encoding chunks...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    prefixed = ["passage: " + c for c in rag.chunks]
    embeddings = model.encode(prefixed, show_progress_bar=True,
                               batch_size=32, normalize_embeddings=True)

    # Build FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(np.array(embeddings).astype("float32"))

    # Save
    print(f"\nSaving index → {index_path}")
    faiss.write_index(index, index_path)

    print(f"Saving metadata → {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(rag.metadata, f, ensure_ascii=False, indent=2)

    print(f"Saving chunks → {chunks_path}")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(rag.chunks, f, ensure_ascii=False)

    print(f"\n✓ Done. {len(rag.chunks)} chunks indexed from 1 document.")
    print("  Deploy to Railway and test. Replace with full corpus later.")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    if not os.path.exists(pdf):
        print(f"ERROR: File not found: {pdf}")
        sys.exit(1)
    build_single(pdf)
