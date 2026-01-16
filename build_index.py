"""
Build FAISS index from all Cameroon law PDFs
Run this once to create the index files
"""

import os
import sys

# Add parent directory to path to import RAG module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RAG import build_faiss_from_folder

if __name__ == "__main__":
    # Paths
    folder = os.path.join(os.getcwd())
    index_path = os.path.join(os.getcwd(), "index_file.index")
    metadata_path = os.path.join(os.getcwd(), "index_file.meta.json")
    
    print("="*60)
    print("Building FAISS Index for Cameroon Laws")
    print("="*60)
    print(f"Source folder: {folder}")
    print(f"Output index: {index_path}")
    print(f"Output metadata: {metadata_path}")
    print("="*60 + "\n")
    
    # Build index
    build_faiss_from_folder(
        folder_path=folder,
        index_output_path=index_path,
        metadata_output_path=metadata_path,
        embedding_model='all-MiniLM-L6-v2'
    )
    
    print("\n" + "="*60)
    print("✓ Index building complete!")
    print("="*60)
    print(f"Files created:")
    print(f"  - {index_path}")
    print(f"  - {metadata_path}")
    print(f"  - {metadata_path.replace('.meta.json', '.chunks.json')}")
    print("="*60)
