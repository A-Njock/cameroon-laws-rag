"""
Simple script to build FAISS index locally
Run this in your LOIS CAMEROUN folder
"""
import os
import sys

# Add current directory to path
sys.path.insert(0, os.getcwd())

from RAG import build_faiss_from_folder

# Build index from all PDFs in current directory
folder_path = "."
index_path = "index_file.index"
metadata_path = "index_file.meta.json"

print("="*60)
print("Building FAISS Index from PDFs")
print("="*60)
print(f"Folder: {folder_path}")
print(f"This will take ~10-15 minutes...")
print("="*60)

build_faiss_from_folder(
    folder_path=folder_path,
    index_output_path=index_path,
    metadata_output_path=metadata_path,
    embedding_model='all-MiniLM-L6-v2'
)

print("\n" + "="*60)
print("✓ Index building complete!")
print("="*60)
print(f"\nGenerated files:")
print(f"  - {index_path}")
print(f"  - {metadata_path}")
print(f"  - index_file.chunks.json")
print("\nNext: Upload these 3 files to Cloudflare R2")
