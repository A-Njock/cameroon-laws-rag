# SIMPLE COLAB INSTRUCTIONS
# Copy this entire file and paste it into Google Colab cells

# ============================================
# CELL 1: Install Dependencies
# ============================================
!pip install -q faiss-cpu sentence-transformers PyPDF2 openai

# ============================================
# CELL 2: Download Repository
# ============================================
!rm -rf cameroon-laws-rag
!git clone https://github.com/A-Njock/cameroon-laws-rag.git
!mv cameroon-laws-rag/*.pdf .
!cp cameroon-laws-rag/RAG.py .

import os
pdf_count = len([f for f in os.listdir('.') if f.endswith('.pdf')])
print(f"✓ Downloaded {pdf_count} PDF files")

# ============================================
# CELL 3: Build Index
# ============================================
from RAG import build_faiss_from_folder

print("="*60)
print("Building FAISS Index")
print("="*60)
print("This will take ~10-15 minutes...")
print("="*60)

build_faiss_from_folder(
    folder_path=".",
    index_output_path="index_file.index",
    metadata_output_path="index_file.meta.json",
    embedding_model='all-MiniLM-L6-v2'
)

print("\n" + "="*60)
print("✓ Index building complete!")
print("="*60)

# ============================================
# CELL 4: Verify Files
# ============================================
import os

files = ["index_file.index", "index_file.meta.json", "index_file.chunks.json"]
total_size = 0

print("Generated files:")
for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f) / (1024 * 1024)
        total_size += size
        print(f"  ✓ {f} ({size:.2f} MB)")
    else:
        print(f"  ✗ {f} - NOT FOUND")

print(f"\nTotal size: {total_size:.2f} MB")

# ============================================
# CELL 5: Download Files
# ============================================
from google.colab import files

print("Downloading index files...")
for f in ["index_file.index", "index_file.meta.json", "index_file.chunks.json"]:
    if os.path.exists(f):
        files.download(f)
        print(f"  ✓ Downloaded {f}")

print("\n✓ All files downloaded!")
print("\nNext: Upload these files to Cloudflare R2")
