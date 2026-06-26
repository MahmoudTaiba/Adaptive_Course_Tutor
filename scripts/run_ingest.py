"""
scripts/run_ingest.py — Ingest a PDF into the Qdrant vector store.

Usage:
    python scripts/run_ingest.py

Edit the `path` and `source` variables below to point at your PDF.
"""

import sys
import os

# Ensure project root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import build_index, chunk_page, extract_pages

# ── Configure your PDF here ────────────────────────────────────────────────────
path = "attention is all you need.pdf"   # path to your PDF
source = "attention"                      # short name used in chunk_ids
# ──────────────────────────────────────────────────────────────────────────────

chunks = []
for page, text in extract_pages(path):
    chunks.extend(chunk_page(text, source, page))

print(f"{len(chunks)} chunks extracted")
build_index(chunks)
print("Index built ✅")
