"""
scripts/eval_retrieval.py — Retrieval evaluation harness (Part A).
Reads data/eval/qa.jsonl and reports Hit@k, Recall@k, MRR.

qa.jsonl format (one JSON object per line):
    {"question": "...", "relevant_chunk_ids": ["attention_p3_0", ...]}

Usage:
    python scripts/eval_retrieval.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieve import retrieve_chunks

QA_PATH = "data/eval/qa.jsonl"
TOP_K = 5


def evaluate(qa_path: str, top_k: int = TOP_K):
    with open(qa_path) as f:
        examples = [json.loads(l) for l in f if l.strip()]

    if not examples:
        print("No eval examples found in qa.jsonl — add some first.")
        return

    hit_k = recall_k = mrr = 0.0

    for ex in examples:
        question = ex["question"]
        relevant = set(ex["relevant_chunk_ids"])
        results = retrieve_chunks(question, top_k=top_k)
        retrieved_ids = [r.chunk_id for r in results]

        # Hit@k — was any relevant chunk in the top-k?
        hit_k += int(any(cid in relevant for cid in retrieved_ids))

        # Recall@k — fraction of relevant chunks retrieved
        retrieved_set = set(retrieved_ids)
        recall_k += len(relevant & retrieved_set) / len(relevant) if relevant else 0.0

        # MRR — 1 / rank of first relevant result
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in relevant:
                mrr += 1.0 / rank
                break

    n = len(examples)
    print(f"Eval over {n} examples (top_k={top_k}):")
    print(f"  Hit@{top_k}:    {hit_k/n:.3f}")
    print(f"  Recall@{top_k}: {recall_k/n:.3f}")
    print(f"  MRR:       {mrr/n:.3f}")


if __name__ == "__main__":
    evaluate(QA_PATH)
