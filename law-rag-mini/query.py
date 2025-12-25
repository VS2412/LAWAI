# query.py
import os
import sys
import json
import argparse
import requests
import subprocess
from typing import List
import chromadb
from chromadb.config import Settings
import numpy as np

# embeddings same strategy as embed.py
try:
    from nomic import embed as nomic_embed
    def get_embeddings(texts):
        try:
            embs = nomic_embed.embed(texts)
        except Exception:
            embs = nomic_embed.embed_documents(texts)
        return [np.array(e, dtype=float) for e in embs]
    print("Using nomic embed backend.")
except Exception:
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    def get_embeddings(texts):
        return [emb.astype(float) for emb in sbert.encode(texts, show_progress_bar=False)]
    print("nomic not available — using sentence-transformers fallback.")

PERSIST_DIR = "chromadb_persist"
COLLECTION_NAME = "constitution_articles"
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(COLLECTION_NAME)

def retrieve(question: str, k=4):
    q_emb = get_embeddings([question])[0].tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=['documents','metadatas','distances']
    )
    hits = []
    for i in range(len(results['ids'][0])):
        hits.append({
            "id": results['ids'][0][i],
            "doc": results['documents'][0][i],
            "meta": results['metadatas'][0][i],
            "distance": results['distances'][0][i]
        })
    return hits

def build_prompt(hits, question):
    # strict instruction to LLM: only use provided context
    context_texts = []
    sources = []
    for h in hits:
        context_texts.append(f"--- SOURCE: {h['id']}\n{h['doc']}")
        sources.append(h['id'])
    context_block = "\n\n".join(context_texts)
    prompt = f"""
You are a legal assistant. ONLY use the provided SOURCE CONTEXTS below — do NOT hallucinate or use external knowledge. If the answer is not contained in the contexts, say "I don't know based on the provided documents."

TASK: For the QUESTION below, produce a multi-part, detailed explanation aimed at a law student:
1) Short answer (1-2 sentences).
2) Clause-by-clause breakdown: quote the exact clause(s) from the provided sources (wrap quotes in " " and cite the source id).
3) Explanation in plain English: translate legal wording into clear, precise points.
4) Practical implications and limits: describe real-world legal effects, who gains power, who is constrained, common misunderstandings.
5) Example scenarios (1-2 short examples showing how the article would apply).
6) Related provisions to check (list other source ids pulled from contexts that are relevant).
7) If anything needed for a legally correct answer is missing from the sources, say exactly what is missing and why.

STYLE: Formal but readable. Use numbered bullets for breakdowns. Cite source ids inline for every factual claim (e.g., (source: article_001.txt::chunk_0)). Keep the short answer concise, the clause breakdown literal, and the explanation helpful for exam prep.

CONTEXTS:
{context_block}

QUESTION:
{question}

Answer:
"""
    return prompt, sources

def call_ollama_http(model: str, prompt: str, max_tokens=512, temperature=0.0):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # other params can be added depending on Ollama version
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # many Ollama versions stream; adapt if you get a different schema
        if isinstance(data, dict) and "text" in data:
            return data["text"]
        # fallback: try to join choices
        if "choices" in data:
            return "".join(c.get("text","") for c in data["choices"])
        return json.dumps(data)
    except Exception as e:
        raise


def call_ollama_cli(model: str, prompt: str):
    try:
        cmd = ["ollama", "run", model]
        p = subprocess.run(
            cmd,
            input=prompt,      # feed prompt to stdin
            text=True,
            capture_output=True,
            timeout=120
        )
        if p.returncode == 0:
            return p.stdout
        else:
            return p.stderr or f"ollama CLI failed with code {p.returncode}"
    except Exception as e:
        return str(e)


def generate_answer(model: str, prompt: str):
    # try HTTP API
    try:
        return call_ollama_http(model, prompt)
    except Exception as e_http:
        # fallback to CLI
        ans = call_ollama_cli(model, prompt)
        return ans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1", help="Ollama model name (local)")
    parser.add_argument("--k", type=int, default=4, help="Number of retrieved chunks")
    parser.add_argument("--question", type=str, help="Question text (or omit to be prompted)")
    args = parser.parse_args()

    if args.question:
        question = args.question
    else:
        question = input("Enter question: ").strip()

    hits = retrieve(question, k=args.k)
    prompt, sources = build_prompt(hits, question)
    answer = generate_answer(args.model, prompt)
    print("\n--- RAG RESULT ---\n")
    print("SOURCES USED:")
    for s in sources:
        print(" -", s)
    print("\nANSWER:\n")
    print(answer.strip())

if __name__ == "__main__":
    main()
