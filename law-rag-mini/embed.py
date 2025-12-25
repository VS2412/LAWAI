# embed.py
import os
import glob
import json
from tqdm import tqdm
import numpy as np
import chromadb
from chromadb.config import Settings

# fallback embedding model
try:
    # try nomic local embed (if installed)
    from nomic import embed as nomic_embed
    def get_embeddings(texts):
        # nomic-embed-text exposes embed() or embed_documents depending on version.
        # Try common call patterns.
        try:
            embs = nomic_embed.embed(texts)
        except Exception:
            embs = nomic_embed.embed_documents(texts)
        return [np.array(e, dtype=float) for e in embs]
    print("Using nomic embed backend.")
except Exception:
    # fallback to sentence-transformers
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    def get_embeddings(texts):
        return [emb.astype(float) for emb in sbert.encode(texts, show_progress_bar=False)]
    print("nomic not available — using sentence-transformers fallback.")

# simple chunker (character-based)
def chunk_text(text, max_chars=1200, overlap=200):
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = min(i + max_chars, L)
        chunk = text[i:end]
        chunks.append(chunk.strip())
        i += max_chars - overlap
    return chunks

DATA_DIR = "data/processed/constitution/"
PERSIST_DIR = "chromadb_persist"

# create chroma client with persistence
client = chromadb.PersistentClient(path=PERSIST_DIR)

COLLECTION_NAME = "constitution_articles"
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    collection = client.get_collection(COLLECTION_NAME)
else:
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"source":"mini-constitution"})

# read files and build embeddings
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
print("FILES LOADED:", files)

if not files:
    raise SystemExit(f"No files found in {DATA_DIR}. Put 10-20 article .txt files there.")

to_insert_ids = []
to_insert_embeddings = []
to_insert_metadatas = []
to_insert_documents = []

for fpath in files:
    fname = os.path.basename(fpath)
    with open(fpath, 'r', encoding='utf-8') as fh:
        text = fh.read().strip()
    chunks = chunk_text(text)
    # embed in batches
    B = 16
    for i in range(0, len(chunks), B):
        batch = chunks[i:i+B]
        embs = get_embeddings(batch)
        for j, emb in enumerate(embs):
            uid = f"{fname}::chunk_{i+j}"
            to_insert_ids.append(uid)
            to_insert_embeddings.append(list(map(float, emb)))
            to_insert_metadatas.append({"source_file": fname, "chunk_index": i+j})
            to_insert_documents.append(batch[j])

# upsert into chroma
print(f"Upserting {len(to_insert_ids)} vectors into Chroma collection '{COLLECTION_NAME}' (persist dir: {PERSIST_DIR})")
collection.upsert(
    ids=to_insert_ids,
    embeddings=to_insert_embeddings,
    metadatas=to_insert_metadatas,
    documents=to_insert_documents
)

# persist
print("Done. Vector DB persisted.")
