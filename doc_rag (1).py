
import faiss
import os
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents(folder):
    texts = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(folder, file))
            for page in reader.pages:
                texts.append(page.extract_text())
    return texts

def create_index(texts):
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, texts

def retrieve(query, index, texts, k=3):
    q_emb = model.encode([query])
    _, idx = index.search(np.array(q_emb), k)
    return [texts[i] for i in idx[0]]
ffmpeg -i audio13.mp4 audio13.wav
