import pymupdf
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path 
import numpy as np

doc = pymupdf.open("data/lecture.pdf")
# print("Number of pages: ", len(doc))

pages = []

for i in range(len(doc)):
    text = doc[i].get_text()
    page_data = {
        "source": "lecture.pdf",
        "page": i+1,
        "text": text
    }

    pages.append(page_data)

doc.close


def chunk_pages(pages, chunk_size = 100, overlap = 20):
    chunks = []

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    step = chunk_size - overlap

    for page_data in pages:
        if page_data["page"] == 1:
            continue

        words = page_data["text"].split()

        for start in range(0, len(words), step):
            chunk_text = " ".join(words[start:start+chunk_size])

            chunks.append({
                "source": page_data["source"],
                "page": page_data["page"],
                "text": chunk_text
            })
    return chunks

chunks = chunk_pages(pages)

# print("Pages processed:", len(pages))
# print("Total chunks:", len(chunks))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    batch_size = 32,
    show_progress_bar=True,
    normalize_embeddings=True
)

Path("index").mkdir(exist_ok = True)

np.save("index/embeddings.npy", embeddings)

with open("index/chunks.json", "w", encoding = "utf-8") as file:
    json.dump(chunks, file, ensure_ascii = False, indent = 2)

print("Embeddings and chunks saved.")

# print("Embedding's shape:", embeddings.shape)
