import faiss
import numpy as np
import json
import os

from config import VECTOR_STORE_DIR

class FAISSStore:
    def __init__(self, dimension: int, index_path=f"{VECTOR_STORE_DIR}"+"/index.faiss", metadata_path="vector_store/storage/metadata.json"):
        self.index_path = index_path
        self.metadata_path = metadata_path

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # Load existing index
            self.index = faiss.read_index(index_path)

            with open(metadata_path, "r") as f:
                self.data = json.load(f)
        else:
            # Create new index
            self.index = faiss.IndexFlatL2(dimension)
            self.data = []

    def add(self, embeddings, chunks, file_name):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)

        for chunk in chunks:
            self.data.append({
                "chunk_id": f"{file_name}_chunk_{len(self.data)}",
                "document": file_name,
                "text": chunk["text"],
                "page": chunk["page"]
            })

        self.save()

    def search(self, query_embedding, k=5, document=None, threshold=None):
        query_vector = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_vector, k * 3)

        results = []

        for distance, idx in zip(distances[0], indices[0]):
            print("Distance:", distance)
            if idx < 0 or idx >= len(self.data):
                continue

            item = self.data[idx]

            # Document filter
            if document and item.get("document") != document:
                continue

            # Similarity threshold
            if threshold is not None and distance > threshold:
                continue

            result = {
                **item,
                "score": round(float(distance), 4)
            }

            results.append(result)

            if len(results) >= k:
                break

        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "w") as f:
            json.dump(self.data, f)