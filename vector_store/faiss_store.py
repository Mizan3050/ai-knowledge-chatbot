import faiss
import numpy as np

class FAISSStore:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.data = []

    def add(self, embeddings, chunks, file_name):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)

        for i, chunk in enumerate(chunks):
            self.data.append({
                "chunk_id": f"{file_name}_chunk_{len(self.data)}",
                "text": chunk["text"],
                "page": chunk["page"]
            })

    def search(self, query_embedding, k=5):
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, k)

        results = [self.data[i] for i in indices[0] if i < len(self.data)]
        return results