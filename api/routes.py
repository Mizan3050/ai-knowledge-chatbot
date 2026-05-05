from fastapi import APIRouter, UploadFile, File
import shutil
import os
from services.embedding_service import get_embeddings
from vector_store.faiss_store import FAISSStore
from services.embedding_service import get_embeddings
from services.llm_service import generate_answer
from utils.file_loader import extract_text_by_page
from utils.chunking import chunk_pages

# Initialize store (temporary, in-memory)
vector_store = None

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vector_store

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract page-wise text
    pages = extract_text_by_page(file_path)

    # Chunk with page info
    chunks = chunk_pages(pages)

    # Extract only text for embeddings
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings
    embeddings = get_embeddings(texts)

    # Initialize FAISS
    dimension = len(embeddings[0])
    vector_store = FAISSStore(dimension)

    # Store with metadata
    vector_store.add(embeddings, chunks, file.filename)

    return {
        "message": "Document processed with metadata",
        "total_chunks": len(chunks)
    }
@router.post("/chat")
async def chat(query: str):
    global vector_store

    if vector_store is None:
        return {"error": "No document uploaded yet"}

    # Step 1: query embedding
    query_embedding = get_embeddings([query])[0]

    # Step 2: retrieve chunks (with metadata)
    results = vector_store.search(query_embedding, k=5)

    # Extract only text for LLM
    context_chunks = [item["text"] for item in results]

    # Step 3: generate answer
    answer = generate_answer(query, context_chunks)

    return {
        "answer": answer,
        "sources": results
    }