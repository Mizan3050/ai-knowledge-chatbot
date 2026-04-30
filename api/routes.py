from fastapi import APIRouter, UploadFile, File
import shutil
import os
from utils.file_loader import extract_text_from_pdf
from utils.chunking import chunk_text
from services.embedding_service import get_embeddings
from vector_store.faiss_store import FAISSStore
from services.embedding_service import get_embeddings
from services.llm_service import generate_answer


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

    # Extract text
    text = extract_text_from_pdf(file_path)

    # Chunk text
    chunks = chunk_text(text)

    # Generate embeddings
    embeddings = get_embeddings(chunks)

    # Initialize FAISS store
    dimension = len(embeddings[0])
    vector_store = FAISSStore(dimension)

    # Store data
    vector_store.add(embeddings, chunks)

    return {
        "message": "Document processed and stored",
        "chunks_stored": len(chunks)
    }

@router.post("/chat")
async def chat(query: str):
    global vector_store

    if vector_store is None:
        return {"error": "No document uploaded yet"}

    # Step 1: query embedding
    query_embedding = get_embeddings([query])[0]

    # Step 2: retrieve relevant chunks
    results = vector_store.search(query_embedding, k=3)

    # Step 3: generate answer
    answer = generate_answer(query, results)

    return {
        "answer": answer,
        "sources": results
    }