from fastapi import APIRouter, UploadFile, File
import shutil
import os
from config import UPLOAD_DIR
from services.embedding_service import get_embeddings
from vector_store.faiss_store import FAISSStore
from services.embedding_service import get_embeddings
from services.llm_service import generate_answer
from utils.file_loader import extract_text_by_page
from utils.chunking import chunk_pages
from datetime import datetime
from fastapi.responses import FileResponse
from fastapi import HTTPException

# Initialize store (temporary, in-memory)
vector_store = None

router = APIRouter()

os.makedirs(UPLOAD_DIR, exist_ok=True)

FILE_UPLOAD_SCHEMA = {
    "requestBody": {
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"}
                        }
                    },
                    "required": ["files"]
                }
            }
        }
    }
}

@router.post("/upload", openapi_extra=FILE_UPLOAD_SCHEMA)
async def upload_files(files: list[UploadFile] = File()):
    global vector_store

    total_chunks = 0
    processed_files = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pages = extract_text_by_page(file_path)
        chunks = chunk_pages(pages)
        texts = [chunk["text"] for chunk in chunks]
        embeddings = get_embeddings(texts)

        if vector_store is None:
            dimension = len(embeddings[0])
            vector_store = FAISSStore(dimension)

        vector_store.add(embeddings, chunks, file.filename)

        total_chunks += len(chunks)
        processed_files.append({
            "filename": file.filename,
            "chunks": len(chunks)
        })

    return {
        "message": "Documents processed with metadata",
        "total_files": len(processed_files),
        "total_chunks": total_chunks,
        "files": processed_files
    }

@router.post("/chat")
async def chat(query: str, document: str = None):
    global vector_store

    if vector_store is None:
        return {"error": "No document uploaded yet"}

    # Step 1: query embedding
    query_embedding = get_embeddings([query])[0]

    # Step 2: retrieve chunks (with metadata)
    results = vector_store.search(
    query_embedding,
    k=5,
    document=document,
    threshold=15
)
    filtered_results = []

    for item in results:
        overlap = keyword_overlap(query, item["text"])

        if overlap >= 1:
            filtered_results.append(item)

    # Extract only text for LLM
    context_chunks = [
    item["text"]
    for item in filtered_results
    if item.get("text")
]

    if not filtered_results:
        return {
            "answer": "No relevant information found.",
            "sources": []
        }

    # Step 3: generate answer
    answer = generate_answer(query, context_chunks)

    return {
        "answer": answer,
        "sources": results
    }

@router.get("/documents")
async def list_documents():
    documents = []

    for filename in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, filename)

        if os.path.isfile(path):
            stat = os.stat(path)

            documents.append({
                "name": filename,
                "size": stat.st_size,
                "uploaded_at": datetime.fromtimestamp(
                    stat.st_ctime
                ),
                "extension": filename.split(".")[-1]
            })

    return {
        "documents": documents
    }

@router.get("/documents/{filename}")
async def get_document(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/pdf"
    )

def keyword_overlap(query, text):
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())

    overlap = query_words.intersection(text_words)

    return len(overlap)