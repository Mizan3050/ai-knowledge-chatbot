from fastapi import APIRouter, UploadFile, File
import shutil
import os
from utils.file_loader import extract_text_from_pdf
from utils.chunking import chunk_text

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text
    text = extract_text_from_pdf(file_path)

    # Chunk text
    chunks = chunk_text(text)

    return {
        "message": "File processed successfully",
        "total_chunks": len(chunks),
        "sample_chunk": chunks[0] if chunks else ""
    }