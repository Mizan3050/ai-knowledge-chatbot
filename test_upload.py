from fastapi import FastAPI, UploadFile, File
from typing import List

app = FastAPI()

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    return {
        "files": [file.filename for file in files]
    }