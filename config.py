import os

BASE_DATA_DIR = os.getenv("DATA_DIR", "/data")

UPLOAD_DIR = os.path.join(BASE_DATA_DIR, "uploads")

VECTOR_STORE_DIR = os.path.join(BASE_DATA_DIR, "vector_store")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)