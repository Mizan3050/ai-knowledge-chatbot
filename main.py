from fastapi import FastAPI
from api.routes import router
from vector_store.faiss_store import FAISSStore
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load FAISS store on startup
vector_store = FAISSStore(dimension=1536)  # embedding size for OpenAI model

app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)