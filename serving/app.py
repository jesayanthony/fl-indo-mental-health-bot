# serving/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: Optional[List[str]] = None

class ChatResponse(BaseModel):
    reply: str

app = FastAPI(title="Mental Health FL Chatbot (CORS test)")

# --- CORS: allow everything for now (dev only) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all origins
    allow_credentials=False,    # must be False if using "*"
    allow_methods=["*"],        # allow all HTTP methods
    allow_headers=["*"],        # allow all headers
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # simple echo so we know it's our app responding
    return ChatResponse(
        reply=f"BOT: aku menerima pesannya: {req.message}"
    )
