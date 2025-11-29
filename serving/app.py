# serving/app.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: Optional[List[str]] = None

class ChatResponse(BaseModel):
    reply: str

app = FastAPI(title="Mental Health FL Chatbot - Minimal")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # For now, just echo back something simple
    return ChatResponse(
        reply=f"Halo {req.user_id}, ini jawaban dummy untuk pesan: {req.message}"
    )
