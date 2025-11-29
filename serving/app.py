# serving/app.py

import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- Config ----------
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "mental-health-fl-bot")
MODEL_SUBDIR = os.getenv("MODEL_SUBDIR", "t5_fedavg_demo")
MODEL_LOCAL_DIR = os.getenv("MODEL_LOCAL_DIR", "/app/model")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "indonlp/cendol-mt5-small-chat")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)


# ---------- Helper to download model from GCS ----------
def download_model_from_gcs(bucket_name: str, subdir: str, local_dir: str):
    local_path = Path(local_dir)
    if local_path.exists() and any(local_path.iterdir()):
        print(f"[INFO] Local model dir {local_dir} already has files, skipping download.")
        return

    print(f"[INFO] Downloading model from gs://{bucket_name}/{subdir} to {local_dir} ...")
    t0 = time.time()
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=subdir))
        if not blobs:
            print(f"[WARN] No blobs found under {subdir}. Check bucket/path.")

        for blob in blobs:
            rel_path = blob.name[len(subdir):].lstrip("/")
            dest_path = local_path / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Downloading {blob.name} -> {dest_path}")
            blob.download_to_filename(str(dest_path))

        print(f"[INFO] Download finished in {time.time() - t0:.2f}s")
    except Exception as e:
        print("[ERROR] Failed to download model from GCS:", e)
        # we'll fall back to HF base model below


# ---------- Request/response models ----------
class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: Optional[List[str]] = None


class ChatResponse(BaseModel):
    reply: str


# ---------- App & model init ----------
app = FastAPI(title="Mental Health FL Chatbot")
print("[INFO] Starting up. Using device:", DEVICE)

# Load model (GCS -> local -> load; fallback to HF if needed)
try:
    download_model_from_gcs(MODEL_BUCKET, MODEL_SUBDIR, MODEL_LOCAL_DIR)
    print(f"[INFO] Loading tokenizer/model from {MODEL_LOCAL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_LOCAL_DIR)
    print("[INFO] Loaded model from GCS checkpoint.")
except Exception as e:
    print("[ERROR] Failed to load model from local dir:", e)
    print(f"[INFO] Falling back to HF model: {FALLBACK_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(FALLBACK_MODEL)

model.to(DEVICE)
model.eval()
print("[INFO] Model ready.")


# ---------- CORS helper ----------
def cors_headers(request: Request) -> dict:
    origin = request.headers.get("origin") or "*"
    # We don't use credentials, so "*" is safe.
    return {
        "Access-Control-Allow-Origin": origin if origin != "null" else "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }


# ---------- Routes ----------
@app.get("/health")
def health(request: Request):
    return JSONResponse(content={"status": "ok"}, headers=cors_headers(request))


# Preflight handler for /chat
@app.options("/chat")
async def options_chat(request: Request):
    return PlainTextResponse("", status_code=200, headers=cors_headers(request))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    # Build simple conversational prompt
    history_text = ""
    if req.history:
        history_text = "\n".join(req.history[-5:])

    prompt = ""
    if history_text:
        prompt += f"Riwayat percakapan sebelumnya:\n{history_text}\n\n"
    prompt += f"Pengguna: {req.message}\nAsisten:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Manually attach CORS headers to response
    return JSONResponse(
        content={"reply": reply},
        headers=cors_headers(request),
    )
