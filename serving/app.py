# serving/app.py

import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------- Config from environment ----------
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "mental-health-fl-bot")
MODEL_SUBDIR = os.getenv("MODEL_SUBDIR", "t5_fedavg_demo")  # in your bucket
MODEL_LOCAL_DIR = os.getenv("MODEL_LOCAL_DIR", "/app/model")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- GCS download helper ----------
def download_model_from_gcs(bucket_name: str, subdir: str, local_dir: str):
    """
    Download all files from gs://bucket_name/subdir into local_dir.
    If local_dir is already populated, we skip downloading.
    """
    local_path = Path(local_dir)
    if local_path.exists() and any(local_path.iterdir()):
        print(f"[INFO] Local model dir {local_dir} already has files, skipping download.")
        return

    print(f"[INFO] Downloading model from gs://{bucket_name}/{subdir} to {local_dir} ...")
    t0 = time.time()

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=subdir)
    for blob in blobs:
        # e.g. subdir='t5_fedavg_demo', blob.name='t5_fedavg_demo/config.json'
        rel_path = blob.name[len(subdir):].lstrip("/")  # 'config.json'
        dest_path = local_path / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Downloading {blob.name} -> {dest_path}")
        blob.download_to_filename(str(dest_path))

    print(f"[INFO] Download finished in {time.time() - t0:.2f} seconds.")


# ---------- FastAPI schemas ----------
class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: Optional[List[str]] = None


class ChatResponse(BaseModel):
    reply: str


# ---------- Initialize FastAPI & model ----------
app = FastAPI(title="Mental Health FL Chatbot")

print("[INFO] Starting up, preparing model...")

download_model_from_gcs(MODEL_BUCKET, MODEL_SUBDIR, MODEL_LOCAL_DIR)

print(f"[INFO] Loading tokenizer/model from {MODEL_LOCAL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_LOCAL_DIR)
model.to(DEVICE)
model.eval()

print("[INFO] Model loaded and ready.")


# ---------- Simple healthcheck ----------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Chat endpoint ----------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Simple context concatenation
    history_text = ""
    if req.history:
        # last few messages from bot only, or you can keep full convo style
        history_text = "\n".join(req.history[-5:])

    # You can design a richer prompt, for now keep it simple
    prompt = ""
    if history_text:
        prompt += f"Riwayat percakapan:\n{history_text}\n\n"
    prompt += f"Pengguna: {req.message}\nAsisten:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
        )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return ChatResponse(reply=reply)
