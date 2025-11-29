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

# --------- Config from env / defaults ----------
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "mental-health-fl-bot")
MODEL_SUBDIR = os.getenv("MODEL_SUBDIR", "t5_fedavg_demo")  # path inside bucket
MODEL_LOCAL_DIR = os.getenv("MODEL_LOCAL_DIR", "/app/model")

FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "indonlp/cendol-mt5-small-chat")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)  # keep CPU usage lighter


# --------- Helpers ----------
def download_model_from_gcs(bucket_name: str, subdir: str, local_dir: str):
    """Download model files from GCS into local_dir, if not already present."""
    local_path = Path(local_dir)
    if local_path.exists() and any(local_path.iterdir()):
        print(f"[INFO] Local model dir {local_dir} already populated, skipping download.")
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
            # e.g. subdir = "t5_fedavg_demo", blob.name = "t5_fedavg_demo/config.json"
            rel_path = blob.name[len(subdir):].lstrip("/")  # "config.json"
            dest_path = local_path / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Downloading {blob.name} -> {dest_path}")
            blob.download_to_filename(str(dest_path))

        print(f"[INFO] Download finished in {time.time() - t0:.2f}s")
    except Exception as e:
        print("[ERROR] Failed to download model from GCS:", e)
        # don't raise; we’ll try fallback model later


class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: Optional[List[str]] = None


class ChatResponse(BaseModel):
    reply: str


app = FastAPI(title="Mental Health FL Chatbot")

print("[INFO] Starting up chatbot service...")
print(f"[INFO] Using device: {DEVICE}")

# --------- Model loading on startup ----------
tokenizer = None
model = None

try:
    # Try to download FL model from GCS and load it
    download_model_from_gcs(MODEL_BUCKET, MODEL_SUBDIR, MODEL_LOCAL_DIR)
    print(f"[INFO] Loading tokenizer/model from local dir: {MODEL_LOCAL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_LOCAL_DIR)
    print("[INFO] Loaded model from GCS checkpoint.")
except Exception as e:
    print("[ERROR] Failed to load model from GCS directory:", e)
    print(f"[INFO] Falling back to base HF model: {FALLBACK_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(FALLBACK_MODEL)

model.to(DEVICE)
model.eval()
print("[INFO] Model ready.")


# --------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history_text = ""
    if req.history:
        # last 5 turns only
        history_text = "\n".join(req.history[-5:])

    prompt = ""
    if history_text:
        prompt += f"Riwayat percakapan sebelumnya:\n{history_text}\n\n"
    prompt += f"Pengguna: {req.message}\nAsisten:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,   # shorter for memory
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
    return ChatResponse(reply=reply)
