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
MODEL_SUBDIR = os.getenv("MODEL_SUBDIR", "cendol_counseling_ft")
MODEL_LOCAL_DIR = os.getenv("MODEL_LOCAL_DIR", "/app/model")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "indonlp/cendol-mt5-small-chat")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)


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


class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: Optional[List[str]] = None


class ChatResponse(BaseModel):
    reply: str


app = FastAPI(title="Mental Health FL Chatbot")
print("[INFO] Starting up. Using device:", DEVICE)

# ----- Model load with robust fallback -----
from pathlib import Path

tokenizer = None
model = None

try:
    # Try to download FL model from GCS
    download_model_from_gcs(MODEL_BUCKET, MODEL_SUBDIR, MODEL_LOCAL_DIR)

    local_path = Path(MODEL_LOCAL_DIR)
    has_files = local_path.exists() and any(local_path.iterdir())

    if not has_files:
        print(f"[WARN] No model files found under {MODEL_LOCAL_DIR}, will try HF model.")
        raise FileNotFoundError(f"No model files under {MODEL_LOCAL_DIR}")

    print(f"[INFO] Loading tokenizer/model from local dir: {MODEL_LOCAL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_LOCAL_DIR)
    print("[INFO] Loaded model from GCS checkpoint.")

except Exception as e:
    print("[ERROR] Failed to load model from local dir:", e)
    print(f"[INFO] Trying to load fallback HF model: {FALLBACK_MODEL}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(FALLBACK_MODEL)
        print("[INFO] Loaded fallback HF model successfully.")
    except Exception as e2:
        print("[CRITICAL] Failed to load HF model as well:", e2)
        print("[INFO] Falling back to simple rule-based responder (no ML).")
        tokenizer = None
        model = None

if model is not None:
    model.to(DEVICE)
    model.eval()
    print("[INFO] Model ready.")
else:
    print("[WARN] No ML model loaded. Using rule-based responses only.")



PERSONA = """
Kamu adalah asisten konseling kesehatan mental berbahasa Indonesia.
Peranmu:
- Mendengarkan dengan empati dan tanpa menghakimi.
- Mengakui dan memvalidasi perasaan pengguna.
- Menjawab dengan kalimat pendek-pendek, jelas, dan hangat.
- Mengajukan 1–2 pertanyaan lanjutan yang lembut untuk memahami situasi.
- Mengingatkan bahwa kamu bukan psikolog/psikiater dan tidak bisa memberi diagnosis atau obat.
- Jika ada indikasi keinginan bunuh diri atau bahaya serius, sarankan untuk segera mencari bantuan profesional atau layanan darurat setempat.

Gaya bahasa:
- Gunakan kata ganti “kamu” untuk pengguna, dan “aku” untuk dirimu.
- Gunakan bahasa santai, sopan, dan menenangkan.
- Jangan mengulang persis kalimat pengguna, cukup rangkum perasaannya.
- Jangan menilai atau menyalahkan.

Sekarang bantu jawab keluhan pengguna dengan penuh empati.
"""


# ----- Model load -----
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


def cors_headers(request: Request) -> dict:
    origin = request.headers.get("origin") or "*"
    if origin == "null":
        origin = "*"
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }


@app.get("/health")
def health(request: Request):
    return JSONResponse(
        content={"status": "ok"},
        headers=cors_headers(request),
    )



@app.options("/chat")
async def options_chat(request: Request):
    return PlainTextResponse("", status_code=200, headers=cors_headers(request))

def rule_based_reply(message: str) -> str:
    msg_lower = message.lower()

    if any(word in msg_lower for word in ["cemas", "anxious", "takut", "anxiety"]):
        return (
            "Aku ikut merasakan kecemasan yang kamu alami. "
            "Rasanya pasti melelahkan kalau pikiran terus berputar dan sulit tenang.\n\n"
            "Coba tarik napas pelan beberapa kali dan sadari dulu apa yang paling sering kamu khawatirkan. "
            "Kalau kamu bersedia, kamu bisa ceritakan lebih detail kapan rasa cemas ini mulai muncul dan apa yang biasanya memicunya."
        )
    if any(word in msg_lower for word in ["sedih", "down", "depres", "putus asa"]):
        return (
            "Terima kasih sudah berani cerita. Kedengarannya kamu sedang merasa sangat sedih dan mungkin cukup sendirian.\n\n"
            "Perasaan itu valid, dan wajar kalau kamu merasa lelah. "
            "Apa yang belakangan ini paling sering membuat kamu merasa seperti ini?"
        )
    if any(word in msg_lower for word in ["marah", "kesal", "frustrasi"]):
        return (
            "Sepertinya kamu sedang memendam banyak rasa marah atau kesal. "
            "Emosi seperti itu wajar dan penting untuk diakui, bukan dipendam terus.\n\n"
            "Apa yang terjadi belakangan ini sampai membuatmu merasa seperti itu?"
        )
    if any(word in msg_lower for word in ["tidur", "insomnia", "susah tidur"]):
        return (
            "Sulit tidur memang bisa sangat mengganggu, baik fisik maupun mental. "
            "Tubuh dan pikiran jadi terasa makin lelah.\n\n"
            "Biasanya sebelum tidur, apa yang paling sering ada di pikiranmu?"
        )

    # generic fallback
    return (
        "Aku dengar kamu sedang mengalami masa yang cukup berat. "
        "Terima kasih sudah mau berbagi cerita di sini.\n\n"
        "Boleh ceritakan lebih detail apa yang sedang kamu rasakan dan apa yang terjadi belakangan ini?"
    )



@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    history_text = ""
    if req.history:
        last_turns = req.history[-6:]
        history_text = "\n".join(last_turns)

    # If no ML model loaded, use rule-based responder
    if model is None or tokenizer is None:
        reply = rule_based_reply(req.message)
        return JSONResponse(
            content={"reply": reply},
            headers=cors_headers(request),
        )

    # --- ML path below: use persona + model ---
    prompt_parts = [PERSONA.strip()]

    if history_text:
        prompt_parts.append("Riwayat percakapan sebelumnya:")
        prompt_parts.append(history_text)

    prompt_parts.append(f"Pengguna: {req.message}")
    prompt_parts.append("Asisten:")

    prompt = "\n\n".join(prompt_parts)

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
            temperature=0.7,
        )

    raw_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Asisten:" in raw_reply:
        reply = raw_reply.split("Asisten:", maxsplit=1)[-1].strip()
    else:
        reply = raw_reply.strip()

    if not reply:
        reply = rule_based_reply(req.message)

    return JSONResponse(
        content={"reply": reply},
        headers=cors_headers(request),
    )
