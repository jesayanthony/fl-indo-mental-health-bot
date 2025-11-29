# models/t5_cendol_chat.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

FEDAVG_PATH = "artifacts/t5_fedavg_demo"
LOCAL_DEMO_PATH = "artifacts/t5_local_demo"
BASE_MODEL_NAME = "indonlp/cendol-mt5-small-chat"
MODEL_NAME = "t5-small"

def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for path in [FEDAVG_PATH, LOCAL_DEMO_PATH]:
        try:
            print(f"Trying to load model from: {path}")
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSeq2SeqLM.from_pretrained(path)
            print(f"Loaded model from {path}")
            model.to(device)
            return tokenizer, model, device
        except Exception as e:
            print(f"Failed to load from {path}: {e}")

    # Fallback to base
    print("Loading base model instead.")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    model.to(device)
    return tokenizer, model, device

def generate_reply(tokenizer, model, device, user_message: str) -> str:
    # Simple single-turn prompt style
    prompt = f"Pengguna: {user_message}\nAsisten:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

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
    return reply

def main():
    tokenizer, model, device = load_model_and_tokenizer()

    # You can change this to anything you like
    user_message = "Aku merasa cemas dan sulit tidur akhir-akhir ini."
    print("USER:", user_message)

    reply = generate_reply(tokenizer, model, device, user_message)
    print("BOT :", reply)

if __name__ == "__main__":
    main()
