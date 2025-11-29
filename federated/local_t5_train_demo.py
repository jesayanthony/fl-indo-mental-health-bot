# federated/local_t5_train_demo.py

import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

from models.t5_cendol_data import ChatDataset

MODEL_NAME = "t5-small"
SAVE_DIR = "artifacts/t5_local_demo"

def get_dummy_pairs() -> List[Dict[str, str]]:
    """
    Tiny dummy dataset just to prove training works.
    Later, replace this with real Psychika / AMoD samples.
    """
    return [
        {
            "user": "Aku merasa cemas dan sulit tidur akhir-akhir ini.",
            "bot": "Maaf ya, itu pasti melelahkan sekali. Coba mulai dengan atur napas pelan-pelan, dan kalau bisa buat rutinitas sebelum tidur supaya tubuhmu lebih rileks."
        },
        {
            "user": "Aku sering merasa tidak cukup baik dibandingkan teman-teman kampusku.",
            "bot": "Perasaan itu wajar, apalagi saat melihat pencapaian orang lain. Tapi nilai dirimu tidak hanya diukur dari prestasi. Coba apresiasi hal-hal kecil yang sudah kamu lakukan setiap hari."
        },
        {
            "user": "Aku merasa sangat capek secara mental, tapi tidak tahu harus cerita ke siapa.",
            "bot": "Terima kasih sudah berani cerita di sini. Kamu tidak sendirian. Kadang menuliskan perasaan di jurnal atau mencari satu orang yang kamu percaya bisa membantu meringankan beban itu."
        },
    ]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)

    # --- Dataset & DataLoader ---
    pairs = get_dummy_pairs()
    train_dataset = ChatDataset(tokenizer, pairs, max_length=256)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 1  # keep small for demo
    total_steps = num_epochs * len(train_loader)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            # Move tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if (batch_idx + 1) % 1 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Step [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}  AvgLoss: {avg_loss:.4f}"
                )

    # --- Save fine-tuned model ---
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Saving model to {SAVE_DIR}")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("Done training demo.")

if __name__ == "__main__":
    main()
