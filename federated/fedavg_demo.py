# federated/fedavg_demo.py

import os, sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from copy import deepcopy
from typing import List, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from models.t5_cendol_data import ChatDataset

MODEL_NAME = "t5-small"
SAVE_DIR = "artifacts/t5_fedavg_demo"


def get_client_datasets(tokenizer) -> Dict[int, ChatDataset]:
    """
    Simulate 3 clients with slightly different conversations.
    Later you'll replace this with real Psychika/AMoD splits.
    """
    client_data: Dict[int, List[Dict[str, str]]] = {
        0: [
            {
                "user": "Aku merasa cemas dan sulit tidur akhir-akhir ini.",
                "bot": "Terima kasih sudah berbagi. Coba latih pernapasan pelan dan buat rutinitas sebelum tidur agar tubuh lebih rileks."
            },
            {
                "user": "Aku takut menghadapi hari Senin karena banyak tugas dan deadline.",
                "bot": "Itu perasaan yang umum. Coba pecah tugas menjadi bagian kecil dan prioritaskan satu hal yang bisa kamu selesaikan hari ini."
            },
        ],
        1: [
            {
                "user": "Aku merasa tidak percaya diri saat presentasi di depan kelas.",
                "bot": "Kecemasan saat presentasi sangat wajar. Coba latih presentasi di depan cermin atau teman dekat untuk membangun kepercayaan diri."
            },
            {
                "user": "Aku sering membandingkan diriku dengan teman-teman yang lebih sukses.",
                "bot": "Membandingkan diri bisa membuatmu lelah. Fokus pada langkah kecil yang bisa kamu kontrol dan apresiasi perkembanganmu sendiri."
            },
        ],
        2: [
            {
                "user": "Aku merasa capek secara mental, tapi tidak tahu harus cerita ke siapa.",
                "bot": "Kamu tidak sendirian. Menulis jurnal atau mencari komunitas yang suportif bisa jadi langkah awal untuk berbagi perasaan."
            },
            {
                "user": "Aku sering merasa sendiri meskipun sedang bersama banyak orang.",
                "bot": "Rasa kesepian di tengah keramaian bisa terasa berat. Coba cari satu hubungan yang lebih dalam dengan orang yang kamu percaya."
            },
        ],
    }

    datasets: Dict[int, ChatDataset] = {}
    for cid, pairs in client_data.items():
        datasets[cid] = ChatDataset(tokenizer, pairs, max_length=256)
    return datasets

def client_update(
    global_model: AutoModelForSeq2SeqLM,
    dataset: ChatDataset,
    device: str,
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 5e-5,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Perform local training on one client's data starting from global_model.
    Returns (updated_state_dict, num_samples).
    """

    # 1. Clone global model weights so we don't mutate the original directly
    local_model = deepcopy(global_model)
    local_model.to(device)
    local_model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr)

    num_samples = 0

    for _ in range(epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = local_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
            optimizer.step()

            num_samples += input_ids.size(0)

    # 2. Return the updated weights and number of samples used
    return local_model.state_dict(), num_samples

def fedavg_aggregate(
    client_weights: List[Dict[str, torch.Tensor]],
    client_nums: List[int],
) -> Dict[str, torch.Tensor]:
    """
    Standard FedAvg:
    global_weight = sum_i (n_i / N_total) * w_i
    """
    total_samples = sum(client_nums)
    assert total_samples > 0, "Total samples must be > 0"

    # Initialize global state dict with zeros of same shape
    global_state = {}
    for key in client_weights[0].keys():
        global_state[key] = torch.zeros_like(client_weights[0][key])

    for cw, n_i in zip(client_weights, client_nums):
        weight = n_i / total_samples
        for key in global_state.keys():
            global_state[key] += cw[key] * weight

    return global_state

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    global_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    global_model.to(device)

    # Build client datasets
    client_datasets = get_client_datasets(tokenizer)
    client_ids = list(client_datasets.keys())
    print("Clients:", client_ids)

    num_rounds = 2  # keep small for demo
    local_epochs = 1

    for rnd in range(num_rounds):
        print(f"\n--- Federated Round {rnd + 1}/{num_rounds} ---")

        client_weights = []
        client_nums = []

        for cid in client_ids:
            print(f" Client {cid}: local training...")
            ds = client_datasets[cid]

            updated_state, n_i = client_update(
                global_model=global_model,
                dataset=ds,
                device=device,
                epochs=local_epochs,
                batch_size=2,
                lr=5e-5,
            )

            client_weights.append(updated_state)
            client_nums.append(n_i)
            print(f"  -> used {n_i} samples")

        # Aggregate
        print(" Aggregating client updates with FedAvg...")
        new_global_state = fedavg_aggregate(client_weights, client_nums)
        global_model.load_state_dict(new_global_state)

        # Optional: quick sanity check – generate once per round
        global_model.eval()
        with torch.no_grad():
            test_text = "Aku merasa cemas dan sulit tidur akhir-akhir ini."
            prompt = f"Pengguna: {test_text}\nAsisten:"
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)
            outputs = global_model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
            )
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(" Sample reply after round", rnd + 1, "->", reply)

    # Save final global model
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\nSaving FedAvg global model to {SAVE_DIR}")
    global_model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    print("Done FedAvg demo.")

if __name__ == "__main__":
    main()
