# models/t5_cendol_data.py

from torch.utils.data import Dataset
from typing import List, Dict, Any

class ChatDataset(Dataset):
    """
    Simple dataset for (user, bot) chat pairs.
    For now we use in-memory list of dicts; later you can load from CSV/JSON.
    """

    def __init__(self, tokenizer, pairs: List[Dict[str, str]], max_length: int = 256):
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        user = pair["user"]
        bot = pair["bot"]

        # You can tune the prompt format later to match your paper style
        prompt = f"Pengguna: {user}\nAsisten:"

        # Encode input (prompt)
        model_inputs = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Encode target (bot response)
        labels = self.tokenizer(
            bot,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        label_ids = labels["input_ids"].squeeze(0)

        # IMPORTANT: pad tokens -> -100 so they are ignored by loss
        pad_id = self.tokenizer.pad_token_id
        label_ids[label_ids == pad_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }
