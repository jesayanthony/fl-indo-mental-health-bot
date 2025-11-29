# data/preprocess.py

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Sample:
    text: str
    label: str | None = None
    client_id: int | None = None  # for federated partition

def load_raw_datasets() -> List[Sample]:
    """
    TODO: load your Psychika and AMoD data here.
    For now, return an empty list or some dummy samples.
    """
    samples: List[Sample] = []
    # Example dummy data:
    # samples.append(Sample(text="Saya merasa cemas akhir-akhir ini", label="anxiety"))
    return samples

def basic_preprocess(sample: Sample) -> Sample:
    """
    Apply your preprocessing: lowercasing, tokenization, etc.
    For now, just lowercasing.
    """
    sample.text = sample.text.lower()
    return sample

def make_client_partitions(
    samples: List[Sample], num_clients: int = 3
) -> Dict[int, List[Sample]]:
    """
    Split samples into num_clients partitions.
    Later we can do non-IID splitting based on label/domain.
    """
    partitions: Dict[int, List[Sample]] = {i: [] for i in range(num_clients)}
    for idx, s in enumerate(samples):
        client_id = idx % num_clients
        s.client_id = client_id
        partitions[client_id].append(s)
    return partitions
