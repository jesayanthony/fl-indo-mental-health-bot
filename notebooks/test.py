import os
import sys

# 1. Compute the project root: one level above "notebooks"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Add ROOT_DIR to sys.path so Python can find "data"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

print("ROOT_DIR:", ROOT_DIR)
print("sys.path[0]:", sys.path[0])

# 3. Now try to import
from data.preprocess import load_raw_datasets, basic_preprocess, make_client_partitions

def main():
    samples = load_raw_datasets()
    print("Num samples:", len(samples))
    parts = make_client_partitions(samples, num_clients=3)
    for cid, v in parts.items():
        print(f"Client {cid} has {len(v)} samples")

if __name__ == "__main__":
    main()
