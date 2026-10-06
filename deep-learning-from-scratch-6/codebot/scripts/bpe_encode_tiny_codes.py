import pickle

import numpy as np

from codebot.tokenizer import BPETokenizer


if __name__ == "__main__":
    text_path = "data/tiny_codes.txt"
    merge_rules_path = "data/tiny_codes.pkl"
    output_path = "data/tiny_codes.bin"

    with open(merge_rules_path, "rb") as f:
        merge_rules = pickle.load(f)

    tokenizer = BPETokenizer(merge_rules)

    with open(text_path, "r") as f:
        text = f.read()

    ids = tokenizer.encode(text, show_progress=True)

    ids_array = np.array(ids, dtype=np.uint16)
    ids_array.tofile(output_path)

    print(f"トークンID数: {len(ids_array):,}")
    print(f"最初の20個のトークンID: {ids_array[:20]}")
