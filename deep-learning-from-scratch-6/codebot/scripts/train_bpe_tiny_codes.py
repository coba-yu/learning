import os
import pickle

from codebot.tokenizer import train_bpe, BASE_VOCAB_SIZE, BPETokenizer


def main() -> None:
    vocab_size = 1000

    text_path = "data/tiny_codes.txt"
    merge_rules_path = "data/tiny_codes.pkl"
    if os.path.exists(merge_rules_path):
        with open(merge_rules_path, "rb") as f:
            merge_rules = pickle.load(f)
        print(f"Loaded merge rules from {merge_rules_path}")
    else:
        print(f"Training merge rules for {merge_rules_path}")
        with open(text_path, "r") as f:
            text = f.read()

        merge_rules = train_bpe(text, vocab_size)
        with open(merge_rules_path, "wb") as f:
            pickle.dump(merge_rules, f)
        print(f"Saved merge rules to {merge_rules_path}")

    tokenizer = BPETokenizer(merge_rules)

    print("最初に学習された10個:")
    for token_id in range(BASE_VOCAB_SIZE, BASE_VOCAB_SIZE + 10):
        byte_seq = tokenizer.id_to_bytes[token_id]
        text = byte_seq.decode("utf-8")
        print(f"  ID {token_id}: '{text}'")

    print("最後に学習された10個:")
    for token_id in range(vocab_size - 10, vocab_size):
        byte_seq = tokenizer.id_to_bytes[token_id]
        text = byte_seq.decode("utf-8")
        print(f"  ID {token_id}: '{text}'")

    with open(text_path, "r") as f:
        sample_text = f.read()[:10000]
    
    byte_count = len(sample_text.encode("utf-8"))
    ids = tokenizer.encode(sample_text)
    ids_count = len(ids)
    compression_ratio = byte_count / ids_count

    print(f"\n=== 圧縮効率 ===")
    print(f"バイト数: {byte_count:,}")
    print(f"トークン数: {ids_count:,}")
    print(f"圧縮比: {compression_ratio:.2f}倍（平均 {compression_ratio:.2f}トークン/バイト）")

if __name__ == "__main__":
    main()
