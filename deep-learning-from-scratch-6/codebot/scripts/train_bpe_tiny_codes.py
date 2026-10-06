import pickle

from codebot.tokenizer import train_bpe


def main() -> None:
    vocab_size = 1000
    with open("data/tiny_codes.txt", "r") as f:
        text = f.read()

    merge_rules = train_bpe(text, vocab_size)

    with open("data/tiny_codes.pkl", "wb") as f:
        pickle.dump(merge_rules, f)


if __name__ == "__main__":
    main()
