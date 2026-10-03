import pickle
import re
from collections import defaultdict

from tqdm import tqdm

BASE_VOCAB_SIZE = 256


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged_ids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1

    return merged_ids

def count_pairs(
    ids: list[int],
    counts: defaultdict[tuple[int, int], int] | None = None,
) -> defaultdict[tuple[int, int], int]:
    if counts is None:
        counts = defaultdict(int)

    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def pretokenize(text: str) -> list[str]:
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return re.findall(pattern, text)


def train_bpe(
    input_text: str,
    vocab_size: int,
    end_token: str = "<|endoftext|>",
) -> dict[tuple[int, int], int]:
    texts = input_text.split(end_token)

    ids_list = []
    for text in texts:
        for pretoken in pretokenize(text):
            ids_list.append(list(pretoken.encode("utf-8")))

    num_merges = vocab_size - BASE_VOCAB_SIZE - 1
    merge_rules = {}

    for step in tqdm(range(num_merges), desc="Training BPE"):
        counts = defaultdict(int)
        for ids in ids_list:
            counts = count_pairs(ids, counts)

        if not counts:
            break

        best_pair = max(counts, key=counts.get)
        new_id = BASE_VOCAB_SIZE + step
        merge_rules[best_pair] = new_id

        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], best_pair, new_id)

    return merge_rules


class BPETokenizer:
    def __init__(
        self,
        merge_rules: dict[tuple[int, int], int],
        end_token: str = "<|endoftext|>",
    ) -> None:
        self.merge_rules = merge_rules
        self.end_token = end_token
        self.end_token_id = BASE_VOCAB_SIZE + len(merge_rules)

        self.id_to_bytes = {i: bytes([i]) for i in range(BASE_VOCAB_SIZE)}
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]
            self.id_to_bytes[self.end_token_id] = self.end_token.encode("utf-8")

        self.vocab_size = len(self.id_to_bytes)

    def _encode_text(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)
        return ids

    def encode(
        self,
        input_text: str,
        show_progress: bool = False,
    ) -> list[int]:
        pattern = "(" + re.escape(self.end_token) + ")"
        texts = re.split(pattern, input_text)
        all_ids = []

        if show_progress:
            texts = tqdm(texts, desc="Encoding")

        for text in texts:
            if text == self.end_token:
                all_ids.append(self.end_token_id)
            else:
                for pretoken in pretokenize(text):
                    ids = self._encode_text(pretoken)
                    all_ids.extend(ids)

        return all_ids

    def decode(self, ids: list[int]) -> str:
        byte_list = [self.id_to_bytes[id] for id in ids]
        text_bytes = b"".join(byte_list)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    @staticmethod
    def load_from(filepath: str) -> "BPETokenizer":
        with open(filepath, "rb") as f:
            merge_rules = pickle.load(f)
        return BPETokenizer(merge_rules)
