import json
from itertools import cycle

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import get_device


class SFTDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: BPETokenizer,
        context_len: int,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.samples = []

        with open(data_path, "r") as f:
            data = json.load(f)
        
        for item in data:
            ids, labels = self._create_sample(item["instruction"], item["response"])
            self.samples.append((ids, labels))

    def _create_sample(
        self,
        instruction: str,
        response: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        response = f"{response}<|endoftext|>"

        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)
        
        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        # 入力と正解を1つずらす
        ids = ids[:-1]
        labels = labels[1:]

        pad_len = self.context_len - len(ids)
        if pad_len > 0:
            ids = ids + [0] * pad_len
            labels = labels + [-100] * pad_len
        elif pad_len < 0:
            ids = ids[:self.context_len]
            labels = labels[:self.context_len]

        return ids, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, labels = self.samples[index]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


if __name__ == "__main__":
    data_path = "data/tiny_codes_sft.json"
    tokenizer_path = "data/tiny_codes.pkl"
    pretrain_model_path = "models/model_pretrain.pt"
    sft_model_save_path = "models/model_sft.pt"

    context_len = 256
    batch_size = 32
    learning_rate = 3e-4
    max_iters = 500

    device = get_device()

    tokenizer = BPETokenizer.load_from(tokenizer_path)
    dataset = SFTDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        context_len=context_len,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = GPT.load_from(pretrain_model_path, device=device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    losses = []
    data_iter = cycle(dataloader)
    pbar = tqdm(range(max_iters))
    for iter in pbar:
        batch_x, batch_y = next(data_iter)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch_y.view(-1),
            ignore_index=-100,  # ignore mask
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    model.save(sft_model_save_path)

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("models/loss_sft.png")
    plt.close()
