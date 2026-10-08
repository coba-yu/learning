import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from codebot.model import GPT
from codebot.utils import get_device


class TokenDataset(Dataset):
    def __init__(self, tokens: np.ndarray, context_len: int) -> None:
        super().__init__()
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.context_len = context_len

    def __len__(self) -> int:
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx:idx + self.context_len]
        y = self.tokens[idx + 1:idx + self.context_len + 1]
        return x, y


if __name__ == "__main__":
    data_path = "data/tiny_codes.bin"
    model_save_path = "models/model_pretrain.pt"

    context_len = 256
    vocab_size = 1000
    batch_size = 32
    learning_rate = 3e-4
    max_iters = 20000
    embed_dim = 384
    n_head = 6
    n_layer = 6
    ff_dim = 4 * embed_dim
    dropout_rate = 0.1

    device = get_device()
    print(f"Using device: {device}")

    dataset = TokenDataset(
        np.fromfile(data_path, dtype=np.uint16),
        context_len,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GPT(
        vocab_size=vocab_size,
        max_context_len=context_len,
        embed_dim=embed_dim,
        n_head=n_head,
        n_layer=n_layer,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    losses = []
    data_iter = cycle(dataloader)
    pbar = tqdm(range(max_iters))
    for i in pbar:
        batch_x, batch_y = next(data_iter)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), batch_y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("models/loss_pretrain.png")
    plt.close()
