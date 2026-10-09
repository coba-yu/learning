import re
from itertools import cycle

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from codebot.model import GPT
from codebot.tokenizer import BPETokenizer
from codebot.utils import generate, get_device


class GRPODataset(Dataset):
    def __init__(self, tokenizer: BPETokenizer) -> None:
        self.tokenizer = tokenizer
        self.data = []
        for i in range(1, 10):
            for j in range(1, 10):
                prompt = "\n".join(
                    (
                        "### Instruction:",
                        f"{i}+{j}=",
                        "",
                        "### Response:",
                        "",
                    )
                )
                ground_truth = i + j
                self.data.append((prompt, ground_truth))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.data[index]

    def get_batch(
        self,
        prompts: list[str],
        responses: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_ids = []
        all_mask = []

        for prompt, response in zip(prompts, responses):
            prompt_ids = self.tokenizer.encode(prompt)
            response_ids = self.tokenizer.encode(response)

            ids = prompt_ids + response_ids
            mask = [0] * len(prompt_ids) + [1] * len(response_ids)

            all_ids.append(ids)
            all_mask.append(mask)

        max_len = max(len(ids) for ids in all_ids)
        padded_ids = []
        padded_mask = []
        for ids, mask in zip(all_ids, all_mask):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)
            padded_mask.append(mask + [0] * pad_len)
        
        ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        mask = torch.tensor(padded_mask, dtype=torch.float, device=device)
        return ids, mask


def calculate_reward(ground_truth: int, response: str) -> float:
    try:
        matches = re.findall(r"(-?\d+)", response)
        if matches:
            predicted = int(matches[-1])
            if predicted == ground_truth:
                return 1.0
        return 0.0
    except:
        return 0.0


def generate_group(
    model: GPT,
    tokenizer: BPETokenizer,
    prompts: list[str],
    gts: list[int],
    group_size: int,
) -> tuple[list[str], list[int], torch.Tensor]:
    all_prompts = []
    all_responses = []
    all_advantages = []

    for prompt, gt in zip(prompts, gts):
        responses = []
        for _ in range(group_size):
            full_text = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                temperature=1.0,
            )
            response = full_text[len(prompt):]
            responses.append(response)

        rewards = torch.tensor([calculate_reward(gt, r) for r in responses])
        advantages = rewards - rewards.mean()

        for response, advantage in zip(responses, advantages):
            all_prompts.append(prompt)
            all_responses.append(response)
            all_advantages.append(advantage)
    
    return all_prompts, all_responses, torch.stack(all_advantages)


def compute_probs(model: GPT, ids: torch.Tensor) -> torch.Tensor:
    logits = model(ids)  # (B, C, V)
    probs = F.softmax(logits[:, :-1, :], dim=-1)  # (B, C - 1, V)
    labels = ids[:, 1:]  # (B, C - 1)

    token_probs = torch.gather(
        probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)  # (B, C - 1)
    return token_probs


def grpo_loss(
    model: GPT,
    old_model: GPT,
    ids: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    probs = compute_probs(model, ids)

    with torch.no_grad():
        old_probs = compute_probs(old_model, ids)
    
    ratio = probs / (old_probs + 1e-8)
    advantages = advantages.unsqueeze(-1)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    mask = mask[:, 1:]
    token_objective = torch.min(unclipped, clipped) * mask

    n_samples = ids.size(0)
    return -token_objective.sum() / n_samples


if __name__ == "__main__":
    tokenizer_path = "data/tiny_codes.pkl"
    sft_model_save_path = "models/model_sft.pt"
    grpo_model_save_path = "models/model_grpo.pt"

    learning_rate = 7e-6
    max_iters = 500
    n_update_per_generation = 2  # 同じ生成データに対しての更新回数
    eval_interval = 10
    epsilon = 0.2  # クリッピング範囲
    group_size = 8
    batch_size = 32

    device = get_device()

    tokenizer = BPETokenizer.load_from(tokenizer_path)
    model = GPT.load_from(sft_model_save_path, device=device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    old_model = GPT.load_from(sft_model_save_path, device=device)
    old_model.eval()

    dataset = GRPODataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_iter = cycle(dataloader)

    accuracies = []
    current_accuracy = 0.0
    pbar = tqdm(range(max_iters))

    for i in pbar:
        prompts, gts = next(data_iter)
        old_model.load_state_dict(model.state_dict())

        all_prompts, all_responses, all_advantages = generate_group(
            model=old_model,
            tokenizer=tokenizer,
            prompts=prompts,
            gts=gts,
            group_size=group_size,
        )

        ids, mask = dataset.get_batch(
            prompts=all_prompts,
            responses=all_responses,
            device=device,
        )
        all_advantages = all_advantages.to(device)

        for _ in range(n_update_per_generation):
            optimizer.zero_grad()
            loss = grpo_loss(
                model=model,
                old_model=old_model,
                ids=ids,
                mask=mask,
                advantages=all_advantages,
                epsilon=epsilon,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if i % eval_interval == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for prompt, gt in dataset.data:
                    response = generate(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        temperature=0.0,
                    )
                    reward = calculate_reward(gt, response)
                    correct += reward > 0
                    total += 1
        
            model.train()
            current_accuracy = correct / total * 100.0
            accuracies.append(current_accuracy)
        
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{current_accuracy:.1f}%",
            }
        )

    model.save(grpo_model_save_path)

    plt.figure(figsize=(10, 6))
    plt.plot(accuracies)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.title("GRPO Training")
    plt.savefig("models/grpo_accuracy.png")
    plt.close()
