import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        head_dim: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_head = n_head
        self.head_dim = head_dim
        E, H, D = embed_dim, n_head, head_dim

        self.W_q = nn.Linear(E, H * D, bias=False)
        self.W_k = nn.Linear(E, H * D, bias=False)
        self.W_v = nn.Linear(E, H * D, bias=False)
        self.W_o = nn.Linear(H * D, E, bias=False)

        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B: batch size
        # C: context length
        # E: embedding dimension
        B, C, E = x.shape

        H, D = self.n_head, self.head_dim

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # (B, C, E) -> (B, C, H, D) -> (B, H, C, D)
        Q = Q.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        K = K.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)
        V = V.view(B, C, H, D).transpose(1, 2)  # (B, H, C, D)

        scores = Q @ K.transpose(-2, -1)  # (B, H, C, C)
        scores /= D ** 0.5

        # Mask
        mask = torch.tril(torch.ones(C, C, device=scores.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # Attention weights
        weights = F.softmax(scores, dim=-1)  # (B, H, C, C)
        weights = self.attention_dropout(weights)
        hidden = weights @ V  # (B, H, C, D)

        # Attention output
        hidden = hidden.transpose(1, 2).contiguous()  # (B, C, H, D)
        hidden = hidden.view(B, C, H * D)  # (B, C, H * D)
        output = self.W_o(hidden)  # (B, C, E)
        output = self.output_dropout(output)
        return output


class FFN(nn.Module):
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int | None = None,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * x_dim

        self.layers = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, x_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        ff_dim: int | None = None,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        head_dim = embed_dim // n_head
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(
            x_dim=embed_dim,
            hidden_dim=ff_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_context_len: int,
        embed_dim: int,
        n_head: int,
        n_layer: int,
        ff_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.max_context_len = max_context_len
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_context_len, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Transformer block
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    n_head=n_head,
                    ff_dim=ff_dim,
                    dropout_rate=dropout_rate,
                )
                for _ in range(n_layer)
            ]
        )

        # Output layer
        self.norm = nn.LayerNorm(embed_dim)
        self.unembed = nn.Linear(embed_dim, vocab_size)

        self.embed.weight = self.unembed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # https://arxiv.org/abs/1810.04805

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # B: batch size
        # C: context length
        B, C = ids.shape
        device = ids.device

        # Embedding
        pos = torch.arange(0, C, dtype=torch.long, device=device)
        emb = self.embed(ids)
        pos_emb = self.pos_embed(pos)
        x = self.dropout(emb + pos_emb)

        # Transformer block
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Output
        logits = self.unembed(x)  # (B, V, vocab_size)
        return logits

    def save(self, file_path: str) -> None:
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "max_context_len": self.max_context_len,
            "embed_dim": self.embed_dim,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        }
        torch.save(checkpoint, file_path)

    @classmethod
    def load_from(cls, file_path: str, device: str = "cpu") -> "GPT":
        checkpoint = torch.load(file_path, map_location=device)
        
        model = cls(
            vocab_size=checkpoint["vocab_size"],
            max_context_len=checkpoint["max_context_len"],
            embed_dim=checkpoint["embed_dim"],
            n_head=checkpoint["n_head"],
            n_layer=checkpoint["n_layer"],
            ff_dim=checkpoint["ff_dim"],
            dropout_rate=checkpoint["dropout_rate"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        return model
