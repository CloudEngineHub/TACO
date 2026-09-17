"""
Tactile joint-denoising modules for Wan video models.

These are parallel to the DiT's video-token stream: a noisy tactile sequence
`(B, T, tactile_dim)` is encoded into DiT-compatible tokens, attended
jointly with video tokens via standard self-attention, and finally projected
back to `(B, T, tactile_dim)` as the velocity-field prediction.

Both modules are tiny (O(few M) parameters) and are trained from scratch.
"""

import torch
import torch.nn as nn

from .wan_video_dit import sinusoidal_embedding_1d


class WanTactileTokenizer(nn.Module):
    def __init__(
        self,
        tactile_dim: int = 12,
        model_dim: int = 5120,
        freq_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        self.tactile_dim = tactile_dim
        self.model_dim = model_dim
        self.freq_dim = freq_dim

        self.feat_proj = nn.Sequential(
            nn.Linear(tactile_dim, hidden_dim),
            nn.SiLU(),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # zero-init output projection: tactile tokens contribute zero at step 0
        self.out_proj = nn.Linear(hidden_dim, model_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        t_emb = sinusoidal_embedding_1d(self.freq_dim, timestep.float())
        t_emb = self.time_proj(t_emb.to(x.dtype))
        h = self.feat_proj(x)
        pos = torch.arange(T, device=x.device, dtype=torch.float32)
        pos_emb = sinusoidal_embedding_1d(h.shape[-1], pos).to(h.dtype)
        h = h + pos_emb.unsqueeze(0) + t_emb.unsqueeze(1)
        h = self.transformer(h)
        return self.out_proj(h)


class WanTactileHead(nn.Module):
    def __init__(
        self,
        model_dim: int = 5120,
        hidden_dim: int = 512,
        tactile_dim: int = 12,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, tactile_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
