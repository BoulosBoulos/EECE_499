"""Intent + VRU style prediction: LSTM per agent, intent and style heads.

Phase 0F (v9): bidirectional 3-layer LSTM @ hidden=384 with 1D CNN frontend.
The heads operate on 2*hidden_dim outputs (forward + backward concatenated).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class IntentStylePredictor(nn.Module):
    """
    Per-agent LSTM: history -> intent + style probs.
    Intent: yield/stop, proceed, turn/merge (veh) or cross, wait/slow (VRU).
    Style: cautious, normal, chaotic (sigma-aligned, post Phase 0C).

    Architecture (v9):
      conv_frontend (Conv1d(input_dim,32) -> GELU -> Conv1d(32,32) -> GELU)
      -> bidirectional LSTM (hidden=384, layers=3, dropout=0.2)
      -> heads on 2 * hidden_dim
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 384,
        num_layers: int = 3,
        intent_classes: int = 3,
        style_classes: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
        conv_channels: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.intent_classes = intent_classes
        self.style_classes = style_classes
        self.bidirectional = bool(bidirectional)
        self.conv_channels = conv_channels

        self.conv_frontend = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers >= 2 else 0.0,
            bidirectional=self.bidirectional,
        )

        out_dim = hidden_dim * (2 if self.bidirectional else 1)
        self.intent_head = nn.Linear(out_dim, intent_classes)
        self.style_head = nn.Linear(out_dim, style_classes)
        self.recon_head = nn.Linear(out_dim, input_dim)

    def _encode(self, z: torch.Tensor, hidden=None):
        """Run conv frontend + LSTM, return per-step hidden states."""
        x = z.transpose(1, 2)             # (B, input_dim, T)
        x = self.conv_frontend(x)         # (B, C, T)
        x = x.transpose(1, 2)             # (B, T, C)
        out, new_hidden = self.lstm(x, hidden)
        return out, new_hidden

    def forward(
        self,
        z: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """
        z: (B, T, input_dim) history.
        Returns: intent_probs (B,T,3), style_probs (B,T,3), (H_intent, H_style), new_hidden
        """
        out, new_hidden = self._encode(z, hidden)
        intent_logits = self.intent_head(out)
        style_logits = self.style_head(out)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        style_probs = torch.softmax(style_logits, dim=-1)
        H_intent = -(intent_probs * (intent_probs + 1e-8).log()).sum(-1)
        H_style = -(style_probs * (style_probs + 1e-8).log()).sum(-1)
        return intent_probs, style_probs, (H_intent, H_style), new_hidden
