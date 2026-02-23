"""Intent + VRU style prediction: LSTM per agent, intent and style heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class IntentStylePredictor(nn.Module):
    """
    Per-agent LSTM: history -> intent + style probs.
    Intent: yield/stop, proceed, turn/merge (veh) or cross, wait/slow (VRU).
    Style: soft, mid, aggressive.
    """

    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 1,
        intent_classes: int = 3,
        style_classes: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.intent_classes = intent_classes
        self.style_classes = style_classes

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.intent_head = nn.Linear(hidden_dim, intent_classes)
        self.style_head = nn.Linear(hidden_dim, style_classes)

    def forward(
        self,
        z: torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """
        z: (B, T, 9) history [delta_xy, delta_v, delta_psi, d_cz, d_cpa, nu, sigma]
        Returns: intent_probs (B,T,3), style_probs (B,T,3), (intent_ent, style_ent), new_hidden
        """
        out, new_hidden = self.lstm(z, hidden)
        intent_logits = self.intent_head(out)
        style_logits = self.style_head(out)
        intent_probs = torch.softmax(intent_logits, dim=-1)
        style_probs = torch.softmax(style_logits, dim=-1)
        H_intent = -(intent_probs * (intent_probs + 1e-8).log()).sum(-1)
        H_style = -(style_probs * (style_probs + 1e-8).log()).sum(-1)
        return intent_probs, style_probs, (H_intent, H_style), new_hidden
