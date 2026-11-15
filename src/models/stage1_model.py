
"""
Stage 1: First combinational ML layer of P642 VTS.

Inputs:
    IA_GT, IB_GT, IC_GT
    DELTA_IA_GT, DELTA_IB_GT, DELTA_IC_GT
    VTS_MANRESET, VTS_AUTORESET
    VAB_GT, VBC_GT
    VTS_BLOCKING, ACCELERATE_IND

Outputs:
    1_OR_OUT, 2_OR_OUT, 3_OR_OUT,
    1_AND_OUT, 2_AND_OUT
"""

import torch
import torch.nn as nn

from src.models.conditional_mlp_film import FILM_SETTINGS, FiLMConditioner, apply_film


STAGE1_INPUTS = [
    "IA_GT", "IB_GT", "IC_GT",
    "DELTA_IA_GT", "DELTA_IB_GT", "DELTA_IC_GT",
    "VTS_MANRESET", "VTS_AUTORESET",
    "VAB_GT", "VBC_GT",
    "VTS_BLOCKING", "ACCELERATE_IND",
]

STAGE1_OUTPUTS = [
    "1_OR_OUT",
    "2_OR_OUT",
    "3_OR_OUT",
    "1_AND_OUT",
    "2_AND_OUT",
]


class Stage1MLP(nn.Module):
    """MLP replacing the five combinational gates in Stage 1."""

    def __init__(self, hidden_dim=32, settings_dim=len(FILM_SETTINGS)):
        super().__init__()

        self.film = FiLMConditioner(settings_dim, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(len(STAGE1_INPUTS), hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(STAGE1_OUTPUTS)),
        )

    def forward(self, signals, settings=None):
        features = self.network[0](signals)
        if settings is not None:
            features = apply_film(features, settings, self.film)
        features = torch.relu(features)
        features = self.network[1](features)
        features = torch.relu(self.network[2](features))
        logits = self.network[3](features)

        # The reference 2_AND gate receives NOT VTS_BLOCKING. Keep the
        # external Stage 1 input raw and apply this inversion only to 2_AND.
        gate2_signals = signals.clone()
        gate2_signals[:, 10] = 1.0 - gate2_signals[:, 10]
        gate2_features = self.network[0](gate2_signals)
        if settings is not None:
            gate2_features = apply_film(gate2_features, settings, self.film)
        gate2_features = torch.relu(gate2_features)
        gate2_features = self.network[1](gate2_features)
        gate2_features = torch.relu(self.network[2](gate2_features))
        gate2_logits = self.network[3](gate2_features)[:, 4:5]

        return torch.cat([logits[:, :4], gate2_logits], dim=1)

    def predict_binary(self, signals, settings=None):
        logits = self.forward(signals, settings)
        return (torch.sigmoid(logits) >= 0.5).float()


if __name__ == "__main__":
    model = Stage1MLP()

    sample = torch.randint(0, 2, (8, len(STAGE1_INPUTS))).float()
    output = model.predict_binary(sample)

    print("Stage 1 model:", model.__class__.__name__)
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Output order:", STAGE1_OUTPUTS)
