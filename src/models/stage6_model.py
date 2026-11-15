
import torch
import torch.nn as nn

from src.models.conditional_mlp_film import FILM_SETTINGS, FiLMConditioner, apply_film


STAGE6_INPUTS = [
    "5_OR_OUT",
    "6_OR_OUT",
    "VTS_BLOCKING",
    "MCB_VTS_OPTO",
    "SR_4_OUT",
]

STAGE6_OUTPUTS = [
    "SLOW_BLOCK",
    "FAST_BLOCK",
    "INDICATION",
]


class Stage6MLP(nn.Module):

    def __init__(self, hidden_dim=32, settings_dim=len(FILM_SETTINGS)):
        super().__init__()
        self.film11 = FiLMConditioner(settings_dim, hidden_dim)
        self.film12 = FiLMConditioner(settings_dim, hidden_dim)
        self.film8 = FiLMConditioner(settings_dim, hidden_dim)

        self.gate11 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.gate12 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.gate8 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, signals, settings=None):

        or5 = signals[:, 0:1]
        or6 = signals[:, 1:2]
        vts_blocking = signals[:, 2:3]
        mcb_vts_opto = signals[:, 3:4]
        sr4 = signals[:, 4:5]

        features11 = self.gate11[0](torch.cat([or5, vts_blocking], dim=1))
        if settings is not None:
            features11 = apply_film(features11, settings, self.film11)
        slow_logits = self.gate11[2](torch.relu(features11))

        features12 = self.gate12[0](torch.cat([or6, vts_blocking], dim=1))
        if settings is not None:
            features12 = apply_film(features12, settings, self.film12)
        fast_logits = self.gate12[2](torch.relu(features12))

        features8 = self.gate8[0](torch.cat([mcb_vts_opto, sr4], dim=1))
        if settings is not None:
            features8 = apply_film(features8, settings, self.film8)
        indication_logits = self.gate8[2](torch.relu(features8))

        return torch.cat(
            [slow_logits, fast_logits, indication_logits],
            dim=1
        )

    def predict_binary(self, signals, settings=None):
        return (torch.sigmoid(self.forward(signals, settings)) >= 0.5).float()


if __name__ == "__main__":

    model = Stage6MLP()
    sample = torch.rand(8, len(STAGE6_INPUTS))
    output = model.predict_binary(sample)

    print("Stage 6 model:", model.__class__.__name__)
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Output order:", STAGE6_OUTPUTS)
