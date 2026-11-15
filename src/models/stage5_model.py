
import torch
import torch.nn as nn

from src.models.conditional_mlp_film import FILM_SETTINGS, FiLMConditioner, apply_film


STAGE5_INPUTS = [
    "SR_3_OUT",
    "MCB_VTS_OPTO",
    "8_AND_OUT",
    "VTS_TIMER_OUT",
    "20_TIMER_OUT",
]

STAGE5_OUTPUTS = [
    "5_OR_OUT",
    "6_OR_OUT",
    "7_OR_OUT",
]


class Stage5MLP(nn.Module):

    def __init__(self, hidden_dim=32, settings_dim=len(FILM_SETTINGS)):
        super().__init__()
        self.film5 = FiLMConditioner(settings_dim, hidden_dim)
        self.film6 = FiLMConditioner(settings_dim, hidden_dim)
        self.film7 = FiLMConditioner(settings_dim, hidden_dim)

        self.gate5 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.gate6 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.gate7 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, signals, settings=None):

        sr3 = signals[:, 0:1]
        mcb_vts = signals[:, 1:2]
        and8 = signals[:, 2:3]
        vts_timer = signals[:, 3:4]
        timer10 = signals[:, 4:5]

        # 5-OR
        features5 = self.gate5[0](torch.cat([sr3, mcb_vts], dim=1))
        if settings is not None:
            features5 = apply_film(features5, settings, self.film5)
        logits5 = self.gate5[2](torch.relu(features5))
        or5 = torch.sigmoid(logits5)

        # 6-OR depends on 5_OR_OUT
        features6 = self.gate6[0](torch.cat([or5, and8], dim=1))
        if settings is not None:
            features6 = apply_film(features6, settings, self.film6)
        logits6 = self.gate6[2](torch.relu(features6))

        # 7-OR
        features7 = self.gate7[0](torch.cat([vts_timer, timer10], dim=1))
        if settings is not None:
            features7 = apply_film(features7, settings, self.film7)
        logits7 = self.gate7[2](torch.relu(features7))

        return torch.cat([logits5, logits6, logits7], dim=1)

    def predict_binary(self, signals, settings=None):
        return (torch.sigmoid(self.forward(signals, settings)) >= 0.5).float()


if __name__ == "__main__":
    model = Stage5MLP()
    sample = torch.rand(8, len(STAGE5_INPUTS))
    output = model.predict_binary(sample)

    print("Stage 5 model:", model.__class__.__name__)
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Output order:", STAGE5_OUTPUTS)
