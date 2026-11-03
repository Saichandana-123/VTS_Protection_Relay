import torch
import torch.nn as nn


STAGE4_INPUTS = [
    "SR_2_OUT",
    "6_AND_OUT",
    "240_TIMER_OUT",
    "2_AND_OUT",
]

STAGE4_OUTPUTS = ["10_AND_OUT"]


class Stage4MLP(nn.Module):

    def __init__(self, hidden_dim=32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, signals):
        # Inversions from the reference gate diagram
        x = signals.clone()
        x[:, 0] = 1.0 - x[:, 0]  # NOT SR_2_OUT
        x[:, 2] = 1.0 - x[:, 2]  # NOT 240_TIMER_OUT
        return self.network(x)

    def predict_binary(self, signals):
        return (torch.sigmoid(self.forward(signals)) >= 0.5).float()


if __name__ == "__main__":
    model = Stage4MLP()
    sample = torch.rand(8, len(STAGE4_INPUTS))
    output = model.predict_binary(sample)

    print("Stage 4 model:", model.__class__.__name__)
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Output order:", STAGE4_OUTPUTS)