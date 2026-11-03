
import torch
import torch.nn as nn


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

    def __init__(self, hidden_dim=32):
        super().__init__()

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

    def forward(self, signals):

        sr3 = signals[:, 0:1]
        mcb_vts = signals[:, 1:2]
        and8 = signals[:, 2:3]
        vts_timer = signals[:, 3:4]
        timer10 = signals[:, 4:5]

        # 5-OR
        logits5 = self.gate5(torch.cat([sr3, mcb_vts], dim=1))
        or5 = torch.sigmoid(logits5)

        # 6-OR depends on 5_OR_OUT
        logits6 = self.gate6(torch.cat([or5, and8], dim=1))

        # 7-OR
        logits7 = self.gate7(
            torch.cat([vts_timer, timer10], dim=1)
        )

        return torch.cat([logits5, logits6, logits7], dim=1)

    def predict_binary(self, signals):
        return (torch.sigmoid(self.forward(signals)) >= 0.5).float()


if __name__ == "__main__":
    model = Stage5MLP()
    sample = torch.rand(8, len(STAGE5_INPUTS))
    output = model.predict_binary(sample)

    print("Stage 5 model:", model.__class__.__name__)
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Output order:", STAGE5_OUTPUTS)
