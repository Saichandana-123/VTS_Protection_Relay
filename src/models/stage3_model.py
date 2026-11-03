
import torch
import torch.nn as nn


STAGE3_INPUTS = [
    "ALL_POLE_DEAD",
    "1_OR_OUT",
    "4_OR_OUT",
    "4_AND_OUT",
    "SR_1_OUT",
    "3_OR_OUT",
    "ANY_VOLTAGE_DEPENDENT_FUNCTION",
]

STAGE3_OUTPUTS = [
    "6_AND_OUT",
    "7_AND_OUT",
    "8_AND_OUT",
    "9_AND_OUT",
]


class Stage3MLP(nn.Module):

    def __init__(self, hidden_dim=32):
        super().__init__()

        # 6-AND: 5 inputs
        self.gate6 = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 7-AND: 2 inputs
        self.gate7 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 8-AND: 2 inputs
        self.gate8 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 9-AND: 2 inputs
        self.gate9 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, signals):

        all_pole_dead = signals[:, 0:1]
        or1 = signals[:, 1:2]
        or4 = signals[:, 2:3]
        and4 = signals[:, 3:4]
        sr1 = signals[:, 4:5]
        or3 = signals[:, 5:6]
        voltage_function = signals[:, 6:7]

        # 6-AND
        gate6_inputs = torch.cat([
            1 - all_pole_dead,
            1 - or1,
            or4,
            1 - and4,
            1 - sr1
        ], dim=1)

        logits6 = self.gate6(gate6_inputs)
        and6 = torch.sigmoid(logits6)

        # 7-AND
        gate7_inputs = torch.cat([1 - or4, or3], dim=1)
        logits7 = self.gate7(gate7_inputs)

        # 8-AND
        gate8_inputs = torch.cat([
            1 - voltage_function,
            and6
        ], dim=1)

        logits8 = self.gate8(gate8_inputs)

        # 9-AND
        gate9_inputs = torch.cat([
            voltage_function,
            1 - and6
        ], dim=1)

        logits9 = self.gate9(gate9_inputs)

        return torch.cat([
            logits6, logits7, logits8, logits9
        ], dim=1)

    def predict_binary(self, signals):
        return (torch.sigmoid(self.forward(signals)) >= 0.5).float()


if __name__ == "__main__":

    model = Stage3MLP()
    sample = torch.rand(8, len(STAGE3_INPUTS))
    output = model.predict_binary(sample)

    print("Stage 3 model:", model.__class__.__name__)
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Output order:", STAGE3_OUTPUTS)
