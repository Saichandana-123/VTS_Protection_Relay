
"""
Stage 2: Second combinational ML layer of P642 VTS.

Inputs:
    V2, I2                  Direct analog inputs
    240_TIMER_OUT
    1_AND_OUT
    2_OR_OUT

Outputs:
    3_AND_OUT
    4_AND_OUT
    4_OR_OUT
    5_AND_OUT

The 4_OR_OUT prediction feeds the 5_AND_OUT subnetwork.
"""

import torch
import torch.nn as nn


STAGE2_INPUTS = [
    "V2",
    "I2",
    "240_TIMER_OUT",
    "1_AND_OUT",
    "2_OR_OUT",
]

STAGE2_OUTPUTS = [
    "3_AND_OUT",
    "4_AND_OUT",
    "4_OR_OUT",
    "5_AND_OUT",
]


class Stage2MLP(nn.Module):

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Predict 3_AND_OUT, 4_AND_OUT and 4_OR_OUT.
        self.first_network = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Predict 5_AND_OUT using the predicted 4_OR_OUT.
        self.final_network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, signals):

        v2 = signals[:, 0:1]
        i2 = signals[:, 1:2]
        timer_out = signals[:, 2:3]
        and1_out = signals[:, 3:4]
        or2_out = signals[:, 4:5]

        # First three outputs.
        first_logits = self.first_network(signals)

        first_outputs = torch.sigmoid(first_logits)

        and3_out = first_outputs[:, 0:1]
        and4_out = first_outputs[:, 1:2]
        or4_out = first_outputs[:, 2:3]

        # 5_AND inputs:
        # 4_OR_OUT, NOT(240_TIMER_OUT), 2_OR_OUT
        final_inputs = torch.cat(
            [or4_out, 1.0 - timer_out, or2_out],
            dim=1
        )

        and5_logits = self.final_network(final_inputs)

        # Return logits for all four outputs.
        return torch.cat(
            [first_logits, and5_logits],
            dim=1
        )

    def predict_binary(self, signals):
        return (torch.sigmoid(self.forward(signals)) >= 0.5).float()


if __name__ == "__main__":

    model = Stage2MLP()

    sample = torch.rand(8, len(STAGE2_INPUTS))

    output = model.predict_binary(sample)

    print("Stage 2 model:", model.__class__.__name__)
    print("Input shape:", sample.shape)
    print("Output shape:", output.shape)
    print("Output order:", STAGE2_OUTPUTS)
