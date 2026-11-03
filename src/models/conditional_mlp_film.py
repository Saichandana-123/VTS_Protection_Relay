
"""
Conditional MLP + FiLM architecture for P642 VTS replication.

Measurements and logic signals enter the main MLP.
Relay settings condition its hidden representations through FiLM.
The network predicts three VTS outputs.

Note:
- This is the neural-network architecture, not the complete relay.
- Stateful logic and deterministic timers remain separate components.
- Training and validation are implemented separately.
"""

import torch
import torch.nn as nn


class FiLMConditioner(nn.Module):
    """Generate feature-wise scale and shift from relay settings."""

    def __init__(self, settings_dim: int, hidden_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(settings_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, settings):
        parameters = self.network(settings)
        gamma, beta = parameters.chunk(2, dim=-1)

        # Identity-like initialization of the modulation.
        gamma = 1.0 + gamma

        return gamma, beta


class ConditionalMLPFiLM(nn.Module):
    """
    Conditional MLP with FiLM-based setting conditioning.

    Inputs:
        signals: measured quantities and binary/logic inputs,
                 shape [batch_size, signal_dim]
        settings: configurable relay settings,
                  shape [batch_size, settings_dim]

    Outputs:
        Three logits, corresponding to:
        SLOW_BLOCK, FAST_BLOCK, INDICATION
    """

    def __init__(
        self,
        signal_dim: int,
        settings_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 3,
    ):
        super().__init__()

        self.input_layer = nn.Linear(signal_dim, hidden_dim)
        self.film = FiLMConditioner(settings_dim, hidden_dim)

        self.hidden_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, signals, settings):
        features = self.input_layer(signals)

        gamma, beta = self.film(settings)

        # Feature-wise Linear Modulation:
        # settings change the hidden features without retraining.
        conditioned_features = gamma * features + beta

        hidden = self.hidden_layer(conditioned_features)
        logits = self.output_layer(hidden)

        return logits


if __name__ == "__main__":
    # Basic architecture smoke test.
    # These dimensions are placeholders until the final feature
    # schema is defined from the validated dataset.

    batch_size = 8
    signal_dim = 16
    settings_dim = 4

    model = ConditionalMLPFiLM(
        signal_dim=signal_dim,
        settings_dim=settings_dim,
    )

    signals = torch.randn(batch_size, signal_dim)
    settings = torch.randn(batch_size, settings_dim)

    logits = model(signals, settings)

    print("Model:", model.__class__.__name__)
    print("Input signals:", tuple(signals.shape))
    print("Input settings:", tuple(settings.shape))
    print("Output logits:", tuple(logits.shape))
    print("Parameter count:", sum(p.numel() for p in model.parameters()))
