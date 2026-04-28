import torch
import torch.nn as nn

class LSTMModelPrice(nn.Module):
    def __init__(self, input_dim, HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, num_layers=3, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, HORIZON)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)
