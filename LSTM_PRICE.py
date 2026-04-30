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


def create_sequences(X, hours, seq_len=48, HORIZON=24):
    X_seq  = []

    for i in range(len(X) - seq_len - HORIZON):
        if hours[i + seq_len - 1] != 12:
            continue
        X_seq.append(X[i:i+seq_len])

    return np.array(X_seq)