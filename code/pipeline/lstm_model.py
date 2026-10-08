"""
lstm_model.py
--------------
The torch part of the breakout-probability model: the network, the training loop with early stopping, and
inference. Kept apart from lstm_features.py so that everything else (features, labels, splits, calibration, the
logistic model) works without torch. Imported lazily by lstm_hurst_train.py and lstm_hurst_predictor.py.

Differences from the old network
  * outputs a LOGIT (the sigmoid is applied later, after calibration) and is trained with BCEWithLogitsLoss;
  * one layer with dropout on the last step and weight decay (the old 2-layer model memorised the ~570 training
    sequences of a company: train AUC 0.998 against 0.877 on the hold-out);
  * early stopping on the validation block's loss, restoring the best epoch (the old script ran a fixed 30 epochs
    and picked the grid winner by its TEST AUC).
"""
import copy

import numpy as np
import torch
import torch.nn as nn


class LSTMWithHurst(nn.Module):
    """LSTM over `seq_length` sessions of scale-free features -> one logit per sequence."""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):                                   # x: (batch, seq_length, n_features)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


def _loader(X, y, batch_size, seed):
    ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)


def predict_logits(model, X, batch_size=2048):
    """Raw logits for sequences X (n, seq_length, n_features) as a numpy vector."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(torch.tensor(X[i:i + batch_size], dtype=torch.float32)).numpy())
    return np.concatenate(out) if out else np.empty(0)


def _loss(model, X, y):
    z = torch.tensor(predict_logits(model, X), dtype=torch.float32)
    return float(nn.BCEWithLogitsLoss()(z, torch.tensor(y, dtype=torch.float32)))


def fit_lstm(Xtr, ytr, Xva, yva, hidden_size, lr, weight_decay=1e-4, max_epochs=120, patience=12,
             batch_size=64, seed=42, dropout=0.3):
    """Train with early stopping on the validation loss. Returns (model at its best epoch, best_epoch, best_val_loss)."""
    torch.manual_seed(seed)
    model = LSTMWithHurst(Xtr.shape[2], hidden_size, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = _loader(Xtr, ytr, batch_size, seed)
    best_loss, best_state, best_epoch, bad = float("inf"), copy.deepcopy(model.state_dict()), 0, 0   # (a NaN loss must not leave this None)
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        val_loss = _loss(model, Xva, yva)
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, best_epoch, bad = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_loss


def fit_lstm_fixed(X, y, hidden_size, lr, epochs, weight_decay=1e-4, batch_size=64, seed=42, dropout=0.3):
    """Train for exactly `epochs` (no validation data): used for the optional refit on all rows."""
    torch.manual_seed(seed)
    model = LSTMWithHurst(X.shape[2], hidden_size, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = _loader(X, y, batch_size, seed)
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


def save_lstm(model, path):
    torch.save(model.state_dict(), path)


def load_lstm(path, input_size, hidden_size, dropout=0.3):
    model = LSTMWithHurst(input_size, hidden_size, dropout=dropout)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
