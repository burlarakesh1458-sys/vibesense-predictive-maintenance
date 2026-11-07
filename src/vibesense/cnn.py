"""Small 1D-CNN on the raw window (WDCNN-style wide first kernel). Torch is only
needed to train/export; serving and the browser demo use the ONNX export."""

from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """Per-window z-score; identical in Python, ONNX and the JS demo."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def build_model(n_classes: int):
    import torch.nn as nn

    def block(cin, cout, k, s=1):
        return [nn.Conv1d(cin, cout, k, s, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU(), nn.MaxPool1d(2)]

    return nn.Sequential(
        *block(1, 16, 64, 8),  # wide kernel = learned band-pass filters, 2048 -> 256 -> 128
        *block(16, 32, 5),  # 64
        *block(32, 64, 3),  # 32
        *block(64, 64, 3),  # 16
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(64, n_classes),
    )


def train_cnn(X, y, n_classes, epochs=15, seed=0, batch=64, lr=2e-3, log=None):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    model = build_model(n_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(X) + batch - 1) // batch))
    Xt = torch.from_numpy(normalize(X)[:, None, :])
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    lossf = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(X))
        tot = 0.0
        for i in range(0, len(X), batch):
            b = perm[i : i + batch]
            xb = Xt[b]
            # augmentation: random circular shift + light gaussian noise
            xb = torch.roll(xb, int(rng.integers(0, xb.shape[-1])), dims=-1)
            xb = xb + 0.05 * torch.randn_like(xb)
            opt.zero_grad()
            loss = lossf(model(xb), yt[b])
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
        if log:
            log(f"  epoch {ep + 1}/{epochs} loss {tot / len(X):.4f}")
    model.eval()
    return model


def predict_proba_torch(model, X, batch=512) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(normalize(X[i : i + batch])[:, None, :])
            out.append(torch.softmax(model(xb), 1).numpy())
    return np.concatenate(out)


def export_onnx(model, path, window: int) -> None:
    import torch

    dummy = torch.zeros(1, 1, window)
    torch.onnx.export(
        model, dummy, str(path), input_names=["signal"], output_names=["logits"],
        dynamic_axes={"signal": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False,
    )
