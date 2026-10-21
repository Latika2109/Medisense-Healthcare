"""Train the PneumoniaMNIST CNN (CPU-friendly); save model, metrics, figures, demo samples."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src import imaging
from src.config import DATA_DIR, FIG_DIR, MODELS_DIR, SEED, ensure_dirs
from src.metrics import binary_metrics, save_confusion_matrix, write_json


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a PneumoniaMNIST split (downloads to data/ on first run)."""
    from medmnist import PneumoniaMNIST

    ds = PneumoniaMNIST(split=split, download=True, root=str(DATA_DIR))
    return ds.imgs.astype(np.uint8), ds.labels.ravel().astype(int)


def main() -> None:
    """Train, evaluate on the official test split, save artifacts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    ensure_dirs()
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.set_num_threads(2)

    (Xtr, ytr), (Xva, yva), (Xte, yte) = load_split("train"), load_split("val"), load_split("test")
    print(f"train={len(ytr)} (pneumonia {ytr.mean():.1%}) val={len(yva)} test={len(yte)} (pneumonia {yte.mean():.1%})")
    model = imaging.SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    pos_weight = torch.tensor((1 - ytr.mean()) / ytr.mean())  # class imbalance
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xt, yt = imaging.to_tensor(Xtr), torch.from_numpy(ytr).float()
    best_auc, best_state, hist = -1.0, None, []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(yt), generator=torch.Generator().manual_seed(SEED + ep))
        tot = 0.0
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        val = binary_metrics(yva, imaging.predict_proba(model, Xva))
        hist.append({"epoch": ep + 1, "train_loss": tot / len(yt), **{f"val_{k}": v for k, v in val.items()}})
        print(f"epoch {ep+1}: loss={tot/len(yt):.4f} val_auc={val['roc_auc']:.3f} val_recall={val['recall']:.3f}")
        if val["roc_auc"] > best_auc:  # checkpoint chosen on validation split only
            best_auc, best_state = val["roc_auc"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    prob = imaging.predict_proba(model, Xte)
    test = binary_metrics(yte, prob)
    print("Test:", {k: round(v, 3) for k, v in test.items()})
    cm = save_confusion_matrix(yte, prob, ["Normal", "Pneumonia"], FIG_DIR / "cnn_confusion.png", "PneumoniaMNIST CNN")
    print("Confusion matrix [[TN, FP],[FN, TP]]:", cm)

    # Grad-CAM gallery + demo samples for the app (8 normal, 8 pneumonia from the test split)
    rng = np.random.default_rng(SEED)
    sel = np.concatenate([rng.choice(np.where(yte == c)[0], 8, replace=False) for c in (0, 1)])
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for ax, i in zip(axes.ravel(), sel[[*range(8, 16), *range(0, 8)]]):
        cam = imaging.grad_cam(model, Xte[i])
        ax.imshow(Xte[i], cmap="gray"); ax.imshow(cam, cmap="jet", alpha=0.4); ax.axis("off")
        ax.set_title(f"true={'P' if yte[i] else 'N'} p={prob[i]:.2f}", fontsize=8)
    fig.suptitle("Grad-CAM (top: pneumonia, bottom: normal)"); fig.tight_layout()
    fig.savefig(FIG_DIR / "cnn_gradcam.png", dpi=120); plt.close(fig)

    torch.save(model.state_dict(), MODELS_DIR / "cnn.pt")
    np.savez_compressed(MODELS_DIR / "cnn_samples.npz", images=Xte[sel], labels=yte[sel])
    write_json({"test": test, "confusion_matrix": cm, "best_val_auc": best_auc, "history": hist, "epochs": args.epochs,
                "n_train": len(ytr), "n_test": len(yte), "seed": SEED}, MODELS_DIR / "cnn_metrics.json")


if __name__ == "__main__":
    main()
