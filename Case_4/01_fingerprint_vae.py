import argparse
import csv
import functools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_records(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of objects")
    out: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


def record_keep_basic(record: Dict[str, Any]) -> bool:
    v = record.get("keep_for_basic_demo")
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, bool):
        return v
    return True


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol


@functools.lru_cache(maxsize=None)
def _get_morgan_generator(radius: int, n_bits: int):
    return GetMorganGenerator(radius=radius, fpSize=n_bits)


def morgan_fp(mol: Chem.Mol, n_bits: int, radius: int) -> np.ndarray:
    gen = _get_morgan_generator(radius=radius, n_bits=n_bits)
    bv = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Invalid split ratios")
    idx = np.arange(n, dtype=int)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, min(n_train, n))
    n_val = max(1, min(n_val, n - n_train - 1)) if n - n_train >= 2 else max(0, min(n_val, n - n_train))
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


class FingerprintVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must have at least one layer")
        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(nn.ReLU())
            if dropout > 0:
                enc_layers.append(nn.Dropout(p=dropout))
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.mu = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

        dec_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU())
            if dropout > 0:
                dec_layers.append(nn.Dropout(p=dropout))
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode_logits(z)
        return recon_logits, mu, logvar


@dataclass
class LossStats:
    total: float
    recon: float
    kld: float


def vae_loss(
    recon_logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = F.binary_cross_entropy_with_logits(recon_logits, x, reduction="sum") / x.size(0)
    kld = (-0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())) / x.size(0)
    total = recon + beta * kld
    return total, recon, kld


def run_epoch(
    model: FingerprintVAE,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    beta: float,
) -> LossStats:
    is_train = optimizer is not None
    model.train(is_train)
    total_sum = 0.0
    recon_sum = 0.0
    kld_sum = 0.0
    n_batches = 0

    for (x,) in loader:
        x = x.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        recon_logits, mu, logvar = model(x)
        loss, recon, kld = vae_loss(recon_logits, x, mu, logvar, beta=beta)
        if is_train:
            loss.backward()
            optimizer.step()
        total_sum += float(loss.detach().cpu())
        recon_sum += float(recon.detach().cpu())
        kld_sum += float(kld.detach().cpu())
        n_batches += 1

    denom = max(1, n_batches)
    return LossStats(total=total_sum / denom, recon=recon_sum / denom, kld=kld_sum / denom)


def tensor_to_bitvect(x01: np.ndarray) -> DataStructs.ExplicitBitVect:
    x01 = (x01 > 0.5).astype(np.int8)
    bits = "".join("1" if b else "0" for b in x01.tolist())
    return DataStructs.CreateFromBitString(bits)


def mean_recon_tanimoto(model: FingerprintVAE, x: np.ndarray, batch_size: int, device: torch.device) -> float:
    if x.shape[0] == 0:
        return float("nan")
    model.eval()
    sims: List[float] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device)
            mu, logvar = model.encode(xb)
            z = mu
            logits = model.decode_logits(z)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            x_true = x[i : i + batch_size]
            for a, b in zip(x_true, probs):
                bv_a = tensor_to_bitvect(a)
                bv_b = tensor_to_bitvect((b > 0.5).astype(np.float32))
                sims.append(DataStructs.TanimotoSimilarity(bv_a, bv_b))
    return float(np.mean(sims)) if len(sims) else float("nan")


def save_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def plot_training_curve(out_png: Path, history: List[Dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    train_total = [h["train_total"] for h in history]
    val_total = [h["val_total"] for h in history]
    train_recon = [h["train_recon"] for h in history]
    val_recon = [h["val_recon"] for h in history]
    train_kld = [h["train_kld"] for h in history]
    val_kld = [h["val_kld"] for h in history]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(epochs, train_total, label="train_total")
    ax.plot(epochs, val_total, label="val_total")
    ax.plot(epochs, train_recon, label="train_recon")
    ax.plot(epochs, val_recon, label="val_recon")
    ax.plot(epochs, train_kld, label="train_kld")
    ax.plot(epochs, val_kld, label="val_kld")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_latent_2d(out_png: Path, z2: np.ndarray, labels: List[str], title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    uniq = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(uniq))))
    for c, lab in zip(colors, uniq):
        mask = np.array([x == lab for x in labels], dtype=bool)
        ax.scatter(z2[mask, 0], z2[mask, 1], s=28, alpha=0.85, label=lab, color=c)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="dataset/electrolyte_solvent_library_s.json")
    p.add_argument("--out-dir", type=str, default="runs/basic_fingerprint_vae")
    p.add_argument("--fp-bits", type=int, default=2048)
    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--hidden-dims", type=str, default="512,256")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-all", action="store_true")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    set_seed(args.seed)

    data_path = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(data_path)
    if not args.use_all:
        records = [r for r in records if record_keep_basic(r)]

    ids: List[str] = []
    names: List[str] = []
    smiles_list: List[str] = []
    families: List[str] = []
    fps: List[np.ndarray] = []

    for r in records:
        sid = str(r.get("id", "")).strip()
        name = str(r.get("name", "")).strip()
        smi = str(r.get("smiles", "")).strip()
        fam = str(r.get("family", "unknown")).strip() or "unknown"
        mol = smiles_to_mol(smi)
        if mol is None:
            continue
        fp = morgan_fp(mol, n_bits=int(args.fp_bits), radius=int(args.fp_radius))
        ids.append(sid)
        names.append(name)
        smiles_list.append(smi)
        families.append(fam)
        fps.append(fp)

    if len(fps) < 10:
        raise RuntimeError("Not enough valid molecules to train a VAE")

    x = np.stack(fps, axis=0).astype(np.float32)
    n = x.shape[0]
    train_idx, val_idx, test_idx = split_indices(n, args.train_ratio, args.val_ratio, args.seed)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dims = [int(s.strip()) for s in str(args.hidden_dims).split(",") if len(s.strip())]
    model = FingerprintVAE(
        input_dim=int(args.fp_bits),
        latent_dim=int(args.latent_dim),
        hidden_dims=hidden_dims,
        dropout=float(args.dropout),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

    def make_loader(sub_idx: np.ndarray, shuffle: bool) -> DataLoader:
        tx = torch.from_numpy(x[sub_idx])
        ds = TensorDataset(tx)
        return DataLoader(ds, batch_size=int(args.batch_size), shuffle=shuffle, drop_last=False)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)
    test_loader = make_loader(test_idx, shuffle=False)

    best_val = float("inf")
    best_path = out_dir / "basic_vae_best.pt"
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        tr = run_epoch(model, train_loader, opt, device=device, beta=float(args.beta))
        va = run_epoch(model, val_loader, None, device=device, beta=float(args.beta))
        scheduler.step(va.total)
        lr_now = float(opt.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_total": tr.total,
                "train_recon": tr.recon,
                "train_kld": tr.kld,
                "val_total": va.total,
                "val_recon": va.recon,
                "val_kld": va.kld,
                "lr": lr_now,
            }
        )
        if va.total < best_val:
            best_val = va.total
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_stats = run_epoch(model, test_loader, None, device=device, beta=float(args.beta))
    tanimoto_val = mean_recon_tanimoto(model, x[val_idx], batch_size=int(args.batch_size), device=device)
    tanimoto_test = mean_recon_tanimoto(model, x[test_idx], batch_size=int(args.batch_size), device=device)

    metrics_path = out_dir / "basic_vae_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_total": int(n),
                "n_train": int(train_idx.size),
                "n_val": int(val_idx.size),
                "n_test": int(test_idx.size),
                "device": str(device),
                "best_val_total": float(best_val),
                "test_total": float(test_stats.total),
                "test_recon": float(test_stats.recon),
                "test_kld": float(test_stats.kld),
                "val_mean_recon_tanimoto": float(tanimoto_val),
                "test_mean_recon_tanimoto": float(tanimoto_test),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with torch.no_grad():
        mu_all: List[np.ndarray] = []
        for i in range(0, n, int(args.batch_size)):
            xb = torch.from_numpy(x[i : i + int(args.batch_size)]).to(device)
            mu, _ = model.encode(xb)
            mu_all.append(mu.detach().cpu().numpy())
        z = np.concatenate(mu_all, axis=0)

    z_path = out_dir / "basic_vae_latent.csv"
    header = ["id", "name", "smiles", "family"] + [f"z{i+1}" for i in range(z.shape[1])]
    rows = []
    for i in range(n):
        rows.append([ids[i], names[i], smiles_list[i], families[i], *[float(v) for v in z[i].tolist()]])
    save_csv(z_path, header=header, rows=rows)

    hist_path = out_dir / "basic_vae_history.csv"
    hist_header = list(history[0].keys()) if history else []
    hist_rows = [[h[k] for k in hist_header] for h in history]
    save_csv(hist_path, header=hist_header, rows=hist_rows)

    plot_training_curve(out_dir / "basic_vae_training.png", history=history)

    if z.shape[1] == 2:
        plot_latent_2d(out_dir / "basic_vae_latent_2d.png", z2=z, labels=families, title="Fingerprint-VAE latent (mu)")


if __name__ == "__main__":
    main()
