import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


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


def morgan_fp(mol: Chem.Mol, n_bits: int, radius: int) -> np.ndarray:
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    bv = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)


def mol_for_draw(smiles: str) -> Optional[Chem.Mol]:
    m = smiles_to_mol(smiles)
    if m is None:
        return None
    try:
        rdDepictor.Compute2DCoords(m)
    except Exception:
        return m
    return m


def save_mol_grid(
    out_png: Path,
    mols: List[Chem.Mol],
    legends: List[str],
    mols_per_row: int,
    legend_font_size: int,
    subimg_w: int,
    subimg_h: int,
) -> None:
    if len(mols) == 0:
        return
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=max(1, int(mols_per_row)),
        subImgSize=(int(subimg_w), int(subimg_h)),
        legends=legends,
        legendFontSize=int(legend_font_size),
        useSVG=False,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_png))



class FingerprintVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
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

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.mu(h)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


def save_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def plot_3panel(out_png: Path, pca2: np.ndarray, tsne2: np.ndarray, vae2: np.ndarray, labels: List[str]) -> None:
    import matplotlib.pyplot as plt

    uniq = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(uniq))))

    def _scatter(ax, xy, title):
        for c, lab in zip(colors, uniq):
            mask = np.array([x == lab for x in labels], dtype=bool)
            ax.scatter(xy[mask, 0], xy[mask, 1], s=18, alpha=0.85, label=lab, color=c)
        ax.set_title(title)
        ax.set_xlabel("dim-1")
        ax.set_ylabel("dim-2")
        ax.grid(True, linestyle="--", alpha=0.2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _scatter(axes[0], pca2, "PCA on fingerprint")
    _scatter(axes[1], tsne2, "t-SNE on fingerprint")
    _scatter(axes[2], vae2, "VAE latent (mu)")
    handles, labs = axes[2].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=min(6, len(uniq)), fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def tanimoto_topk(fp: np.ndarray, fps: np.ndarray, k: int) -> List[Tuple[int, float]]:
    bv = DataStructs.CreateFromBitString("".join("1" if b > 0.5 else "0" for b in fp.tolist()))
    sims: List[Tuple[int, float]] = []
    for i in range(fps.shape[0]):
        bv_i = DataStructs.CreateFromBitString("".join("1" if b > 0.5 else "0" for b in fps[i].tolist()))
        sims.append((i, float(DataStructs.TanimotoSimilarity(bv, bv_i))))
    sims.sort(key=lambda t: t[1], reverse=True)
    return sims[:k]


def find_by_query(
    records: List[Dict[str, Any]],
    ids: List[str],
    names: List[str],
    smiles: List[str],
    query: str,
) -> int:
    q = query.strip().lower()
    for i, s in enumerate(smiles):
        if s.strip().lower() == q:
            return i
    for i, n in enumerate(names):
        if q == n.strip().lower():
            return i
    for i, n in enumerate(names):
        if q in n.strip().lower():
            return i
    for i, rid in enumerate(ids):
        if q == rid.strip().lower():
            return i
    raise ValueError(f"Query not found in dataset: {query}")


def parse_pair(s: str) -> Tuple[str, str]:
    if ":" in s:
        a, b = s.split(":", 1)
        return a.strip(), b.strip()
    if "->" in s:
        a, b = s.split("->", 1)
        return a.strip(), b.strip()
    raise ValueError("pair must be formatted like A:B or A->B")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="dataset/electrolyte_solvent_library_s.json")
    p.add_argument("--ckpt", type=str, default="runs/basic_fingerprint_vae/basic_vae_best.pt")
    p.add_argument("--out-dir", type=str, default="runs/basic_fingerprint_vae/analysis")
    p.add_argument("--use-all", action="store_true")
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    p.add_argument("--tsne-seed", type=int, default=42)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--queries", type=str, default="EC,DME")
    p.add_argument("--pairs", type=str, default="DMC:DEC,EC:PC,THF:DME")
    p.add_argument("--n-steps", type=int, default=11)
    p.add_argument("--mols-per-row", type=int, default=6)
    p.add_argument("--legend-font-size", type=int, default=28)
    p.add_argument("--subimg-w", type=int, default=260)
    p.add_argument("--subimg-h", type=int, default=200)
    args = p.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})
    fp_bits = int(train_args.get("fp_bits", 2048))
    fp_radius = int(train_args.get("fp_radius", 2))
    latent_dim = int(train_args.get("latent_dim", 2))
    hidden_dims = [int(x.strip()) for x in str(train_args.get("hidden_dims", "512,256")).split(",") if len(x.strip())]
    dropout = float(train_args.get("dropout", 0.1))
    seed = int(train_args.get("seed", 42))

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
        fp = morgan_fp(mol, n_bits=fp_bits, radius=fp_radius)
        ids.append(sid)
        names.append(name)
        smiles_list.append(smi)
        families.append(fam)
        fps.append(fp)

    x = np.stack(fps, axis=0).astype(np.float32)
    n = x.shape[0]
    if n < 5:
        raise RuntimeError("Not enough molecules for analysis")

    model = FingerprintVAE(input_dim=fp_bits, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        mu = model.encode_mu(torch.from_numpy(x)).numpy()

    pca2 = PCA(n_components=2, random_state=seed).fit_transform(x)
    perplexity = float(args.tsne_perplexity)
    if n - 1 <= 0:
        perplexity = 2.0
    else:
        perplexity = min(perplexity, max(2.0, (n - 1) / 3.0))
    tsne2 = TSNE(n_components=2, perplexity=perplexity, random_state=int(args.tsne_seed), init="pca", learning_rate="auto").fit_transform(x)

    if mu.shape[1] == 2:
        vae2 = mu
    else:
        vae2 = PCA(n_components=2, random_state=seed).fit_transform(mu)

    plot_3panel(out_dir / "compare_pca_tsne_vae.png", pca2=pca2, tsne2=tsne2, vae2=vae2, labels=families)

    alias = {
        "EC": "O=C1OCCO1",
        "PC": "CC1COC(=O)O1",
        "DME": "COCCOC",
        "THF": "C1CCOC1",
        "DMC": "COC(=O)OC",
        "DEC": "CCOC(=O)OCC",
    }

    query_tokens = [t.strip() for t in str(args.queries).split(",") if len(t.strip())]
    nn_rows: List[List[Any]] = []
    nn_vis: Dict[str, List[Tuple[int, float]]] = {}
    for q in query_tokens:
        qv = alias.get(q, q)
        qi = find_by_query(records, ids, names, smiles_list, qv)
        zq = mu[qi]
        d = np.linalg.norm(mu - zq.reshape(1, -1), axis=1)
        nn = np.argsort(d)
        nn_vis[q] = [(int(j), float(d[j])) for j in nn[: int(args.topk)]]
        for rank, j in enumerate(nn[: int(args.topk)], start=1):
            nn_rows.append([q, ids[qi], names[qi], smiles_list[qi], rank, ids[j], names[j], smiles_list[j], families[j], float(d[j])])

    save_csv(
        out_dir / "nearest_neighbors_latent.csv",
        header=["query", "query_id", "query_name", "query_smiles", "rank", "nn_id", "nn_name", "nn_smiles", "nn_family", "l2_dist"],
        rows=nn_rows,
    )

    for q, idxs in nn_vis.items():
        qv = alias.get(q, q)
        qi = find_by_query(records, ids, names, smiles_list, qv)
        mols: List[Chem.Mol] = []
        legends: List[str] = []
        m0 = mol_for_draw(smiles_list[qi])
        if m0 is not None:
            mols.append(m0)
            legends.append(f"{q}\n{names[qi]}\n{families[qi]}")
        for rank, (j, dist) in enumerate(idxs, start=1):
            m = mol_for_draw(smiles_list[j])
            if m is None:
                continue
            mols.append(m)
            legends.append(f"rank {rank}\n{names[j]}\n{families[j]}\nd={dist:.3f}")
        tag = "".join([c if c.isalnum() else "_" for c in q]) or "query"
        save_mol_grid(
            out_dir / f"nearest_neighbors_{tag}.png",
            mols=mols,
            legends=legends,
            mols_per_row=int(args.mols_per_row),
            legend_font_size=int(args.legend_font_size),
            subimg_w=int(args.subimg_w),
            subimg_h=int(args.subimg_h),
        )

    pair_tokens = [t.strip() for t in str(args.pairs).split(",") if len(t.strip())]
    for ps in pair_tokens:
        a_raw, b_raw = parse_pair(ps)
        a = alias.get(a_raw, a_raw)
        b = alias.get(b_raw, b_raw)
        ia = find_by_query(records, ids, names, smiles_list, a)
        ib = find_by_query(records, ids, names, smiles_list, b)
        za = mu[ia]
        zb = mu[ib]

        steps = int(args.n_steps)
        ts = np.linspace(0.0, 1.0, steps)
        interp_rows: List[List[Any]] = []
        interp_best_idx: List[int] = []
        interp_best_sim: List[float] = []
        interp_sim_a: List[float] = []
        interp_sim_b: List[float] = []

        for t in ts:
            zt = (1.0 - t) * za + t * zb
            with torch.no_grad():
                logits = model.decode_logits(torch.from_numpy(zt.astype(np.float32)).unsqueeze(0)).squeeze(0).numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            top = tanimoto_topk(probs, x, k=max(1, int(args.topk)))
            best_i, best_sim = top[0]

            bva = DataStructs.CreateFromBitString("".join("1" if v > 0.5 else "0" for v in x[ia].tolist()))
            bvb = DataStructs.CreateFromBitString("".join("1" if v > 0.5 else "0" for v in x[ib].tolist()))
            bvd = DataStructs.CreateFromBitString("".join("1" if v > 0.5 else "0" for v in probs.tolist()))
            sim_a = float(DataStructs.TanimotoSimilarity(bva, bvd))
            sim_b = float(DataStructs.TanimotoSimilarity(bvb, bvd))
            interp_best_idx.append(int(best_i))
            interp_best_sim.append(float(best_sim))
            interp_sim_a.append(sim_a)
            interp_sim_b.append(sim_b)

            interp_rows.append(
                [
                    float(t),
                    ids[ia],
                    names[ia],
                    ids[ib],
                    names[ib],
                    sim_a,
                    sim_b,
                    ids[best_i],
                    names[best_i],
                    families[best_i],
                    float(best_sim),
                ]
            )

        tag_a = a_raw.replace(" ", "_")
        tag_b = b_raw.replace(" ", "_")
        save_csv(
            out_dir / f"interp_{tag_a}_{tag_b}.csv",
            header=[
                "t",
                "end_a_id",
                "end_a_name",
                "end_b_id",
                "end_b_name",
                "tanimoto_to_a",
                "tanimoto_to_b",
                "nn1_id",
                "nn1_name",
                "nn1_family",
                "nn1_tanimoto",
            ],
            rows=interp_rows,
        )

        import matplotlib.pyplot as plt

        t_vals = [r[0] for r in interp_rows]
        sim_a_vals = [r[5] for r in interp_rows]
        sim_b_vals = [r[6] for r in interp_rows]
        sim_nn_vals = [r[10] for r in interp_rows]

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(t_vals, sim_a_vals, label="decoded vs A")
        ax.plot(t_vals, sim_b_vals, label="decoded vs B")
        ax.plot(t_vals, sim_nn_vals, label="decoded vs NN1")
        ax.set_xlabel("t")
        ax.set_ylabel("Tanimoto")
        ax.set_title(f"Latent interpolation: {a_raw} -> {b_raw}")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"interp_{tag_a}_{tag_b}.png", dpi=200)
        plt.close(fig)

        mols: List[Chem.Mol] = []
        legends: List[str] = []
        ma = mol_for_draw(smiles_list[ia])
        mb = mol_for_draw(smiles_list[ib])
        if ma is not None:
            mols.append(ma)
            legends.append(f"A\n{a_raw}\n{names[ia]}")
        for step_i, (t, j, sim_nn, sa, sb) in enumerate(zip(ts.tolist(), interp_best_idx, interp_best_sim, interp_sim_a, interp_sim_b), start=1):
            m = mol_for_draw(smiles_list[j])
            if m is None:
                continue
            mols.append(m)
            legends.append(f"t={t:.2f}\nNN1: {names[j]}\nT(nn)={sim_nn:.2f}\nT(A)={sa:.2f} T(B)={sb:.2f}")
        if mb is not None:
            mols.append(mb)
            legends.append(f"B\n{b_raw}\n{names[ib]}")
        save_mol_grid(
            out_dir / f"interp_{tag_a}_{tag_b}_mols.png",
            mols=mols,
            legends=legends,
            mols_per_row=int(args.mols_per_row),
            legend_font_size=int(args.legend_font_size),
            subimg_w=int(args.subimg_w),
            subimg_h=int(args.subimg_h),
        )


if __name__ == "__main__":
    main()
