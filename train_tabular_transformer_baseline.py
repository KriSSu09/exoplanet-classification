"""
Phase 2.0 — Plumbing sanity baseline
Train a column-token Transformer from scratch on the merged Phase 1 dataset.
No SSL, no physics losses, no adversary.

Run:
    python train_tabular_transformer_baseline.py
"""

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm



# -----------------------------
# Project paths (relative to repo root)
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "exoplanet_dataset_phase1")
MERGED_PATH = os.path.join(DATA_DIR, "phase1_merged.parquet")

OUTDIR = os.path.join(PROJECT_ROOT, "runs", "phase2_baseline_tabular_transformer")

# Optional: also save a copy of the manifest used in Phase 1 (if present)
MANIFEST_PATH = os.path.join(DATA_DIR, "phase1_manifest.json")


# -----------------------------
# Column names (change here if your parquet uses different names)
# -----------------------------
LABEL_COL = "target_class"
MISSION_COL = "mission"
SUPERVISED_COL = "supervised_eligible"

# 3-class mapping (consistent with Phase 1 decisions)
LABEL_MAP = {"positive": 0, "candidate": 1, "negative": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# Columns to always ignore if present
IGNORE_COLS_DEFAULT = {
    LABEL_COL, MISSION_COL, SUPERVISED_COL,
    "row_uid", "row_id", "source_table", "source_key",
    "snapshot_time_utc", "dataset_version_id",
    "y_source_col", "y_source_val", "y3",
}

# -----------------------------
# Training hyperparams (tweak freely)
# -----------------------------
SEED = 0
EPOCHS = 20
BATCH_SIZE = 256
LR = 3e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0

# Model size (baseline)
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 4
DROPOUT = 0.1

# Split fractions
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # fraction of train remaining


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class FeatureSpec:
    numeric_cols: List[str]
    cat_cols: List[str]
    cat_vocabs: Dict[str, Dict[str, int]]  # col -> {token -> id}


def infer_feature_types(df: pd.DataFrame, ignore_cols: set) -> Tuple[List[str], List[str]]:
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    numeric_cols, cat_cols = [], []
    for c in feature_cols:
        dt = df[c].dtype
        if pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)
    return numeric_cols, cat_cols


def build_cat_vocabs(df: pd.DataFrame, cat_cols: List[str], max_vocab: int = 50000) -> Dict[str, Dict[str, int]]:
    """
    Build vocab mapping for each categorical column.
    id=0 reserved for MISSING/UNK.
    """
    vocabs = {}
    for c in cat_cols:
        series = df[c].astype("string")
        uniq = series.dropna().unique().tolist()
        if len(uniq) > max_vocab:
            top = series.value_counts(dropna=True).head(max_vocab - 1).index.tolist()
            uniq = top
        vocabs[c] = {tok: i + 1 for i, tok in enumerate(uniq)}
    return vocabs


def compute_numeric_stats(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for c in numeric_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().any():
            mu = float(x.mean(skipna=True))
            sd = float(x.std(skipna=True))
        else:
            mu, sd = 0.0, 1.0
        if not math.isfinite(sd) or sd < 1e-12:
            sd = 1.0
        stats[c] = {"mean": mu, "std": sd}
    return stats


class TabularTokenDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feat: FeatureSpec, num_stats: Dict[str, Dict[str, float]]):
        self.df = df.reset_index(drop=True)
        self.feat = feat
        self.num_stats = num_stats

        self.y = self.df[LABEL_COL].map(LABEL_MAP).astype(int).values
        self.mission = self.df[MISSION_COL].astype(str).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # numeric values and missing mask (1 = missing)
        num_vals, num_mask = [], []
        for c in self.feat.numeric_cols:
            v = row[c]
            missing = pd.isna(v)
            num_mask.append(1.0 if missing else 0.0)
            if missing:
                v = self.num_stats[c]["mean"]
            else:
                v = float(v)
            v = (v - self.num_stats[c]["mean"]) / self.num_stats[c]["std"]
            num_vals.append(v)

        # categorical ids and missing mask
        cat_ids, cat_mask = [], []
        for c in self.feat.cat_cols:
            v = row[c]
            missing = pd.isna(v)
            cat_mask.append(1.0 if missing else 0.0)
            if missing:
                cat_ids.append(0)
            else:
                cat_ids.append(self.feat.cat_vocabs[c].get(str(v), 0))

        return {
            "num_vals": torch.tensor(num_vals, dtype=torch.float32),
            "num_mask": torch.tensor(num_mask, dtype=torch.float32),
            "cat_ids": torch.tensor(cat_ids, dtype=torch.long),
            "cat_mask": torch.tensor(cat_mask, dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "mission": self.mission[idx],
        }


class ColumnTokenTransformer(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: List[int],
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_classes: int = 3,
        n_missions: int = 3,
    ):
        super().__init__()
        self.d_model = d_model

        # Tokens: [CLS] + [MISSION] + numeric feature tokens + categorical feature tokens
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mission_emb = nn.Embedding(n_missions, d_model)

        # Numeric: per-column projection from scalar -> d_model, plus column embedding
        self.num_col_emb = nn.Embedding(n_num, d_model)
        self.num_w = nn.Parameter(torch.randn(n_num, d_model) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(n_num, d_model))
        self.num_missing_emb = nn.Parameter(torch.zeros(d_model))

        # Categorical: per-column embedding tables + column embedding
        self.cat_col_emb = nn.Embedding(len(cat_cardinalities), d_model)
        self.cat_tables = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cardinalities])
        self.cat_missing_emb = nn.Parameter(torch.zeros(d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(d_model, n_classes)

    @staticmethod
    def mission_to_id(mission: List[str]) -> torch.Tensor:
        mapping = {"kepler": 0, "tess": 1, "k2": 2}
        ids = [mapping.get(str(m).lower(), 0) for m in mission]
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, num_vals, num_mask, cat_ids, cat_mask, mission: List[str]):
        B = num_vals.size(0)
        device = num_vals.device

        cls_tok = self.cls.expand(B, -1, -1)
        mid = self.mission_to_id(mission).to(device)
        mission_tok = self.mission_emb(mid).unsqueeze(1)

        # Numeric tokens
        if num_vals.size(1) > 0:
            idx = torch.arange(num_vals.size(1), device=device)
            col = self.num_col_emb(idx).unsqueeze(0).expand(B, -1, -1)
            val = num_vals.unsqueeze(-1)
            proj = val * self.num_w.unsqueeze(0) + self.num_b.unsqueeze(0)
            miss = num_mask.unsqueeze(-1) * self.num_missing_emb
            num_tok = col + proj + miss
        else:
            num_tok = torch.zeros(B, 0, self.d_model, device=device)

        # Categorical tokens
        if cat_ids.size(1) > 0:
            cat_toks = []
            for j, table in enumerate(self.cat_tables):
                ids = cat_ids[:, j].clamp(min=0, max=table.num_embeddings - 1)
                emb = table(ids)  # (B,d)
                col = self.cat_col_emb(torch.tensor(j, device=device)).unsqueeze(0).expand(B, -1)
                miss = cat_mask[:, j].unsqueeze(-1) * self.cat_missing_emb
                cat_toks.append((emb + col + miss).unsqueeze(1))
            cat_tok = torch.cat(cat_toks, dim=1)
        else:
            cat_tok = torch.zeros(B, 0, self.d_model, device=device)

        x = torch.cat([cls_tok, mission_tok, num_tok, cat_tok], dim=1)
        x = self.dropout(x)
        x = self.encoder(x)

        h = self.norm(x[:, 0, :])
        logits = self.head(h)
        return logits


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    model.eval()
    y_true, y_pred, missions = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            logits = model(
                batch["num_vals"].to(device),
                batch["num_mask"].to(device),
                batch["cat_ids"].to(device),
                batch["cat_mask"].to(device),
                batch["mission"],
            )
            pred = logits.argmax(dim=-1).cpu().numpy().tolist()
            y_true.extend(batch["y"].numpy().tolist())
            y_pred.extend(pred)
            missions.extend(batch["mission"])

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    per_m = {}
    for m in sorted(set(missions)):
        idx = [i for i, mm in enumerate(missions) if mm == m]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        per_m[m] = classification_report(yt, yp, output_dict=True, zero_division=0)

    return {"report": report, "confusion_matrix": cm.tolist(), "per_mission": per_m}


def main():
    set_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    if not os.path.exists(MERGED_PATH):
        raise FileNotFoundError(f"Could not find merged parquet at: {MERGED_PATH}")

    print(f"Loading: {MERGED_PATH}")
    df = pd.read_parquet(MERGED_PATH)

    # Supervised slice
    if SUPERVISED_COL not in df.columns:
        raise KeyError(f"Expected column '{SUPERVISED_COL}' not found in merged parquet.")
    df = df[df[SUPERVISED_COL] == True].copy()

    if LABEL_COL not in df.columns:
        raise KeyError(f"Expected column '{LABEL_COL}' not found in merged parquet.")
    df = df[df[LABEL_COL].isin(LABEL_MAP.keys())].copy()

    if MISSION_COL not in df.columns:
        raise KeyError(f"Expected column '{MISSION_COL}' not found in merged parquet.")

    ignore_cols = set(IGNORE_COLS_DEFAULT)
    numeric_cols, cat_cols = infer_feature_types(df, ignore_cols)

    print(f"Rows (supervised): {len(df)}")
    print(f"Numeric cols: {len(numeric_cols)} | Categorical cols: {len(cat_cols)}")

    # Stratify by mission+label so the split isn't accidentally mission-skewed
    strat = df[MISSION_COL].astype(str) + "::" + df[LABEL_COL].astype(str)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=strat)

    strat2 = train_df[MISSION_COL].astype(str) + "::" + train_df[LABEL_COL].astype(str)
    train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=SEED, stratify=strat2)

    # Preprocessing fitted on train only
    cat_vocabs = build_cat_vocabs(train_df, cat_cols)
    num_stats = compute_numeric_stats(train_df, numeric_cols)
    feat = FeatureSpec(numeric_cols=numeric_cols, cat_cols=cat_cols, cat_vocabs=cat_vocabs)

    train_ds = TabularTokenDataset(train_df, feat, num_stats)
    val_ds = TabularTokenDataset(val_df, feat, num_stats)
    test_ds = TabularTokenDataset(test_df, feat, num_stats)

    cat_cards = [max(v.values(), default=0) + 1 for v in cat_vocabs.values()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = ColumnTokenTransformer(
        n_num=len(numeric_cols),
        cat_cardinalities=cat_cards,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        n_classes=3,
        n_missions=3,
    ).to(device)

    # Class weights
    y_train = train_df[LABEL_COL].map(LABEL_MAP).values
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0))
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Class counts:", counts.tolist())
    print("Class weights:", weights.tolist())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = -1.0
    best_path = os.path.join(OUTDIR, "best_model.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{EPOCHS}",
            leave=False,
        )

        for batch in pbar:
            opt.zero_grad(set_to_none=True)
            logits = model(
                batch["num_vals"].to(device),
                batch["num_mask"].to(device),
                batch["cat_ids"].to(device),
                batch["cat_mask"].to(device),
                batch["mission"],
            )
            loss = F.cross_entropy(logits, batch["y"].to(device), weight=weights_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            loss_val = float(loss.detach().cpu())
            losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        val_out = evaluate(model, val_loader, device)
        val_macro_f1 = val_out["report"]["macro avg"]["f1-score"]
        print(f"Epoch {epoch:02d} | train_loss={np.mean(losses):.4f} | val_macroF1={val_macro_f1:.4f}")

        if val_macro_f1 > best_val:
            best_val = val_macro_f1
            torch.save(model.state_dict(), best_path)

    # Save artifacts
    artifacts = {
        "paths": {
            "merged_parquet": MERGED_PATH,
            "manifest": MANIFEST_PATH if os.path.exists(MANIFEST_PATH) else None,
        },
        "feature_spec": {
            "numeric_cols": numeric_cols,
            "cat_cols": cat_cols,
        },
        "cat_vocabs": cat_vocabs,
        "num_stats": num_stats,
        "label_map": LABEL_MAP,
        "ignore_cols": sorted(list(ignore_cols)),
        "hyperparams": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT,
        },
    }
    with open(os.path.join(OUTDIR, "artifacts.json"), "w") as f:
        json.dump(artifacts, f, indent=2)

    # Final test eval
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_out = evaluate(model, test_loader, device)

    with open(os.path.join(OUTDIR, "test_metrics.json"), "w") as f:
        json.dump(test_out, f, indent=2)

    print("\n=== TEST RESULTS ===")
    print("macro-F1:", test_out["report"]["macro avg"]["f1-score"])
    print("Confusion matrix (0=pos,1=cand,2=neg):")
    print(np.array(test_out["confusion_matrix"]))

    for m, rep in test_out["per_mission"].items():
        print(f"{m:>8} macro-F1 = {rep['macro avg']['f1-score']:.4f}")

    print(f"\nSaved run to: {OUTDIR}")


if __name__ == "__main__":
    main()
