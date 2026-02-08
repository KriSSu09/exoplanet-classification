"""
Phase 2.0 — Manifest-driven plumbing sanity baseline

Train a column-token Transformer from scratch on the merged Phase 1 dataset.
No SSL, no physics losses, no adversary.

Key change vs old baseline:
- Feature selection is driven by phase1_manifest.json:
  features = union over missions of leakage_filter.feature_cols
           + union over missions of canonical.created keys (the canonical column names)

This guarantees we never accidentally train on Phase 1 metadata/provenance fields
(e.g., label_source_value), which would cause leakage and perfect scores.

Run:
    python train_tabular_transformer_baseline.py
"""

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

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
MANIFEST_PATH = os.path.join(DATA_DIR, "phase1_manifest.json")

OUTDIR = os.path.join(PROJECT_ROOT, "runs", "phase2_baseline_tabular_transformer_manifest")

# -----------------------------
# Column names (Phase 1 schema)
# -----------------------------
LABEL_COL = "target_class"
MISSION_COL = "mission"
SUPERVISED_COL = "supervised_eligible"

# 3-class mapping (consistent with Phase 1)
LABEL_MAP = {"positive": 0, "candidate": 1, "negative": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# -----------------------------
# Training hyperparams
# -----------------------------
SEED = 42
EPOCHS = 50
BATCH_SIZE = 256
LR = 3e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_DELTA = 1e-4

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


def load_manifest(manifest_path: str) -> Dict:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Could not find Phase 1 manifest at: {manifest_path}\n"
            "Phase 2.0 requires the manifest to select leakage-safe features."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_feature_list_from_manifest(manifest: Dict) -> Tuple[List[str], Dict]:
    """
    Features are:
      - union of per-mission leakage_filter.feature_cols
      - union of per-mission canonical.created keys (canonical column names)

    We also return a small report about counts for artifacts/debugging.
    """
    sources = manifest.get("sources", {})
    if not sources:
        raise ValueError("Manifest has no 'sources' section; cannot derive features.")

    leakage_union: Set[str] = set()
    canon_union: Set[str] = set()

    per_mission_counts = {}

    for m, info in sources.items():
        leakage_cols = info.get("leakage_filter", {}).get("feature_cols", [])
        canon_created = info.get("canonical", {}).get("created", {})  # dict canon_name -> src_col

        leakage_union.update(leakage_cols)
        canon_union.update(list(canon_created.keys()))

        per_mission_counts[m] = {
            "n_leakage_features": int(len(leakage_cols)),
            "n_canon_created": int(len(canon_created)),
        }

    all_features = sorted(leakage_union.union(canon_union))

    report = {
        "n_features_total": int(len(all_features)),
        "n_features_from_leakage_union": int(len(leakage_union)),
        "n_features_from_canonical_union": int(len(canon_union)),
        "per_mission": per_mission_counts,
    }
    return all_features, report


def infer_feature_types_from_whitelist(df: pd.DataFrame, feature_whitelist: List[str]) -> Tuple[List[str], List[str]]:
    """
    Infer numeric vs categorical only within the given whitelist, and only if columns exist in df.
    """
    existing = [c for c in feature_whitelist if c in df.columns]

    numeric_cols, cat_cols = [], []
    for c in existing:
        dt = df[c].dtype
        if pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)

    # deterministic order for reproducibility
    return sorted(numeric_cols), sorted(cat_cols)


def build_cat_vocabs(df: pd.DataFrame, cat_cols: List[str], max_vocab: int = 50000) -> Dict[str, Dict[str, int]]:
    """
    Build vocab mapping for each categorical column.
    id=0 reserved for MISSING/UNK.
    """
    vocabs: Dict[str, Dict[str, int]] = {}
    for c in cat_cols:
        series = df[c].astype("string")
        uniq = series.dropna().unique().tolist()
        if len(uniq) > max_vocab:
            top = series.value_counts(dropna=True).head(max_vocab - 1).index.tolist()
            uniq = top
        vocabs[c] = {tok: i + 1 for i, tok in enumerate(uniq)}
    return vocabs


def compute_numeric_stats(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
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


def assert_no_leakage_columns_in_features(features: List[str]):
    """
    Guardrails against known Phase 1 metadata/provenance columns that must never be model inputs.
    """
    banned = {
        LABEL_COL,
        MISSION_COL,
        SUPERVISED_COL,
        "row_uid",
        "label_source",
        "label_source_value",
        # common identifiers / provenance / timestamps (should not be used as features)
        "kepid", "kepoi_name", "kepler_name",
        "tid", "toi", "toidisplay", "toipfx", "ctoi_alias",
        "pl_name", "pl_letter", "k2_name", "epic_hostname", "epic_candname", "hostname",
        "hd_name", "hip_name", "tic_id", "gaia_dr2_id", "gaia_dr3_id",
        "rowupdate", "release_date", "toi_created", "pl_pubdate", "releasedate", "disc_pubdate",
        # any old aliases you previously used
        "y_source_col", "y_source_val", "y3",
        "source_table", "source_key",
        "snapshot_time_utc", "dataset_version_id",
    }
    offenders = sorted([c for c in features if c in banned])
    if offenders:
        raise RuntimeError(
            "Leakage/provenance columns found in feature list derived from manifest.\n"
            f"Offending columns: {offenders}\n"
            "This should never happen. Check Phase 1 manifest or feature selection logic."
        )


# -----------------------------
# Dataset
# -----------------------------
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


# -----------------------------
# Model
# -----------------------------
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


# -----------------------------
# Evaluation
# -----------------------------
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


# -----------------------------
# Main
# -----------------------------
def main():
    set_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    if not os.path.exists(MERGED_PATH):
        raise FileNotFoundError(f"Could not find merged parquet at: {MERGED_PATH}")

    manifest = load_manifest(MANIFEST_PATH)

    # Derive feature list from manifest (union leakage-safe + canonicals)
    feature_whitelist, feat_report = build_feature_list_from_manifest(manifest)
    assert_no_leakage_columns_in_features(feature_whitelist)

    print(f"Loading: {MERGED_PATH}")
    df = pd.read_parquet(MERGED_PATH)

    # Supervised slice
    for req in [SUPERVISED_COL, LABEL_COL, MISSION_COL]:
        if req not in df.columns:
            raise KeyError(f"Expected column '{req}' not found in merged parquet.")

    df = df[df[SUPERVISED_COL] == True].copy()
    df = df[df[LABEL_COL].isin(LABEL_MAP.keys())].copy()

    # Ensure whitelist columns exist; drop those not present
    feature_whitelist_existing = [c for c in feature_whitelist if c in df.columns]
    missing_from_parquet = [c for c in feature_whitelist if c not in df.columns]

    numeric_cols, cat_cols = infer_feature_types_from_whitelist(df, feature_whitelist_existing)

    # Final hard guardrail: absolutely ensure no known leakage columns slipped in
    assert_no_leakage_columns_in_features(numeric_cols + cat_cols)

    print(f"Rows (supervised): {len(df)}")
    print(f"Feature whitelist (manifest-derived): {len(feature_whitelist)} "
          f"(present in parquet: {len(feature_whitelist_existing)}, missing: {len(missing_from_parquet)})")
    print(f"Numeric cols: {len(numeric_cols)} | Categorical cols: {len(cat_cols)}")

    # Stratify by mission+label to keep balanced splits
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

    # Preserve deterministic column/cardinality order
    cat_cards = [max(cat_vocabs[c].values(), default=0) + 1 for c in cat_cols]

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

    # Class weights (train only)
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
    patience_left = EARLY_STOP_PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
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

        improved = val_macro_f1 > best_val + EARLY_STOP_MIN_DELTA
        if improved:
            best_val = val_macro_f1
            patience_left = EARLY_STOP_PATIENCE
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping triggered at epoch {epoch:02d}. Best val_macroF1={best_val:.4f}")
                break

    # Save artifacts
    artifacts = {
        "paths": {
            "merged_parquet": MERGED_PATH,
            "manifest": MANIFEST_PATH,
        },
        "manifest_feature_derivation": feat_report,
        "manifest_created_utc": manifest.get("created_utc"),
        "feature_spec": {
            "numeric_cols": numeric_cols,
            "cat_cols": cat_cols,
            "feature_whitelist_total": feature_whitelist,
            "feature_whitelist_present_in_parquet": feature_whitelist_existing,
            "feature_whitelist_missing_from_parquet": missing_from_parquet,
        },
        "cat_vocabs": cat_vocabs,
        "num_stats": num_stats,
        "label_map": LABEL_MAP,
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
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        },
        "splits": {
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "stratification": "mission::label",
        },
    }

    with open(os.path.join(OUTDIR, "artifacts.json"), "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    # Final test eval
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_out = evaluate(model, test_loader, device)

    with open(os.path.join(OUTDIR, "test_metrics.json"), "w", encoding="utf-8") as f:
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
