"""
Phase 2.1 — Self-supervised pretraining (Masked Feature Modeling, MFM)

Pretrain a column-token Transformer encoder on ALL Phase 1 rows (including unlabeled)
by masking numeric features and predicting masked values (Gaussian NLL).
Mission token is always present.

Key principles:
- Manifest-driven, leakage-safe feature selection (union of leakage_filter.feature_cols + canonical.created keys)
- Freeze preprocessing decisions in artifacts so Phase 2.2/2.3 can reuse IDENTICAL preprocessing
- Robust numeric preprocessing: abs+log1p for *err* columns + z-score + optional z-clip
- Stable Gaussian NLL: clamp logvar + variance floor
- Efficient training: micro-batch + grad accumulation + adaptive LR + warmup+cosine

Run:
    python train_tabular_transformer_mfm_pretrain.py
"""

import os
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -----------------------------
# Project paths (relative to repo root)
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "exoplanet_dataset_phase1")
MERGED_PATH = os.path.join(DATA_DIR, "phase1_merged.parquet")
MANIFEST_PATH = os.path.join(DATA_DIR, "phase1_manifest.json")

OUTDIR = os.path.join(PROJECT_ROOT, "runs", "phase2_1_mfm_pretrain_tabular_transformer")

# -----------------------------
# Column names (Phase 1 schema)
# -----------------------------
LABEL_COL = "target_class"
MISSION_COL = "mission"
SUPERVISED_COL = "supervised_eligible"

# -----------------------------
# Training hyperparams
# -----------------------------
SEED = 42

EPOCHS = 100
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0

MICRO_BATCH_SIZE = 640
GRAD_ACCUM_STEPS = 2

WARMUP_FRAC = 0.05
MIN_LR_FRAC = 0.1

EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_DELTA = 1e-4

# Model size
D_MODEL = 192
N_HEADS = 6
N_LAYERS = 4
DROPOUT = 0.1

# Split fractions
VAL_SIZE = 0.1  # SSL dev split; stratify by mission only

# Masking
P_MASK_TOTAL = 0.30  # fraction of *observed* numeric features to mask per row (average)
P_FAMILY_MASK = 0.30
P_PHYSCORE_MASK = 0.10  # remainder is individual-feature masking

# Weighting to avoid ultra-rare features dominating
MIN_OBS_PROB = 0.02  # cap inverse-frequency weights

# Preprocessing
Z_CLIP_MIN = -10.0
Z_CLIP_MAX = 10.0

# Loss stability
LOGVAR_MIN = -6.0
LOGVAR_MAX = 6.0
VAR_FLOOR = 1e-3


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


@dataclass
class NumericTransformSpec:
    """
    Serializable preprocessing specification for numeric transforms.
    kind:
      - "abs_log1p": x <- log1p(abs(x))
    """
    name: str
    kind: str
    columns: List[str]


def load_manifest(manifest_path: str) -> Dict:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Could not find Phase 1 manifest at: {manifest_path}\n"
            "Phase 2.1 requires the manifest to select leakage-safe features."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_feature_list_from_manifest(manifest: Dict) -> Tuple[List[str], Dict]:
    """
    Features are:
      - union of per-mission leakage_filter.feature_cols
      - union of per-mission canonical.created keys (canonical column names)
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
    existing = [c for c in feature_whitelist if c in df.columns]
    numeric_cols, cat_cols = [], []
    for c in existing:
        dt = df[c].dtype
        if pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)
    return sorted(numeric_cols), sorted(cat_cols)


def build_cat_vocabs(df: pd.DataFrame, cat_cols: List[str], max_vocab: int = 50000) -> Dict[str, Dict[str, int]]:
    """
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
        # common identifiers / provenance / timestamps (must never be features)
        "kepid", "kepoi_name", "kepler_name",
        "tid", "toi", "toidisplay", "toipfx", "ctoi_alias",
        "pl_name", "pl_letter", "k2_name", "epic_hostname", "epic_candname", "hostname",
        "hd_name", "hip_name", "tic_id", "gaia_dr2_id", "gaia_dr3_id",
        "rowupdate", "release_date", "toi_created", "pl_pubdate", "releasedate", "disc_pubdate",
        "source_table", "source_key",
        "snapshot_time_utc", "dataset_version_id",
        # old aliases
        "y_source_col", "y_source_val", "y3",
    }
    offenders = sorted([c for c in features if c in banned])
    if offenders:
        raise RuntimeError(
            "Leakage/provenance columns found in feature list derived from manifest.\n"
            f"Offending columns: {offenders}"
        )


def build_feature_families_from_manifest(manifest: Dict, numeric_cols: List[str]) -> Dict[str, List[int]]:
    """
    Returns family -> list of indices into numeric_cols.
    Tries to use manifest['feature_families'] if present; otherwise falls back to a single family.
    """
    families: Dict[str, List[int]] = {}
    ff = manifest.get("feature_families", None)

    if isinstance(ff, dict) and len(ff) > 0:
        for fam, cols in ff.items():
            if not isinstance(cols, list):
                continue
            idxs = [numeric_cols.index(c) for c in cols if c in numeric_cols]
            if idxs:
                families[fam] = idxs

    if not families:
        families["all_numeric"] = list(range(len(numeric_cols)))

    return families


def build_phys_core_indices(numeric_cols: List[str]) -> List[int]:
    phys_core = [
        "period_days",
        "duration_hours",
        "depth_frac",
        "planet_radius_rearth",
        "star_radius_rsun",
        "star_teff_k",
        "star_logg_cgs",
        "star_metallicity_dex",
        "distance_pc",
    ]
    return [numeric_cols.index(c) for c in phys_core if c in numeric_cols]


def compute_observed_probabilities(df: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
    obs = np.zeros(len(numeric_cols), dtype=np.float32)
    for j, c in enumerate(numeric_cols):
        obs[j] = float(pd.to_numeric(df[c], errors="coerce").notna().mean())
    return obs


def build_numeric_transform_specs(numeric_cols: List[str]) -> List[NumericTransformSpec]:
    """
    Current policy:
      - abs_log1p for any numeric col whose name contains 'err' (case-insensitive).
    """
    err_cols = [c for c in numeric_cols if "err" in c.lower()]
    specs: List[NumericTransformSpec] = []
    if err_cols:
        specs.append(NumericTransformSpec(
            name="abs_log1p_for_err",
            kind="abs_log1p",
            columns=err_cols,
        ))
    return specs


def apply_numeric_transforms(col: str, x: float, specs: List[NumericTransformSpec]) -> float:
    for spec in specs:
        if col in spec.columns:
            if spec.kind == "abs_log1p":
                return math.log1p(abs(x))
    return x


def compute_numeric_stats(
        df: pd.DataFrame,
        numeric_cols: List[str],
        transform_specs: List[NumericTransformSpec],
) -> Dict[str, Dict[str, float]]:
    """
    Compute z-score stats in TRANSFORMED space.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for c in numeric_cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(x) > 0:
            xt = x.apply(lambda v: apply_numeric_transforms(c, float(v), transform_specs))
            mu = float(xt.mean())
            sd = float(xt.std())
        else:
            mu, sd = 0.0, 1.0

        if not math.isfinite(sd) or sd < 1e-12:
            sd = 1.0

        stats[c] = {"mean": mu, "std": sd}
    return stats


# -----------------------------
# Dataset
# -----------------------------
class TabularTokenDatasetSSL(Dataset):
    """
    SSL dataset: returns numeric/cat tensors + missing masks + mission. No y.
    Numeric preprocessing:
      - apply transform specs (e.g. abs_log1p)
      - z-score normalize using transformed stats
      - optional z-clip
    """

    def __init__(
            self,
            df: pd.DataFrame,
            feat: FeatureSpec,
            num_stats: Dict[str, Dict[str, float]],
            transform_specs: List[NumericTransformSpec],
            z_clip: Optional[Tuple[float, float]] = (-10.0, 10.0),
    ):
        self.df = df.reset_index(drop=True)
        self.feat = feat
        self.num_stats = num_stats
        self.transform_specs = transform_specs
        self.z_clip = z_clip
        self.mission = self.df[MISSION_COL].astype(str).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        num_vals: List[float] = []
        num_missing: List[float] = []

        for c in self.feat.numeric_cols:
            v = row[c]
            missing = pd.isna(v)
            num_missing.append(1.0 if missing else 0.0)

            if missing:
                # mean is already in transformed space
                vt = float(self.num_stats[c]["mean"])
            else:
                vt = float(v)
                vt = apply_numeric_transforms(c, vt, self.transform_specs)

            z = (vt - self.num_stats[c]["mean"]) / self.num_stats[c]["std"]

            if self.z_clip is not None:
                z = float(np.clip(z, self.z_clip[0], self.z_clip[1]))

            num_vals.append(float(z))

        cat_ids: List[int] = []
        cat_missing: List[float] = []
        for c in self.feat.cat_cols:
            v = row[c]
            missing = pd.isna(v)
            cat_missing.append(1.0 if missing else 0.0)
            if missing:
                cat_ids.append(0)
            else:
                cat_ids.append(self.feat.cat_vocabs[c].get(str(v), 0))

        return {
            "num_vals": torch.tensor(num_vals, dtype=torch.float32),
            "num_missing": torch.tensor(num_missing, dtype=torch.float32),
            "cat_ids": torch.tensor(cat_ids, dtype=torch.long),
            "cat_missing": torch.tensor(cat_missing, dtype=torch.float32),
            "mission": self.mission[idx],
        }


# -----------------------------
# Model
# -----------------------------
class ColumnTokenTransformerMFM(nn.Module):
    def __init__(
            self,
            n_num: int,
            cat_cardinalities: List[int],
            d_model: int = 192,
            n_heads: int = 6,
            n_layers: int = 4,
            dropout: float = 0.1,
            n_missions: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_num = n_num

        # Tokens: [CLS] + [MISSION] + numeric tokens + categorical tokens
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mission_emb = nn.Embedding(n_missions, d_model)

        # Numeric token components
        self.num_col_emb = nn.Embedding(n_num, d_model)
        self.num_w = nn.Parameter(torch.randn(n_num, d_model) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(n_num, d_model))
        self.num_missing_emb = nn.Parameter(torch.zeros(d_model))
        self.num_masked_emb = nn.Parameter(torch.zeros(d_model))

        # Categorical token components (kept for future; ok if empty)
        self.cat_col_emb = nn.Embedding(len(cat_cardinalities), d_model)
        self.cat_tables = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cardinalities])
        self.cat_missing_emb = nn.Parameter(torch.zeros(d_model))
        self.cat_masked_emb = nn.Parameter(torch.zeros(d_model))

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

        # MFM head for numeric: predict mu + logvar from token outputs
        self.mfm_mu = nn.Linear(d_model, 1)
        self.mfm_logvar = nn.Linear(d_model, 1)

    @staticmethod
    def mission_to_id(mission: List[str]) -> torch.Tensor:
        mapping = {"kepler": 0, "tess": 1, "k2": 2}
        ids = [mapping.get(str(m).lower(), 0) for m in mission]
        return torch.tensor(ids, dtype=torch.long)

    def forward(self, num_vals, num_missing, num_mfm_mask, cat_ids, cat_missing, cat_mfm_mask, mission: List[str]):
        B = num_vals.size(0)
        device = num_vals.device

        cls_tok = self.cls.expand(B, -1, -1)
        mid = self.mission_to_id(mission).to(device)
        mission_tok = self.mission_emb(mid).unsqueeze(1)

        # Numeric tokens
        if num_vals.size(1) > 0:
            idx = torch.arange(num_vals.size(1), device=device)
            col = self.num_col_emb(idx).unsqueeze(0).expand(B, -1, -1)

            masked_vals = num_vals * (1.0 - num_mfm_mask)
            val = masked_vals.unsqueeze(-1)
            proj = val * self.num_w.unsqueeze(0) + self.num_b.unsqueeze(0)

            miss = num_missing.unsqueeze(-1) * self.num_missing_emb
            mfm = num_mfm_mask.unsqueeze(-1) * self.num_masked_emb
            num_tok = col + proj + miss + mfm
        else:
            num_tok = torch.zeros(B, 0, self.d_model, device=device)

        # Categorical tokens
        if cat_ids.size(1) > 0:
            cat_toks = []
            for j, table in enumerate(self.cat_tables):
                ids = cat_ids[:, j].clamp(min=0, max=table.num_embeddings - 1)
                emb = table(ids)
                col = self.cat_col_emb(torch.tensor(j, device=device)).unsqueeze(0).expand(B, -1)
                miss = cat_missing[:, j].unsqueeze(-1) * self.cat_missing_emb
                mfm = cat_mfm_mask[:, j].unsqueeze(-1) * self.cat_masked_emb
                cat_toks.append((emb + col + miss + mfm).unsqueeze(1))
            cat_tok = torch.cat(cat_toks, dim=1)
        else:
            cat_tok = torch.zeros(B, 0, self.d_model, device=device)

        x = torch.cat([cls_tok, mission_tok, num_tok, cat_tok], dim=1)
        x = self.dropout(x)
        x = self.encoder(x)

        # Numeric token outputs positions: 0=CLS, 1=MISSION, 2..2+n_num-1
        if self.n_num > 0:
            num_out = x[:, 2:2 + self.n_num, :]
            num_out = self.norm(num_out)
        else:
            num_out = torch.zeros(B, 0, self.d_model, device=device)

        mu = self.mfm_mu(num_out).squeeze(-1)
        logvar = self.mfm_logvar(num_out).squeeze(-1)
        return mu, logvar


# -----------------------------
# Masking
# -----------------------------
@torch.no_grad()
def build_mfm_mask_numeric(
        num_missing: torch.Tensor,
        obs_prob: np.ndarray,
        feature_families: Dict[str, List[int]],
        phys_core: List[int],
        p_mask_total: float,
        p_family: float,
        p_physcore: float,
        device: torch.device,
) -> torch.Tensor:
    """
    Returns num_mfm_mask: (B, n_num), masks only observed entries (num_missing==0).
    """
    B, n = num_missing.shape
    num_mfm_mask = torch.zeros((B, n), dtype=torch.float32, device=device)

    obs = (1.0 - num_missing).bool()

    # sampling weights: inverse frequency capped
    p_obs = np.clip(obs_prob, MIN_OBS_PROB, 1.0)
    w = (1.0 / p_obs).astype(np.float32)
    w = w / (w.sum() + 1e-8)
    w_t = torch.tensor(w, dtype=torch.float32, device=device)

    fam_names = list(feature_families.keys())

    for i in range(B):
        obs_idx = torch.where(obs[i])[0]
        if obs_idx.numel() == 0:
            continue

        k = max(1, int(math.ceil(p_mask_total * obs_idx.numel())))

        u = float(torch.rand(1).item())

        # physics-core masking
        if u < p_physcore and len(phys_core) > 0:
            pc = torch.tensor(phys_core, device=device, dtype=torch.long)
            pc_obs = pc[obs[i, pc]]
            if pc_obs.numel() > 0:
                kk = min(k, pc_obs.numel())
                sel = pc_obs[torch.randperm(pc_obs.numel(), device=device)[:kk]]
                num_mfm_mask[i, sel] = 1.0
                continue

        # family masking
        if u < (p_physcore + p_family) and len(fam_names) > 0:
            fam = fam_names[int(torch.randint(low=0, high=len(fam_names), size=(1,)).item())]
            idxs = feature_families[fam]
            fam_t = torch.tensor(idxs, device=device, dtype=torch.long)
            fam_obs = fam_t[obs[i, fam_t]]
            if fam_obs.numel() > 0:
                kk = min(k, fam_obs.numel())
                sel = fam_obs[torch.randperm(fam_obs.numel(), device=device)[:kk]]
                num_mfm_mask[i, sel] = 1.0
                continue

        # individual-feature masking
        sel: List[int] = []
        attempts = 0
        while len(sel) < k and attempts < 10_000:
            j = int(torch.multinomial(w_t, num_samples=1).item())
            attempts += 1
            if obs[i, j] and (j not in sel):
                sel.append(j)
        if sel:
            num_mfm_mask[i, torch.tensor(sel, device=device)] = 1.0

    return num_mfm_mask


# -----------------------------
# Loss
# -----------------------------
def gaussian_nll(z_true: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)
    var = torch.exp(logvar) + VAR_FLOOR
    return 0.5 * (torch.log(var) + (z_true - mu) ** 2 / var)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_mfm(
        model: nn.Module,
        loader: DataLoader,
        device: str,
        obs_prob: np.ndarray,
        feature_families: Dict[str, List[int]],
        phys_core: List[int],
) -> Dict:
    model.eval()
    losses: List[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            num_vals = batch["num_vals"].to(device)
            num_missing = batch["num_missing"].to(device)

            num_mfm_mask = build_mfm_mask_numeric(
                num_missing=num_missing,
                obs_prob=obs_prob,
                feature_families=feature_families,
                phys_core=phys_core,
                p_mask_total=P_MASK_TOTAL,
                p_family=P_FAMILY_MASK,
                p_physcore=P_PHYSCORE_MASK,
                device=num_vals.device,
            )

            mu, logvar = model(
                num_vals=num_vals,
                num_missing=num_missing,
                num_mfm_mask=num_mfm_mask,
                cat_ids=batch["cat_ids"].to(device),
                cat_missing=batch["cat_missing"].to(device),
                cat_mfm_mask=torch.zeros_like(batch["cat_missing"]).to(device),
                mission=batch["mission"],
            )

            mask = (num_mfm_mask > 0.5) & (num_missing < 0.5)
            if not mask.any():
                continue

            nll = gaussian_nll(num_vals, mu, logvar)
            batch_loss = nll[mask].mean()
            loss_val = float(batch_loss.cpu())
            losses.append(loss_val)

    return {"mfm_nll": float(np.mean(losses)) if losses else float("nan")}


# -----------------------------
# Scheduler
# -----------------------------
def make_warmup_cosine_scheduler(optimizer, num_updates: int, warmup_frac: float, min_lr_frac: float):
    warmup_updates = max(1, int(num_updates * warmup_frac))

    def lr_lambda(update_idx: int):
        if update_idx < warmup_updates:
            return float(update_idx + 1) / float(warmup_updates)
        progress = (update_idx - warmup_updates) / max(1, (num_updates - warmup_updates))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_frac + (1.0 - min_lr_frac) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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

    if MISSION_COL not in df.columns:
        raise KeyError(f"Expected column '{MISSION_COL}' not found in merged parquet.")

    # Use ALL rows for SSL, keep only rows with known mission
    df = df[df[MISSION_COL].notna()].copy()
    df[MISSION_COL] = df[MISSION_COL].astype(str)

    feature_whitelist_existing = [c for c in feature_whitelist if c in df.columns]
    missing_from_parquet = [c for c in feature_whitelist if c not in df.columns]

    numeric_cols, cat_cols = infer_feature_types_from_whitelist(df, feature_whitelist_existing)
    assert_no_leakage_columns_in_features(numeric_cols + cat_cols)

    print(f"Rows (ALL for SSL): {len(df)}")
    print(
        f"Feature whitelist (manifest-derived): {len(feature_whitelist)} "
        f"(present in parquet: {len(feature_whitelist_existing)}, missing: {len(missing_from_parquet)})"
    )
    print(f"Numeric cols: {len(numeric_cols)} | Categorical cols: {len(cat_cols)}")

    # SSL split: stratify by mission only
    strat = df[MISSION_COL].astype(str)
    train_df, val_df = train_test_split(df, test_size=VAL_SIZE, random_state=SEED, stratify=strat)

    # Fit preprocessing on train only (THIS becomes canonical for 2.2/2.3)
    cat_vocabs = build_cat_vocabs(train_df, cat_cols)
    transform_specs = build_numeric_transform_specs(numeric_cols)
    num_stats = compute_numeric_stats(train_df, numeric_cols, transform_specs)

    feat = FeatureSpec(numeric_cols=numeric_cols, cat_cols=cat_cols, cat_vocabs=cat_vocabs)

    # For masking weights
    obs_prob = compute_observed_probabilities(train_df, numeric_cols)
    feature_families = build_feature_families_from_manifest(manifest, numeric_cols)
    phys_core = build_phys_core_indices(numeric_cols)

    # Datasets
    z_clip = (Z_CLIP_MIN, Z_CLIP_MAX) if (Z_CLIP_MIN is not None and Z_CLIP_MAX is not None) else None
    train_ds = TabularTokenDatasetSSL(train_df, feat, num_stats, transform_specs, z_clip=z_clip)
    val_ds = TabularTokenDatasetSSL(val_df, feat, num_stats, transform_specs, z_clip=z_clip)

    train_loader = DataLoader(train_ds, batch_size=MICRO_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=MICRO_BATCH_SIZE, shuffle=False, num_workers=0)

    cat_cards = [max(cat_vocabs[c].values(), default=0) + 1 for c in cat_cols]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = ColumnTokenTransformerMFM(
        n_num=len(numeric_cols),
        cat_cardinalities=cat_cards,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        n_missions=3,
    ).to(device)

    # Effective batch + adaptive LR (sqrt scaling)
    base_ref_batch = 256
    base_lr_ref = 3e-4
    effective_batch = MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS
    lr = base_lr_ref * math.sqrt(effective_batch / base_ref_batch)

    print(f"Micro batch: {MICRO_BATCH_SIZE} | Grad accum: {GRAD_ACCUM_STEPS} | Effective batch: {effective_batch}")
    print(f"Adaptive LR (sqrt scaling from {base_lr_ref} @ {base_ref_batch}): {lr:.6g}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    micro_steps_per_epoch = math.ceil(len(train_ds) / MICRO_BATCH_SIZE)
    updates_per_epoch = math.ceil(micro_steps_per_epoch / GRAD_ACCUM_STEPS)
    total_updates = updates_per_epoch * EPOCHS

    scheduler = make_warmup_cosine_scheduler(
        opt,
        num_updates=total_updates,
        warmup_frac=WARMUP_FRAC,
        min_lr_frac=MIN_LR_FRAC,
    )

    best_val = float("inf")
    best_path = os.path.join(OUTDIR, "best_encoder.pt")
    patience_left = EARLY_STOP_PATIENCE

    history = {"train_mfm_nll": [], "val_mfm_nll": [], "lr": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses: List[float] = []
        opt.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Pretrain {epoch}/{EPOCHS}", leave=False)
        for micro_step, batch in enumerate(pbar, start=1):
            num_vals = batch["num_vals"].to(device)
            num_missing = batch["num_missing"].to(device)

            num_mfm_mask = build_mfm_mask_numeric(
                num_missing=num_missing,
                obs_prob=obs_prob,
                feature_families=feature_families,
                phys_core=phys_core,
                p_mask_total=P_MASK_TOTAL,
                p_family=P_FAMILY_MASK,
                p_physcore=P_PHYSCORE_MASK,
                device=num_vals.device,
            )

            mu, logvar = model(
                num_vals=num_vals,
                num_missing=num_missing,
                num_mfm_mask=num_mfm_mask,
                cat_ids=batch["cat_ids"].to(device),
                cat_missing=batch["cat_missing"].to(device),
                cat_mfm_mask=torch.zeros_like(batch["cat_missing"]).to(device),
                mission=batch["mission"],
            )

            mask = (num_mfm_mask > 0.5) & (num_missing < 0.5)
            if not mask.any():
                continue

            nll = gaussian_nll(num_vals, mu, logvar)
            loss = nll[mask].mean()

            (loss / GRAD_ACCUM_STEPS).backward()

            lv = float(loss.detach().cpu())
            losses.append(lv)

            if (micro_step % GRAD_ACCUM_STEPS) == 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                opt.zero_grad(set_to_none=True)
                scheduler.step()

            current_lr = opt.param_groups[0]["lr"]
            pbar.set_postfix(mfm_nll=f"{lv:.4f}", lr=f"{current_lr:.2e}")

        # flush remainder if not divisible
        if (micro_steps_per_epoch % GRAD_ACCUM_STEPS) != 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()

        train_nll = float(np.mean(losses)) if losses else float("nan")
        val_out = evaluate_mfm(model, val_loader, device, obs_prob, feature_families, phys_core)
        val_nll = float(val_out["mfm_nll"])

        history["train_mfm_nll"].append(train_nll)
        history["val_mfm_nll"].append(val_nll)
        history["lr"].append(float(opt.param_groups[0]["lr"]))

        print(
            f"Epoch {epoch:03d} | train_mfmNLL={train_nll:.4f} | val_mfmNLL={val_nll:.4f} | lr={opt.param_groups[0]['lr']:.2e}")

        improved = val_nll < best_val - EARLY_STOP_MIN_DELTA
        if improved:
            best_val = val_nll
            patience_left = EARLY_STOP_PATIENCE
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch:03d}. Best val_mfmNLL={best_val:.6f}")
                break

        if device == "cuda":
            torch.cuda.empty_cache()

    # Save artifacts (preprocessing is frozen here for 2.2/2.3)
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
        "preprocessing": {
            "numeric": {
                "normalization": "zscore",
                "stats_fit_split": "ssl_train_only",
                "transforms": [asdict(s) for s in transform_specs],
                "z_clip": {"min": Z_CLIP_MIN, "max": Z_CLIP_MAX} if z_clip is not None else None,
            }
        },
        "ssl_split": {
            "val_size": VAL_SIZE,
            "stratification": "mission",
        },
        "mfm_config": {
            "p_mask_total": P_MASK_TOTAL,
            "p_family_mask": P_FAMILY_MASK,
            "p_physcore_mask": P_PHYSCORE_MASK,
            "feature_families": {k: len(v) for k, v in feature_families.items()},
            "phys_core_present": [numeric_cols[i] for i in phys_core],
        },
        "loss_config": {
            "type": "gaussian_nll",
            "logvar_clip": {"min": LOGVAR_MIN, "max": LOGVAR_MAX},
            "var_floor": VAR_FLOOR,
        },
        "optimization": {
            "micro_batch_size": MICRO_BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": effective_batch,
            "lr_ref_batch": base_ref_batch,
            "lr_ref_value": base_lr_ref,
            "lr_effective": lr,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "scheduler": {
                "type": "warmup+cosine",
                "warmup_frac": WARMUP_FRAC,
                "min_lr_frac": MIN_LR_FRAC,
                "total_updates": total_updates,
                "updates_per_epoch": updates_per_epoch,
            },
        },
        "hyperparams": {
            "seed": SEED,
            "epochs": EPOCHS,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        },
        "history": history,
    }

    with open(os.path.join(OUTDIR, "artifacts.json"), "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    with open(os.path.join(OUTDIR, "pretrain_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_val_mfm_nll": best_val, "history": history}, f, indent=2)

    print(f"\nSaved pretraining run to: {OUTDIR}")
    print(f"Best val MFM NLL: {best_val:.6f}")
    print(f"Best encoder checkpoint: {best_path}")


if __name__ == "__main__":
    main()
