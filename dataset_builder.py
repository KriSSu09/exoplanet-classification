"""
Phase 1 dataset builder (schema-locked using the 3 Exoplanet Archive column-definition pages)

Key locked decisions (from docs):
- Kepler KOI table is `cumulative`, label source is `koi_disposition` with values:
  CANDIDATE / FALSE POSITIVE / NOT DISPOSITIONED / CONFIRMED. :contentReference[oaicite:0]{index=0}
- TESS TOI label source is `tfopwg_disp` with values:
  APC/CP/FA/FP/KP/PC. :contentReference[oaicite:1]{index=1}
- K2 Planets+Candidates label source is `disposition` with values including:
  CANDIDATE / FALSE POSITIVE / CONFIRMED / REFUTED. :contentReference[oaicite:2]{index=2}

Units sanity (from docs):
- KOI: koi_period [days], koi_time0bk [BJD-2454833], koi_duration [hours], koi_depth [ppm]. :contentReference[oaicite:3]{index=3}
- TOI: pl_orbper [days], pl_tranmid [BJD], pl_trandurh [hours], pl_trandep [ppm]. :contentReference[oaicite:4]{index=4}
- K2: pl_orbper [days], pl_trandur [hours], pl_trandep [%], pl_tranmid [days] (time system described by pl_tsystemref). :contentReference[oaicite:5]{index=5}

Leakage policy:
- Drop all vetting/disposition fields AND *all pipeline scores/metrics* (score/rank/classifier outputs).
- Drop free text and links.
- Keep IDs/names only as metadata columns (not model features).

Uncertainties/limits decision:
- Include uncertainties/limits ONLY for a curated set of physics-critical bases to avoid feature explosion,
  while still capturing measurement quality:
  period, epoch, duration, depth, planet radius, stellar radius, Teff, logg, distance/parallax.
  (All 3 sources document uncertainties/limits broadly; TOI and K2 explicitly provide err1/err2/lim columns. :contentReference[oaicite:6]{index=6})

Outputs:
- merged parquet (wide)
- metadata JSON (queries + schema + denylist report + missingness stats)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import requests
from pandas import DataFrame

# =============================
# Exoplanet Archive TAP config
# =============================

EXOARCHIVE_TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

MISSION_TABLES = {
    # docs pages you gave are for `cumulative`, `toi`, `k2pandc`
    "kepler": "cumulative",
    "tess": "toi",
    "k2": "k2pandc",
}

# -----------------------------
# Locked label columns per table
# -----------------------------
LABEL_SOURCE_COL = {
    "kepler": "koi_disposition",  # :contentReference[oaicite:7]{index=7}
    "tess": "tfopwg_disp",  # :contentReference[oaicite:8]{index=8}
    "k2": "disposition",  # :contentReference[oaicite:9]{index=9}
}

# 3-class target (plus unlabeled)
LABEL_COL = "target_class"  # {positive, candidate, negative} or NaN

LABEL_MAP_BY_MISSION = {
    "kepler": {
        "CONFIRMED": "positive",
        "CANDIDATE": "candidate",
        "FALSE POSITIVE": "negative",
        "NOT DISPOSITIONED": None,
    },
    "tess": {
        "CP": "positive",
        "KP": "positive",
        "PC": "candidate",
        "APC": "candidate",
        "FP": "negative",
        "FA": "negative",
        None: None,
        "": None,
    },
    "k2": {
        "CONFIRMED": "positive",
        "CANDIDATE": "candidate",
        "FALSE POSITIVE": "negative",
        "REFUTED": "negative",
        # sometimes appears with bracket annotations in docs; normalize below
    },
}

# =============================
# Leakage + roles
# =============================

MISSION_COL = "mission"
ROW_UID_COL = "row_uid"

# Keep these as metadata (for provenance), but we will NOT treat them as model features.
META_COLS_COMMON = {MISSION_COL, ROW_UID_COL, LABEL_COL, "label_source", "label_source_value"}
META_COLS_BY_MISSION = {
    "kepler": ["kepid", "kepoi_name", "kepler_name"],
    "tess": ["tid", "toi", "toidisplay", "toipfx", "ctoi_alias"],
    "k2": ["pl_name", "pl_letter", "k2_name", "epic_hostname", "epic_candname",
           "hostname", "hd_name", "hip_name", "tic_id", "gaia_dr2_id", "gaia_dr3_id"],
}

# Exact denylist: always drop as features and also drop from output unless in META allowlist.
DENY_EXACT = {
    # label sources and other disposition-like
    "koi_disposition", "koi_pdisposition", "koi_vet_stat", "koi_vet_date", "koi_score",
    "tfopwg_disp",
    "disposition", "disp_refname",
    # links / text
    "koi_comment", "koi_datalink_dvr", "koi_datalink_dvs",
    # mission-specific
    "default_flag",
    "pl_pnum",
}

# Regex denylist:
# - drop any *scores/metrics* likely to encode pipeline decisions
# - drop any vetting flags / FP flags
# - drop URLs, comments, provenance strings that could leak
DENY_REGEX = [
    r".*dispos.*",
    r".*vet.*",
    r".*fpflag.*",
    r".*robovet.*",
    r".*autovet.*",
    r".*score.*",  # drop all scores/metrics to be safe (your decision)
    r".*rank.*",
    r".*prob.*",  # FP prob tables exist; keep out of phase 1 to avoid leakage
    r".*comment.*",
    r".*datalink.*",
    r".*url.*",
    r".*prov.*",  # provenance text often correlates with processing stage
    r".*refname.*",  # publication refs are not features
    r".*bibcode.*",
    r"^pl_n.*",
    r"^st_n.*",
    r"^disc_.*",
    r"^discoverymethod$",
]

_DENY_RE = [re.compile(p, flags=re.IGNORECASE) for p in DENY_REGEX]


def _should_drop(col: str) -> bool:
    c = col.strip()
    if c in DENY_EXACT:
        return True
    for r in _DENY_RE:
        if r.fullmatch(c) or r.match(c):
            return True
    return False


def _is_identity_like(col: str) -> bool:
    """
    True if col is an identifier/name/alias/designation-like field.
    Keep in parquet as metadata if desired, but do NOT treat as a model feature.
    """
    c = col.strip().lower()

    # Strong exact triggers
    if c in {
        "kepid", "kepoi_name", "kepler_name",
        "tid", "toi", "toidisplay", "toipfx", "ctoi_alias",
        "pl_name", "pl_letter", "k2_name", "epic_hostname", "epic_candname",
        "hostname", "hd_name", "hip_name",
        "tic_id", "gaia_dr2_id", "gaia_dr3_id",
    }:
        return True

    # Pattern triggers
    if "alias" in c or "designation" in c:
        return True
    if c.endswith("_name") or c.endswith("name"):
        return True
    if "hostname" in c:
        return True

    # Catalog IDs / target IDs: keep as metadata only
    if c.endswith("_id") or c == "id":
        return True

    return False


def _is_stringified_numeric(col: str) -> bool:
    """
    True if col is one of the Exoplanet Archive '*str' columns that duplicate numeric values.
    These should not be fed to the model.
    """
    return col.strip().lower().endswith("str") or col.strip().lower().endswith("_str")


# =============================
# Uncertainty / limit inclusion
# =============================

# Curated “physics-critical” bases (canonical intent)
# We include err1/err2/err/lim columns ONLY if their base feature matches one of these.
UNC_BASE_ALLOWLIST = {
    # orbital / transit
    "koi_period", "koi_time0bk", "koi_time0", "koi_duration", "koi_depth",
    "pl_orbper", "pl_tranmid", "pl_trandurh", "pl_trandur", "pl_trandep",
    # radii
    "koi_prad", "koi_srad",
    "pl_rade", "st_rad",
    # stellar
    "koi_steff", "koi_slogg", "koi_smet",
    "st_teff", "st_logg", "st_met",
    # distance
    "st_dist", "sy_dist",
    # (add parallax fields if present in tables you pull)
}

UNC_PATTERNS = [
    re.compile(r"^(?P<base>.+)_err1$", re.IGNORECASE),
    re.compile(r"^(?P<base>.+)_err2$", re.IGNORECASE),
    re.compile(r"^(?P<base>.+)_err$", re.IGNORECASE),
    re.compile(r"^(?P<base>.+)_lim$", re.IGNORECASE),
    re.compile(r"^(?P<base>.+)err1$", re.IGNORECASE),
    re.compile(r"^(?P<base>.+)err2$", re.IGNORECASE),
    re.compile(r"^(?P<base>.+)err$", re.IGNORECASE),
    re.compile(r"^(?P<base>.+)lim$", re.IGNORECASE),
]


def _is_allowed_unc_col(col: str) -> bool:
    c = col.strip()
    for pat in UNC_PATTERNS:
        m = pat.match(c)
        if m:
            base = m.group("base")
            return base in UNC_BASE_ALLOWLIST
    return False


# =============================
# Canonical unit harmonization
# =============================

# Canonical features we will *add* (keeping raw columns too)
# Depth: canonicalize to fractional (unitless)
# - KOI depth is ppm :contentReference[oaicite:10]{index=10}
# - TOI depth is ppm :contentReference[oaicite:11]{index=11}
# - K2 depth is % :contentReference[oaicite:12]{index=12}
CANONICAL_MAP = {
    "period_days": [("kepler", "koi_period"), ("tess", "pl_orbper"), ("k2", "pl_orbper")],
    "duration_hours": [("kepler", "koi_duration"), ("tess", "pl_trandurh"), ("k2", "pl_trandur")],
    "epoch_days": [("kepler", "koi_time0bk"), ("tess", "pl_tranmid"), ("k2", "pl_tranmid")],
    "depth_frac": [("kepler", "koi_depth"), ("tess", "pl_trandep"), ("k2", "pl_trandep")],
    "planet_radius_rearth": [("kepler", "koi_prad"), ("tess", "pl_rade"), ("k2", "pl_rade")],
    "star_radius_rsun": [("kepler", "koi_srad"), ("tess", "st_rad"), ("k2", "st_rad")],
    "star_teff_k": [("kepler", "koi_steff"), ("tess", "st_teff"), ("k2", "st_teff")],
    "star_logg_cgs": [("kepler", "koi_slogg"), ("tess", "st_logg"), ("k2", "st_logg")],
    "star_metallicity_dex": [("kepler", "koi_smet"), ("k2", "st_met")],
    # TOI table doesn’t list st_met in your doc excerpt
    "distance_pc": [("tess", "st_dist"), ("k2", "sy_dist")],
}


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _depth_to_frac(mission: str, depth: pd.Series) -> pd.Series:
    d = _to_numeric(depth)
    if mission in ("kepler", "tess"):
        # ppm -> fraction
        return d * 1e-6
    if mission == "k2":
        # percent -> fraction
        return d / 100.0
    return d


# =============================
# TAP helpers
# =============================

def _one_line_sql(sql: str) -> str:
    # collapse all whitespace (incl newlines/tabs) to single spaces
    return re.sub(r"\s+", " ", sql).strip()


def tap_sync_query(sql: str, timeout_s: int = 120) -> pd.DataFrame:
    params = {
        "query": _one_line_sql(sql),
        "format": "json",
        "lang": "ADQL",
    }
    r = requests.get(EXOARCHIVE_TAP_SYNC, params=params, timeout=timeout_s)
    # Helpful debugging if something else goes wrong:
    if r.status_code >= 400:
        raise requests.HTTPError(
            f"HTTP {r.status_code} from TAP\n"
            f"Query: {params['query']}\n"
            f"Response: {r.text[:2000]}",
            response=r,
        )
    return pd.DataFrame(r.json())


def resolve_tap_table_id(table: str, timeout_s: int) -> Tuple[str, str]:
    """
    Resolve user-facing table name (e.g. 'cumulative') to the published TAP (schema_name, table_name),
    e.g. ('exo_tap', 'CUMULATIVE').
    """
    sql = f"""
    SELECT schema_name, table_name
    FROM TAP_SCHEMA.tables
    WHERE LOWER(table_name) = LOWER('{table}')
    """
    df = tap_sync_query(sql, timeout_s=timeout_s)
    if df.empty:
        # fallback: try substring match (useful if user passes schema-qualified name or slight variants)
        sql2 = f"""
        SELECT schema_name, table_name
        FROM TAP_SCHEMA.tables
        WHERE LOWER(table_name) LIKE LOWER('%{table}%')
        """
        df2 = tap_sync_query(sql2, timeout_s=timeout_s)
        if df2.empty:
            raise ValueError(
                f"Could not resolve table '{table}' in TAP_SCHEMA.tables. "
                f"Try inspecting https://exoplanetarchive.ipac.caltech.edu/TAP/tables"
            )
        # pick best match: exact case-insensitive match if present, else first
        exact = df2[df2["table_name"].str.lower() == table.lower()]
        row = (exact.iloc[0] if not exact.empty else df2.iloc[0])
        return str(row["schema_name"]), str(row["table_name"])

    row = df.iloc[0]
    return str(row["schema_name"]), str(row["table_name"])


def fetch_table_all(table: str, timeout_s: int, max_rows: int | None = None) -> pd.DataFrame:
    schema_name, table_name = resolve_tap_table_id(table, timeout_s=timeout_s)
    fqtn = f"{schema_name}.{table_name}"  # schema-qualified table name
    top = f"TOP {int(max_rows)}" if max_rows is not None else ""
    sql = f"SELECT {top} * FROM {fqtn}"
    return tap_sync_query(sql, timeout_s=timeout_s)


# =============================
# Build steps
# =============================

def _normalize_label_string(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.upper().replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Handle K2 bracket annotations like "FALSE POSITIVE [CANDIDATE]" or "REFUTED [PLANET]"
    s = re.sub(r"\s*\[.*]\s*$", "", s).strip()

    return s


def add_labels(df: pd.DataFrame, mission: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    src_col = LABEL_SOURCE_COL[mission]
    if src_col not in df.columns:
        raise KeyError(f"Expected label source col '{src_col}' not found for mission={mission}")

    raw = df[src_col].map(_normalize_label_string)
    mapping = LABEL_MAP_BY_MISSION[mission]

    y = raw.map(lambda v: mapping.get(v, None))
    df = df.copy()
    df["label_source"] = src_col
    df["label_source_value"] = raw
    df[LABEL_COL] = y.astype("object")  # keep NaN for unlabeled

    report = {
        "mission": mission,
        "label_source_col": src_col,
        "counts": {str(k): int(v) for k, v in y.value_counts(dropna=False).to_dict().items()},
    }
    return df, report


def apply_supervised_scope_filters(df: pd.DataFrame, mission: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Phase-1 scope decisions:
    - Kepler supervised stability: koi_vet_stat = DONE for labeled rows (Q1 agreed). :contentReference[oaicite:13]{index=13}
    - K2: default_flag = 1 (canonical parameter set). :contentReference[oaicite:14]{index=14}
    """
    df = df.copy()
    report: Dict[str, Any] = {"mission": mission, "filters": []}

    if mission == "kepler":
        if "koi_vet_stat" in df.columns:
            # keep all for self-supervised pool; but we enforce DONE later by marking supervised_eligible
            done = df["koi_vet_stat"].astype(str).str.upper().eq("DONE")
            labeled = df[LABEL_COL].notna()
            df["supervised_eligible"] = (~labeled) | done
            report["filters"].append({"supervised_eligible": "unlabeled OR koi_vet_stat == DONE"})
        else:
            # if missing, fail loudly (we don’t want silent drift)
            raise KeyError("Kepler table missing koi_vet_stat; cannot enforce DONE-only supervised stability.")
    else:
        df["supervised_eligible"] = True

    if mission == "k2":
        if "default_flag" in df.columns:
            before = len(df)
            df = df[df["default_flag"] == 1].copy()
            report["filters"].append({"default_flag": "==1", "rows_removed": before - len(df)})
        else:
            raise KeyError("K2 table missing default_flag; cannot select default parameter set.")

    return df, report


def split_roles_and_filter_columns(
        df: pd.DataFrame,
        mission: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Output contains:
      - metadata columns (kept in parquet)
      - feature columns (safe for model)

    Refinements:
      - identity-like + name/id fields are metadata-only
      - '*str' columns are dropped from features
      - catalog/discovery timestamps are metadata-only (avoid temporal artifacts)
      - uncertainties/limits: keep only for curated bases
    """
    df = df.copy()

    # meta columns: common + mission-specific (deterministic list)
    meta_keep = set(META_COLS_COMMON)
    for c in META_COLS_BY_MISSION.get(mission, []):
        if c in df.columns:
            meta_keep.add(c)

    # metadata-only fields (kept in parquet, excluded from features)
    META_ONLY_EXACT = {
        "rowupdate", "release_date", "toi_created",
        "pl_pubdate", "releasedate", "disc_pubdate",
    }

    kept_cols: List[str] = []
    dropped: List[str] = []
    dropped_reason: Dict[str, str] = {}
    feature_cols: List[str] = []

    for c in df.columns:
        lc = c.strip().lower()

        # Always keep metadata
        if c in meta_keep or c == "supervised_eligible":
            kept_cols.append(c)
            continue

        # Explicit metadata-only catalog timestamps
        if lc in META_ONLY_EXACT:
            kept_cols.append(c)
            continue

        # Drop leakage and pipeline proxies
        if _should_drop(c):
            dropped.append(c)
            dropped_reason[c] = "denylist"
            continue

        # Drop stringified numeric duplicates
        if _is_stringified_numeric(c):
            dropped.append(c)
            dropped_reason[c] = "stringified_numeric"
            continue

        # Identity-like fields: keep only if you explicitly listed them in META_COLS_BY_MISSION
        if _is_identity_like(c):
            dropped.append(c)
            dropped_reason[c] = "identity_like"
            continue

        # Uncertainties/limits: keep only if allowed
        if any(p.match(c) for p in UNC_PATTERNS):
            if _is_allowed_unc_col(c):
                kept_cols.append(c)
                feature_cols.append(c)
            else:
                dropped.append(c)
                dropped_reason[c] = "unc_not_allowed"
            continue

        # Otherwise: keep as feature
        kept_cols.append(c)
        feature_cols.append(c)

    out = df[kept_cols].copy()
    report = {
        "mission": mission,
        "n_in": int(df.shape[1]),
        "n_out": int(out.shape[1]),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "dropped_reason_counts": pd.Series(list(dropped_reason.values())).value_counts().to_dict(),
        "dropped_sample": dropped[:200],
    }
    return out, report


def drop_high_missingness_features(
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        threshold: float,
) -> tuple[DataFrame | None, list[str], dict[str, float | int | Any]]:
    miss = df[feature_cols].isna().mean()
    drop = miss[miss >= threshold].index.tolist()

    keep_cols = [c for c in feature_cols if c not in drop]
    df = df.drop(columns=drop)

    report = {
        "threshold": threshold,
        "dropped": drop,
        "n_dropped": len(drop),
    }
    return df, keep_cols, report


def add_row_uid(df: pd.DataFrame, mission: str) -> pd.DataFrame:
    df = df.copy()

    id_cols = [c for c in META_COLS_BY_MISSION.get(mission, set()) if c in df.columns]

    if id_cols:
        # Make absolutely sure everything is a string and NaNs become empty
        key_df = df[id_cols].copy()
        for c in id_cols:
            col = key_df[c]
            # Convert NaN/None to empty string, then to str
            key_df[c] = col.where(~col.isna(), "").map(lambda v: "" if v is None else str(v))
        key = key_df.apply(lambda row: "|".join(row.values.tolist()), axis=1)
    else:
        key = df.index.map(str)

    def _hash(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    df[MISSION_COL] = mission
    df[ROW_UID_COL] = (mission + "|" + key).map(_hash)
    return df


def add_canonical_features(df: pd.DataFrame, mission: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    created: Dict[str, str] = {}
    missing: List[str] = []

    for canon, sources in CANONICAL_MAP.items():
        src = next((col for (m, col) in sources if m == mission and col in df.columns), None)
        if src is None:
            missing.append(canon)
            continue

        if canon == "depth_frac":
            df[canon] = _depth_to_frac(mission, df[src])
        else:
            df[canon] = _to_numeric(df[src])

        created[canon] = src

    report = {"mission": mission, "created": created, "missing": missing}
    return df, report


def coerce_object_columns_to_numeric(
        df: pd.DataFrame,
        candidate_cols: Sequence[str],
        min_convert_frac: float = 0.98,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Many TAP JSON columns arrive as object dtype even when they are numeric.
    This function attempts numeric coercion on specified columns.

    Rule: if >= min_convert_frac of *non-null* values can be converted, replace column with numeric.
    """
    df = df.copy()
    converted: Dict[str, str] = {}
    skipped: List[str] = []

    for c in candidate_cols:
        if c not in df.columns:
            continue
        if not (df[c].dtype == object or str(df[c].dtype).lower() == "string"):
            continue

        s = df[c]
        non_null = s.notna().sum()
        if non_null == 0:
            skipped.append(c)
            continue

        num = pd.to_numeric(s, errors="coerce")
        ok = num.notna().sum()

        if ok / non_null >= min_convert_frac:
            df[c] = num
            converted[c] = f"object->numeric ({ok}/{non_null}={ok / non_null:.3f})"
        else:
            skipped.append(c)

    report = {"converted": converted, "skipped_count": len(skipped)}
    return df, report


def build_feature_families(columns: Sequence[str], dtypes: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Feature-family grouping for explainability.
    Uses dtypes to separate categorical vs numeric.
    """
    fams: Dict[str, List[str]] = {
        "canonical_orbital": [],
        "canonical_transit_shape": [],
        "canonical_stellar": [],
        "canonical_distance": [],
        "astrometry": [],
        "photometry": [],
        "flags": [],
        "uncertainty_limit": [],
        "other_numeric": [],
        "other_categorical": [],
        "catalog_timestamps": [],
    }

    for c in columns:
        lc = c.lower()

        # uncertainty/limit metadata
        if any(p.match(c) for p in UNC_PATTERNS):
            fams["uncertainty_limit"].append(c)
            continue

        # canonical physics (created features)
        if lc in {"period_days", "duration_hours", "epoch_days"}:
            fams["canonical_orbital"].append(c)
            continue
        if lc in {"depth_frac", "planet_radius_rearth"}:
            fams["canonical_transit_shape"].append(c)
            continue
        if lc in {"star_radius_rsun", "star_teff_k", "star_logg_cgs", "star_metallicity_dex"}:
            fams["canonical_stellar"].append(c)
            continue
        if lc == "distance_pc":
            fams["canonical_distance"].append(c)
            continue

        # astrometry-ish
        if lc == "ra" or lc == "dec" or lc.startswith("ra") or lc.startswith("dec"):
            fams["astrometry"].append(c)
            continue

        # photometry-ish
        if ("mag" in lc) or lc.endswith("tmag") or lc.endswith("kepmag"):
            fams["photometry"].append(c)
            continue

        # flags
        if lc.endswith("_flag") or lc.endswith("flag"):
            fams["flags"].append(c)
            continue

        # catalog timestamps / publication fields (separate family for explainability)
        if any(tok in lc for tok in ["rowupdate", "release", "created", "pubdate", "releasedate", "disc_pubdate"]):
            fams.setdefault("catalog_timestamps", []).append(c)
            continue

        # dtype-aware fallback
        dt = dtypes.get(c)
        # pandas dtype.kind: 'iufc' numeric; 'b' bool; 'O' object; 'U/S' strings; 'M' datetime; etc.
        kind = getattr(dt, "kind", None)
        if kind in {"i", "u", "f", "c", "b"}:
            fams["other_numeric"].append(c)
        else:
            fams["other_categorical"].append(c)

    fams = {k: sorted(v) for k, v in fams.items() if v}
    return fams


@dataclass(frozen=True)
class BuildConfig:
    out_dir: str = "./exoplanet_dataset_phase1"
    request_timeout_s: int = 180
    max_rows_per_table: Optional[int] = None  # for debugging
    sleep_s_between_calls: float = 0.25
    parquet_engine: str = "pyarrow"
    parquet_compression: str = "zstd"


def build_phase1(cfg: BuildConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    manifest: Dict[str, Any] = {
        "config": asdict(cfg),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {},
        "global": {},
        "citations": {
            "koi_disposition_values": "KOI disposition values documented in KOI cumulative columns page.",
            "toi_tfopwg_disp_values": "TOI tfopwg_disp values documented in TOI columns page.",
            "k2_disposition_values": "K2 disposition values documented in k2pandc columns page.",
        },
    }

    frames: List[pd.DataFrame] = []

    for mission, table in MISSION_TABLES.items():
        print(f"[fetch] {mission} -> {table}")
        df = fetch_table_all(table, timeout_s=cfg.request_timeout_s, max_rows=cfg.max_rows_per_table)
        time.sleep(cfg.sleep_s_between_calls)

        df.columns = [c.strip() for c in df.columns]

        # Labels (locked by docs)
        df, label_report = add_labels(df, mission)

        # Scope filters (DONE-only supervised eligibility for KOI; default_flag=1 for K2)
        df, scope_report = apply_supervised_scope_filters(df, mission)

        # Add uid + mission
        df = add_row_uid(df, mission)

        # Role split + leakage filtering
        df, leakage_report = split_roles_and_filter_columns(df, mission)

        df, feature_cols, miss_report = drop_high_missingness_features(
            df,
            leakage_report["feature_cols"],
            threshold=0.98,
        )
        leakage_report["feature_cols"] = feature_cols

        # Coerce object-typed feature columns that are actually numeric
        df, num_coerce_report = coerce_object_columns_to_numeric(
            df,
            candidate_cols=leakage_report.get("feature_cols", []),
            min_convert_frac=0.98,
        )

        # Feature families
        feature_families = build_feature_families(leakage_report.get("feature_cols", []), df.dtypes.to_dict())

        # Canonical features (unit harmonized)
        df, canon_report = add_canonical_features(df, mission)

        # Record source stats
        manifest["sources"][mission] = {
            "table": table,
            "n_rows": int(df.shape[0]),
            "n_cols_after_filter": int(df.shape[1]),
            "labeling": label_report,
            "scope": scope_report,
            "leakage_filter": leakage_report,
            "feature_families": feature_families,
            "canonical": canon_report,
            "columns_after_filter": list(df.columns),
            "high_missingness_filter": miss_report,
            "numeric_coercion": num_coerce_report,
        }

        # Write per-mission parquet
        out_m = os.path.join(cfg.out_dir, f"raw_{mission}.parquet")
        df.to_parquet(out_m, engine=cfg.parquet_engine, index=False, compression=cfg.parquet_compression)
        manifest["sources"][mission]["parquet"] = out_m

        frames.append(df)

    # Merge (union schema)
    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    all_feature_cols = []
    for m in manifest["sources"]:
        all_feature_cols.extend(manifest["sources"][m]["leakage_filter"].get("feature_cols", []))
    all_feature_cols = sorted(set(all_feature_cols))
    manifest["global"]["feature_families"] = build_feature_families(all_feature_cols, merged.dtypes.to_dict())

    # Missingness stats (helps decide if we need simulated missingness later)
    miss = merged.isna().mean().sort_values(ascending=False)
    manifest["global"]["n_rows_total"] = int(merged.shape[0])
    manifest["global"]["n_cols_total"] = int(merged.shape[1])
    manifest["global"]["top_missingness"] = {k: float(v) for k, v in miss.head(200).to_dict().items()}

    # Write merged parquet + manifest
    out_parquet = os.path.join(cfg.out_dir, "phase1_merged.parquet")
    merged.to_parquet(out_parquet, engine=cfg.parquet_engine, index=False, compression=cfg.parquet_compression)

    out_manifest = os.path.join(cfg.out_dir, "phase1_manifest.json")
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("[done]")
    print(" merged:", out_parquet)
    print(" manifest:", out_manifest)


if __name__ == "__main__":
    cfg = BuildConfig(
        out_dir="./exoplanet_dataset_phase1",
        request_timeout_s=180,
        max_rows_per_table=None,  # set e.g. 100 for a quick smoke test
        sleep_s_between_calls=0.1,
        parquet_engine="pyarrow",
        parquet_compression="zstd",
    )
    build_phase1(cfg)
