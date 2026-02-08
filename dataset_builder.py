"""
Exoplanet Candidate Dataset Builder (Phase 1)

This module builds a mission-agnostic, leakage-resistant, explainability-ready tabular dataset for
classifying exoplanet candidate signals into three classes:

    - positive   : confirmed planet
    - candidate  : planet candidate
    - negative   : false positive / false alarm / refuted

The dataset is constructed from the NASA Exoplanet Archive TAP service using three mission tables:

    - Kepler : cumulative (KOI cumulative table)
    - TESS   : toi        (TESS Objects of Interest)
    - K2     : k2pandc    (K2 Planets and Candidates)

Design principles
-----------------
1) Stable ground truth labeling:
   Labels are derived ONLY from the documented disposition fields for each table and then removed
   from the feature set to prevent leakage.

2) Leakage avoidance:
   - Remove disposition/vetting fields, pipeline scores/metrics, free text, URLs/links, and other
     pipeline-derived signals that could act as label proxies.
   - Remove discovery metadata and "how studied is it" proxies (disc_* and pl_n*/st_n* counts).
   - Remove multiplicity proxies (e.g., system planet counts) to reduce selection/prior artifacts.
   - Keep identifiers (names/IDs) strictly as metadata, not model features.

3) Missingness-aware schema:
   The merged dataset uses a union schema across missions. Missingness is expected and preserved.
   Features that are >= 98% missing within a mission are dropped deterministically at build time
   to avoid "dead tokens" in downstream models.

4) Uncertainty/limit fields:
   Include uncertainties/limits only for a curated set of physics-critical bases (period, epoch,
   duration, depth, radii, stellar Teff/logg/metallicity, distance). This provides measurement
   quality information without exploding the schema.

5) Canonical physics features:
   Add a compact set of canonical features shared across missions with unit-consistent transforms,
   while also retaining mission-native raw columns (after leakage filtering) for maximum signal.

Outputs
-------
The builder writes:
    - raw_kepler.parquet
    - raw_tess.parquet
    - raw_k2.parquet
    - phase1_merged.parquet
    - phase1_manifest.json   (complete provenance, schema decisions, and summary stats)

Usage
-----
Run as a script:
    python dataset_builder.py

Or import and call:
    build_phase1(BuildConfig(...))
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

# =============================================================================
# TAP service configuration
# =============================================================================

EXOARCHIVE_TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

MISSION_TABLES: Dict[str, str] = {
    "kepler": "cumulative",
    "tess": "toi",
    "k2": "k2pandc",
}

# =============================================================================
# Labeling policy (schema-locked)
# =============================================================================

LABEL_SOURCE_COL: Dict[str, str] = {
    "kepler": "koi_disposition",
    "tess": "tfopwg_disp",
    "k2": "disposition",
}

LABEL_COL = "target_class"  # {positive,candidate,negative} or NaN for unlabeled

LABEL_MAP_BY_MISSION: Dict[str, Dict[Optional[str], Optional[str]]] = {
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
    },
}

# =============================================================================
# Metadata + feature roles
# =============================================================================

MISSION_COL = "mission"
ROW_UID_COL = "row_uid"

# Columns that are always treated as metadata (retained in parquet; never used as model features)
META_COLS_COMMON = {MISSION_COL, ROW_UID_COL, LABEL_COL, "label_source", "label_source_value"}

# Deterministic (ordered) mission-specific metadata columns
META_COLS_BY_MISSION: Dict[str, List[str]] = {
    "kepler": ["kepid", "kepoi_name", "kepler_name"],
    "tess": ["tid", "toi", "toidisplay", "toipfx", "ctoi_alias"],
    "k2": [
        "pl_name",
        "pl_letter",
        "k2_name",
        "epic_hostname",
        "epic_candname",
        "hostname",
        "hd_name",
        "hip_name",
        "tic_id",
        "gaia_dr2_id",
        "gaia_dr3_id",
    ],
}

# Metadata-only catalog timestamp fields (kept for provenance/debugging; excluded from model features)
META_ONLY_EXACT = {
    "rowupdate",
    "release_date",
    "toi_created",
    "pl_pubdate",
    "releasedate",
    "disc_pubdate",
}

# =============================================================================
# Leakage prevention
# =============================================================================

DENY_EXACT = {
    # -------------------------------------------------------------------------
    # 1) Label sources / dispositions / vetting + close proxies
    # -------------------------------------------------------------------------
    "koi_disposition",
    "koi_pdisposition",
    "koi_vet_stat",
    "koi_vet_date",
    "koi_score",
    "tfopwg_disp",
    "disposition",
    "disp_refname",

    # -------------------------------------------------------------------------
    # 2) Free-text and links (never model features)
    # -------------------------------------------------------------------------
    "koi_comment",
    "koi_datalink_dvr",
    "koi_datalink_dvs",

    # -------------------------------------------------------------------------
    # 3) Used for filtering only; never a feature
    # -------------------------------------------------------------------------
    "default_flag",

    # -------------------------------------------------------------------------
    # 4) Multiplicity proxies (selection / prior artifacts)
    # -------------------------------------------------------------------------
    "pl_pnum",
    "koi_tce_plnt_num",
    "sy_pnum",
    "sy_snum",
    "sy_mnum",
    "koi_count",  # KOI-specific multiplicity proxy

    # -------------------------------------------------------------------------
    # 5) Kepler (KOI cumulative): analysis config / model-choice / coverage proxies
    # -------------------------------------------------------------------------
    "koi_fittype",
    "koi_limbdark_mod",
    "koi_trans_mod",
    "koi_quarters",
    "koi_num_transits",

    # Kepler: pipeline/statistics that are likely near-direct vetting proxies
    "koi_model_snr",
    "koi_model_dof",
    "koi_model_chisq",
    "koi_max_sngle_ev",
    "koi_max_mult_ev",
    "koi_bin_oedp_sig",

    # Kepler: derived LD coefficients (table/fitter-dependent; non-agnostic)
    "koi_ldm_coeff1",
    "koi_ldm_coeff2",
    "koi_ldm_coeff3",
    "koi_ldm_coeff4",

    # -------------------------------------------------------------------------
    # 6) K2 (k2pandc): provenance/coverage/status proxies
    # -------------------------------------------------------------------------
    "k2_campaigns",
    "k2_campaigns_num",
    "pl_tsystemref",
    "soltype",
    "pl_controv_flag",

    # K2: taxonomy strings (non-agnostic categorical descriptors)
    "st_spectype",
    "st_metratio",

    # -------------------------------------------------------------------------
    # 7) DROP bucket: detection / discovery channel flags (shortcut risk)
    # -------------------------------------------------------------------------
    "tran_flag",
    "rv_flag",
    "ttv_flag",
    "ptv_flag",
    "ast_flag",
    "ima_flag",
    "etv_flag",
    "micro_flag",
    "pul_flag",
    "cb_flag",
    "dkin_flag",
    "obm_flag",

    # -------------------------------------------------------------------------
    # 8) OPTIONAL bucket: sky position / tiling / indexing artifacts
    # -------------------------------------------------------------------------
    "ra",
    "dec",
    "glat",
    "glon",
    "elat",
    "elon",
    "x",
    "y",
    "z",
    "htm20",

    # OPTIONAL: proper motion / parallax / motion aggregates (contextual)
    "st_pmra",
    "st_pmdec",
    "sy_pm",
    "sy_pmra",
    "sy_pmdec",
    "sy_plx",

    # OPTIONAL: epoch-like / observation-window fields (cadence/scheduling shortcuts)
    "koi_time0",
    "koi_time0bk",
    "pl_tranmid",
}

DENY_REGEX = [
    # -------------------------------------------------------------------------
    # 1) Dispositions / vetting / FP flags
    # -------------------------------------------------------------------------
    r".*dispos.*",
    r".*vet.*",
    r".*fpflag.*",
    r".*robovet.*",
    r".*autovet.*",

    # -------------------------------------------------------------------------
    # 2) Pipeline scores/metrics/probabilities (conservative)
    # -------------------------------------------------------------------------
    r".*score.*",
    r".*rank.*",
    r".*prob.*",

    # -------------------------------------------------------------------------
    # 3) Comments/links/provenance/publication refs
    # -------------------------------------------------------------------------
    r".*comment.*",
    r".*datalink.*",
    r".*url.*",
    r".*prov.*",
    r".*refname.*",
    r".*bibcode.*",

    # -------------------------------------------------------------------------
    # 4) "How studied is it" proxies (follow-up counts)
    #    NOTE: guard against matching 'pl_name'
    # -------------------------------------------------------------------------
    r"^pl_n(?!ame).*",
    r"^st_n.*",

    # -------------------------------------------------------------------------
    # 5) Discovery metadata (selection-prior artifacts)
    # -------------------------------------------------------------------------
    r"^disc_.*",
    r"^discoverymethod$",

    # -------------------------------------------------------------------------
    # 6) Kepler pixel-vetting statistic families (mission/pipeline-specific)
    # -------------------------------------------------------------------------
    r"^koi_fwm_.*",
    r"^koi_dicco_.*",
    r"^koi_dikco_.*",

    # -------------------------------------------------------------------------
    # 7) Modeling/configuration meta (future-proof; redundant with DENY_EXACT but ok)
    # -------------------------------------------------------------------------
    r".*tsystemref.*",

    # -------------------------------------------------------------------------
    # 8) Any flag-style columns (future-proof; complements explicit flag names)
    # -------------------------------------------------------------------------
    r".*_flag$",
    r"^flag$",

    # -------------------------------------------------------------------------
    # 9) Coordinates / indexing families (future-proof)
    # -------------------------------------------------------------------------
    r"^htm\d+$",

    # -------------------------------------------------------------------------
    # 10) Transit epoch / mid-transit time family (future-proof)
    # -------------------------------------------------------------------------
    r"^pl_tranmid$",
    r"^koi_time0$",
    r"^koi_time0bk$",
]

_DENY_RE = [re.compile(p, flags=re.IGNORECASE) for p in DENY_REGEX]


def _should_drop(col: str) -> bool:
    """Return True if a column should be removed from the dataset (not merely from model features)."""
    c = col.strip()
    if c in DENY_EXACT:
        return True
    for r in _DENY_RE:
        if r.fullmatch(c) or r.match(c):
            return True
    return False


def _is_identity_like(col: str) -> bool:
    """
    Return True if the column represents a target identifier/name/alias.
    Identity-like fields are retained only if explicitly listed in META_COLS_BY_MISSION.
    """
    c = col.strip().lower()

    if c in {
        "kepid", "kepoi_name", "kepler_name",
        "tid", "toi", "toidisplay", "toipfx", "ctoi_alias",
        "pl_name", "pl_letter", "k2_name", "epic_hostname", "epic_candname",
        "hostname", "hd_name", "hip_name",
        "tic_id", "gaia_dr2_id", "gaia_dr3_id",
    }:
        return True

    if "alias" in c or "designation" in c:
        return True
    if c.endswith("_name") or c.endswith("name"):
        return True
    if "hostname" in c:
        return True

    if c.endswith("_id") or c == "id":
        return True

    return False


def _is_stringified_numeric(col: str) -> bool:
    """
    Return True for '*str' columns that duplicate numeric values (e.g., 'pl_trandepstr').
    These are excluded from model features to avoid redundant, inconsistent representations.
    """
    c = col.strip().lower()
    return c.endswith("str") or c.endswith("_str")


# =============================================================================
# Uncertainty / limit inclusion
# =============================================================================

UNC_BASE_ALLOWLIST = {
    # Orbital / transit
    "koi_period", "koi_duration", "koi_depth",
    "pl_orbper", "pl_trandurh", "pl_trandur", "pl_trandep",

    # Radii
    "koi_prad", "koi_srad",
    "pl_rade", "st_rad",

    # Stellar
    "koi_steff", "koi_slogg", "koi_smet",
    "st_teff", "st_logg", "st_met",

    # Distance
    "st_dist", "sy_dist",
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
    """Return True if an uncertainty/limit column is permitted by the base allowlist."""
    c = col.strip()
    for pat in UNC_PATTERNS:
        m = pat.match(c)
        if m:
            base = m.group("base")
            return base in UNC_BASE_ALLOWLIST
    return False


# =============================================================================
# Canonical unit harmonization (added features)
# =============================================================================

CANONICAL_MAP = {
    "period_days": [("kepler", "koi_period"), ("tess", "pl_orbper"), ("k2", "pl_orbper")],
    "duration_hours": [("kepler", "koi_duration"), ("tess", "pl_trandurh"), ("k2", "pl_trandur")],
    "depth_frac": [("kepler", "koi_depth"), ("tess", "pl_trandep"), ("k2", "pl_trandep")],
    "planet_radius_rearth": [("kepler", "koi_prad"), ("tess", "pl_rade"), ("k2", "pl_rade")],
    "star_radius_rsun": [("kepler", "koi_srad"), ("tess", "st_rad"), ("k2", "st_rad")],
    "star_teff_k": [("kepler", "koi_steff"), ("tess", "st_teff"), ("k2", "st_teff")],
    "star_logg_cgs": [("kepler", "koi_slogg"), ("tess", "st_logg"), ("k2", "st_logg")],
    "star_metallicity_dex": [("kepler", "koi_smet"), ("k2", "st_met")],
    "distance_pc": [("tess", "st_dist"), ("k2", "sy_dist")],
}


def _to_numeric(s: pd.Series) -> pd.Series:
    """Numeric coercion with NaN for non-parsable values."""
    return pd.to_numeric(s, errors="coerce")


def _depth_to_frac(mission: str, depth: pd.Series) -> pd.Series:
    """
    Convert transit depth to a unitless fraction:
      - Kepler/TESS depth is in ppm => fraction = ppm * 1e-6
      - K2 depth is in percent => fraction = % / 100
    """
    d = _to_numeric(depth)
    if mission in ("kepler", "tess"):
        return d * 1e-6
    if mission == "k2":
        return d / 100.0
    return d


# =============================================================================
# TAP helpers
# =============================================================================

def _one_line_sql(sql: str) -> str:
    """Exoplanet Archive TAP requires single-line queries; collapse all whitespace."""
    return re.sub(r"\s+", " ", sql).strip()


def tap_sync_query(sql: str, timeout_s: int = 120) -> pd.DataFrame:
    """Execute a TAP sync ADQL query and parse the JSON response into a DataFrame."""
    params = {"query": _one_line_sql(sql), "format": "json", "lang": "ADQL"}
    r = requests.get(EXOARCHIVE_TAP_SYNC, params=params, timeout=timeout_s)

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
    Resolve a user table name (e.g. 'cumulative') to (schema_name, table_name) as published in TAP.
    """
    sql = f"""
    SELECT schema_name, table_name
    FROM TAP_SCHEMA.tables
    WHERE LOWER(table_name) = LOWER('{table}')
    """
    df = tap_sync_query(sql, timeout_s=timeout_s)

    if df.empty:
        sql2 = f"""
        SELECT schema_name, table_name
        FROM TAP_SCHEMA.tables
        WHERE LOWER(table_name) LIKE LOWER('%{table}%')
        """
        df2 = tap_sync_query(sql2, timeout_s=timeout_s)
        if df2.empty:
            raise ValueError(
                f"Could not resolve table '{table}' in TAP_SCHEMA.tables. "
                "Inspect https://exoplanetarchive.ipac.caltech.edu/TAP/tables"
            )
        exact = df2[df2["table_name"].str.lower() == table.lower()]
        row = exact.iloc[0] if not exact.empty else df2.iloc[0]
        return str(row["schema_name"]), str(row["table_name"])

    row = df.iloc[0]
    return str(row["schema_name"]), str(row["table_name"])


def fetch_table_all(table: str, timeout_s: int, max_rows: int | None = None) -> pd.DataFrame:
    """Fetch all columns (and optionally only TOP N rows) from a TAP table."""
    schema_name, table_name = resolve_tap_table_id(table, timeout_s=timeout_s)
    fqtn = f"{schema_name}.{table_name}"
    top = f"TOP {int(max_rows)}" if max_rows is not None else ""
    sql = f"SELECT {top} * FROM {fqtn}"
    return tap_sync_query(sql, timeout_s=timeout_s)


# =============================================================================
# Build steps
# =============================================================================

def _normalize_label_string(x: Any) -> Optional[str]:
    """Normalize disposition strings to uppercase tokens and strip bracket annotations."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None

    s = str(x).strip()
    if not s:
        return None

    s = s.upper().replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*\[.*]\s*$", "", s).strip()  # handle 'REFUTED [PLANET]' variants
    return s


def add_labels(df: pd.DataFrame, mission: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Add:
      - label_source        : name of the source column used
      - label_source_value  : normalized string from that column
      - target_class        : mapped {positive,candidate,negative} or NaN
    """
    src_col = LABEL_SOURCE_COL[mission]
    if src_col not in df.columns:
        raise KeyError(f"Expected label source col '{src_col}' not found for mission={mission}")

    raw = df[src_col].map(_normalize_label_string)
    mapping = LABEL_MAP_BY_MISSION[mission]
    y = raw.map(lambda v: mapping.get(v, None))

    out = df.copy()
    out["label_source"] = src_col
    out["label_source_value"] = raw
    out[LABEL_COL] = y.astype("object")

    report = {
        "mission": mission,
        "label_source_col": src_col,
        "counts": {str(k): int(v) for k, v in y.value_counts(dropna=False).to_dict().items()},
    }
    return out, report


def apply_supervised_scope_filters(df: pd.DataFrame, mission: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply mission-specific scope constraints and add a supervised eligibility flag.

    - Kepler: supervised labels should be stable; require koi_vet_stat == 'DONE' for labeled rows.
              Unlabeled rows remain eligible for self-supervised uses.
    - K2    : keep only default_flag == 1 for a canonical parameter set.
    """
    out = df.copy()
    report: Dict[str, Any] = {"mission": mission, "filters": []}

    if mission == "kepler":
        if "koi_vet_stat" not in out.columns:
            raise KeyError("Kepler table missing koi_vet_stat; cannot enforce DONE-only supervised stability.")
        done = out["koi_vet_stat"].astype(str).str.upper().eq("DONE")
        labeled = out[LABEL_COL].notna()
        out["supervised_eligible"] = (~labeled) | done
        report["filters"].append({"supervised_eligible": "unlabeled OR koi_vet_stat == DONE"})
    else:
        out["supervised_eligible"] = True

    if mission == "k2":
        if "default_flag" not in out.columns:
            raise KeyError("K2 table missing default_flag; cannot select default parameter set.")
        before = len(out)
        out = out[out["default_flag"] == 1].copy()
        report["filters"].append({"default_flag": "==1", "rows_removed": before - len(out)})

    return out, report


def split_roles_and_filter_columns(df: pd.DataFrame, mission: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construct the mission dataset by:
      - retaining metadata columns
      - selecting feature columns while applying leakage and redundancy filters

    The returned DataFrame includes both metadata and model-safe features. The report includes the
    final feature list and drop reasons for auditing.
    """
    out = df.copy()

    # Determine metadata columns to keep
    meta_keep = set(META_COLS_COMMON)
    for c in META_COLS_BY_MISSION.get(mission, []):
        if c in out.columns:
            meta_keep.add(c)

    kept_cols: List[str] = []
    feature_cols: List[str] = []
    dropped: List[str] = []
    dropped_reason: Dict[str, str] = {}

    for c in out.columns:
        lc = c.strip().lower()

        # Always keep known metadata and supervised_eligible
        if c in meta_keep or c == "supervised_eligible":
            kept_cols.append(c)
            continue

        # Explicit metadata-only timestamp fields
        if lc in META_ONLY_EXACT:
            kept_cols.append(c)
            continue

        # Hard leakage removal
        if _should_drop(c):
            dropped.append(c)
            dropped_reason[c] = "denylist"
            continue

        # Remove redundant stringified numerics
        if _is_stringified_numeric(c):
            dropped.append(c)
            dropped_reason[c] = "stringified_numeric"
            continue

        # Remove identity-like columns unless explicitly included as metadata
        if _is_identity_like(c):
            dropped.append(c)
            dropped_reason[c] = "identity_like"
            continue

        # Uncertainty/limit handling
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

    for c in feature_cols:
        if _should_drop(c):
            raise RuntimeError(f"Leakage filter bug: denied column made it into features: {c}")

    filtered = out[kept_cols].copy()
    report = {
        "mission": mission,
        "n_in": int(out.shape[1]),
        "n_out": int(filtered.shape[1]),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "dropped_reason_counts": pd.Series(list(dropped_reason.values())).value_counts().to_dict(),
        "dropped_sample": dropped[:200],
    }
    return filtered, report


def drop_high_missingness_features(
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        threshold: float,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Remove features whose missingness rate is >= threshold within a mission.

    This prunes tokens that would be almost always masked and therefore waste capacity
    in downstream column-token models.
    """
    miss = df[list(feature_cols)].isna().mean()
    drop = miss[miss >= threshold].index.tolist()

    keep_cols = [c for c in feature_cols if c not in drop]
    out = df.drop(columns=drop)

    report = {"threshold": float(threshold), "dropped": drop, "n_dropped": int(len(drop))}
    return out, keep_cols, report


def add_row_uid(df: pd.DataFrame, mission: str) -> pd.DataFrame:
    """
    Add a deterministic, mission-scoped row UID derived from mission metadata identifiers.
    This UID is intended for stable row referencing and provenance, not for entity linking.
    """
    out = df.copy()

    id_cols = [c for c in META_COLS_BY_MISSION.get(mission, []) if c in out.columns]
    if id_cols:
        key_df = out[id_cols].copy()
        for c in id_cols:
            col = key_df[c]
            key_df[c] = col.where(~col.isna(), "").map(lambda v: "" if v is None else str(v))
        key = key_df.apply(lambda row: "|".join(row.values.tolist()), axis=1)
    else:
        key = out.index.map(str)

    def _hash(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    out[MISSION_COL] = mission
    out[ROW_UID_COL] = (mission + "|" + key).map(_hash)
    return out


def add_canonical_features(df: pd.DataFrame, mission: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Add a compact set of canonical features shared across missions, with unit harmonization.

    Raw columns are retained (after filtering) and canonicals are added as additional columns.
    """
    out = df.copy()
    created: Dict[str, str] = {}
    missing: List[str] = []

    for canon, sources in CANONICAL_MAP.items():
        src = next((col for (m, col) in sources if m == mission and col in out.columns), None)
        if src is None:
            missing.append(canon)
            continue

        if canon == "depth_frac":
            out[canon] = _depth_to_frac(mission, out[src])
        else:
            out[canon] = _to_numeric(out[src])

        created[canon] = src

    report = {"mission": mission, "created": created, "missing": missing}
    return out, report


def coerce_object_columns_to_numeric(
        df: pd.DataFrame,
        candidate_cols: Sequence[str],
        min_convert_frac: float = 0.98,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convert feature columns from object dtype to numeric when conversion is reliable.

    A column is converted if >= min_convert_frac of its non-null values can be parsed numerically.
    This prevents misclassification of numeric columns as categorical due to TAP JSON typing.
    """
    out = df.copy()
    converted: Dict[str, str] = {}
    skipped: List[str] = []

    for c in candidate_cols:
        if c not in out.columns:
            continue

        dt = out[c].dtype
        if not (dt == object or str(dt).lower() == "string"):
            continue

        s = out[c]
        non_null = int(s.notna().sum())
        if non_null == 0:
            skipped.append(c)
            continue

        num = pd.to_numeric(s, errors="coerce")
        ok = int(num.notna().sum())

        if ok / non_null >= min_convert_frac:
            out[c] = num
            converted[c] = f"object->numeric ({ok}/{non_null}={ok / non_null:.3f})"
        else:
            skipped.append(c)

    report = {"converted": converted, "skipped_count": int(len(skipped))}
    return out, report


def build_feature_families(columns: Sequence[str], dtypes: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Build a feature-family grouping for downstream explainability and reporting.

    The grouping is concept-driven (canonical physics, photometry, astrometry, flags, uncertainty/limits)
    and uses dtypes for a robust numeric/categorical fallback.
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

        # uncertainty/limit columns
        if any(p.match(c) for p in UNC_PATTERNS):
            fams["uncertainty_limit"].append(c)
            continue

        # canonical physics
        if lc in {"period_days", "duration_hours"}:
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

        # astrometry
        if lc == "ra" or lc == "dec" or lc.startswith("ra") or lc.startswith("dec"):
            fams["astrometry"].append(c)
            continue

        # photometry
        if ("mag" in lc) or lc.endswith("tmag") or lc.endswith("kepmag"):
            fams["photometry"].append(c)
            continue

        # flags
        if lc.endswith("_flag") or lc.endswith("flag"):
            fams["flags"].append(c)
            continue

        # timestamp-like strings (should typically be metadata-only; included for completeness)
        if any(tok in lc for tok in ["rowupdate", "release", "created", "pubdate", "releasedate", "disc_pubdate"]):
            fams["catalog_timestamps"].append(c)
            continue

        # dtype-based fallback
        dt = dtypes.get(c)
        kind = getattr(dt, "kind", None)
        if kind in {"i", "u", "f", "c", "b"}:
            fams["other_numeric"].append(c)
        else:
            fams["other_categorical"].append(c)

    return {k: sorted(v) for k, v in fams.items() if v}


# =============================================================================
# Builder orchestration
# =============================================================================

@dataclass(frozen=True)
class BuildConfig:
    out_dir: str = "./exoplanet_dataset_phase1"
    request_timeout_s: int = 180
    max_rows_per_table: Optional[int] = None
    sleep_s_between_calls: float = 0.25
    parquet_engine: str = "pyarrow"
    parquet_compression: str = "zstd"
    high_missingness_threshold: float = 0.98
    numeric_coerce_min_frac: float = 0.98


def build_phase1(cfg: BuildConfig) -> None:
    """
    Build and persist Phase 1 datasets and a manifest containing all provenance and summary statistics.

    The mission pipelines are run independently and then concatenated to form a merged union-schema dataset.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)

    manifest: Dict[str, Any] = {
        "config": asdict(cfg),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {},
        "global": {},
        "notes": {
            "labels": "target_class derived from mission disposition fields, then disposition fields removed from features",
            "missingness": f"features with missingness >= {cfg.high_missingness_threshold} dropped per mission",
            "uncertainty": "uncertainties/limits included only for curated physics-critical bases",
        },
    }

    frames: List[pd.DataFrame] = []

    for mission, table in MISSION_TABLES.items():
        print(f"[fetch] {mission} -> {table}")

        df = fetch_table_all(table, timeout_s=cfg.request_timeout_s, max_rows=cfg.max_rows_per_table)
        time.sleep(cfg.sleep_s_between_calls)
        df.columns = [c.strip() for c in df.columns]

        df, label_report = add_labels(df, mission)
        df, scope_report = apply_supervised_scope_filters(df, mission)
        df = add_row_uid(df, mission)

        df, leakage_report = split_roles_and_filter_columns(df, mission)

        df, kept_feature_cols, miss_report = drop_high_missingness_features(
            df,
            leakage_report["feature_cols"],
            threshold=cfg.high_missingness_threshold,
        )
        leakage_report["feature_cols"] = kept_feature_cols

        df, num_coerce_report = coerce_object_columns_to_numeric(
            df,
            candidate_cols=leakage_report["feature_cols"],
            min_convert_frac=cfg.numeric_coerce_min_frac,
        )

        feature_families = build_feature_families(leakage_report["feature_cols"], df.dtypes.to_dict())
        df, canon_report = add_canonical_features(df, mission)

        # Persist mission parquet
        out_m = os.path.join(cfg.out_dir, f"raw_{mission}.parquet")
        df.to_parquet(out_m, engine=cfg.parquet_engine, index=False, compression=cfg.parquet_compression)

        manifest["sources"][mission] = {
            "table": table,
            "parquet": out_m,
            "n_rows": int(df.shape[0]),
            "n_cols_after_filter": int(df.shape[1]),
            "labeling": label_report,
            "scope": scope_report,
            "leakage_filter": leakage_report,
            "high_missingness_filter": miss_report,
            "numeric_coercion": num_coerce_report,
            "feature_families": feature_families,
            "canonical": canon_report,
            "columns_after_filter": list(df.columns),
        }

        frames.append(df)

    # Merge datasets (union schema)
    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # Global feature families from the union of per-mission feature lists
    all_feature_cols: List[str] = []
    for m in manifest["sources"]:
        all_feature_cols.extend(manifest["sources"][m]["leakage_filter"]["feature_cols"])
    all_feature_cols = sorted(set(all_feature_cols))
    manifest["global"]["feature_families"] = build_feature_families(all_feature_cols, merged.dtypes.to_dict())

    # Missingness summary for diagnostics
    miss = merged.isna().mean().sort_values(ascending=False)
    manifest["global"]["n_rows_total"] = int(merged.shape[0])
    manifest["global"]["n_cols_total"] = int(merged.shape[1])
    manifest["global"]["top_missingness"] = {k: float(v) for k, v in miss.head(200).to_dict().items()}

    # Persist merged parquet + manifest
    out_parquet = os.path.join(cfg.out_dir, "phase1_merged.parquet")
    merged.to_parquet(out_parquet, engine=cfg.parquet_engine, index=False, compression=cfg.parquet_compression)

    out_manifest = os.path.join(cfg.out_dir, "phase1_manifest.json")
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("[done]")
    print(" merged:", out_parquet)
    print(" manifest:", out_manifest)


# =============================================================================
# Script entrypoint
# =============================================================================

if __name__ == "__main__":
    cfg = BuildConfig(
        out_dir="./exoplanet_dataset_phase1",
        request_timeout_s=180,
        max_rows_per_table=None,
        sleep_s_between_calls=0.05,
        parquet_engine="pyarrow",
        parquet_compression="zstd",
        high_missingness_threshold=0.98,
        numeric_coerce_min_frac=0.98,
    )
    build_phase1(cfg)
