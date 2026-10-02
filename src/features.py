"""Compute per-cell descriptive statistics and derived early-cycle variables."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.stats import kurtosis


BATCH_FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

BASE_NON_CAPACITY = [
    "dq_min",
    "dq_kurtosis",
    "IR_mean_100",
    "temperature_slope_100",
    "chargetime_mean_100",
    "max_c_rate",
]

FEATURE_SETS = {
    "base_initial_q": ["dq_min", "dq_kurtosis", "initial_QDischarge", *BASE_NON_CAPACITY[2:]],
    "base_mean_q": ["dq_min", "dq_kurtosis", "QDischarge_mean_100", *BASE_NON_CAPACITY[2:]],
    "no_dq_shape": ["dq_min", "initial_QDischarge", *BASE_NON_CAPACITY[2:]],
    "dq_std_substitution": ["dq_std", "dq_kurtosis", "initial_QDischarge", *BASE_NON_CAPACITY[2:]],
    "dq_mean_substitution": ["dq_mean", "dq_kurtosis", "initial_QDischarge", *BASE_NON_CAPACITY[2:]],
    "dq_range_substitution": ["dq_range", "dq_kurtosis", "initial_QDischarge", *BASE_NON_CAPACITY[2:]],
    "dq_only": ["dq_min", "dq_kurtosis"],
    "no_protocol": [
        "dq_min", "dq_kurtosis", "initial_QDischarge", "IR_mean_100",
        "temperature_slope_100", "chargetime_mean_100",
    ],
}


def _flat_numeric(obj: h5py.Dataset) -> np.ndarray:
    return np.asarray(obj[()]).astype(float, copy=False).reshape(-1)


def _deref(file: h5py.File, value: Any) -> h5py.Group | h5py.Dataset:
    while isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    return file[value] if isinstance(value, h5py.Reference) else value


def _decode_text(obj: h5py.Dataset) -> str:
    raw = np.asarray(obj[()]).reshape(-1)
    if raw.dtype.kind in "ui":
        return "".join(chr(int(x)) for x in raw if int(x))
    if raw.dtype.kind in "SU":
        return "".join(x.decode() if isinstance(x, bytes) else str(x) for x in raw)
    return str(raw[0]) if raw.size else ""


def _field(group: h5py.Group, name: str) -> h5py.Dataset:
    lookup = {key.lower(): key for key in group.keys()}
    if name.lower() not in lookup:
        raise KeyError(f"Missing field {name!r}; available={sorted(group.keys())}")
    return group[lookup[name.lower()]]


def _read_vector(file: h5py.File, dataset: h5py.Dataset) -> np.ndarray:
    if h5py.check_dtype(ref=dataset.dtype) is None:
        return _flat_numeric(dataset)
    refs = np.asarray(dataset[()]).reshape(-1)
    if refs.size == 1:
        return _flat_numeric(_deref(file, refs[0]))
    values: list[float] = []
    for ref in refs:
        values.extend(_flat_numeric(_deref(file, ref)).tolist())
    return np.asarray(values, dtype=float)


def _read_cycle_vector(file: h5py.File, dataset: h5py.Dataset, zero_based_index: int) -> np.ndarray:
    if h5py.check_dtype(ref=dataset.dtype) is not None:
        refs = np.asarray(dataset[()]).reshape(-1)
        if zero_based_index >= refs.size:
            return np.array([], dtype=float)
        return _flat_numeric(_deref(file, refs[zero_based_index]))
    values = np.asarray(dataset[()])
    if values.ndim != 2:
        return np.array([], dtype=float)
    if zero_based_index < values.shape[1]:
        return np.asarray(values[:, zero_based_index], dtype=float).reshape(-1)
    if zero_based_index < values.shape[0]:
        return np.asarray(values[zero_based_index, :], dtype=float).reshape(-1)
    return np.array([], dtype=float)


def _safe_life(obj: h5py.Dataset) -> float:
    values = _flat_numeric(obj)
    return float(values[0]) if values.size else np.nan


def _safe_slope(cycles: np.ndarray, values: np.ndarray) -> float:
    mask = np.isfinite(cycles) & np.isfinite(values) & (values != 0)
    if mask.sum() < 2 or np.unique(cycles[mask]).size < 2:
        return np.nan
    return float(np.polyfit(cycles[mask], values[mask], 1)[0])


def _at_cycle(cycles: np.ndarray, values: np.ndarray, cycle: int) -> float:
    match = np.flatnonzero(np.isclose(cycles, cycle))
    return float(values[match[0]]) if match.size else np.nan


def _extract_c_rates(policy: str) -> tuple[float, float]:
    rates = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)C", policy)]
    return (rates[0], max(rates)) if rates else (np.nan, np.nan)


def _delta_features(v: np.ndarray, q10: np.ndarray, q100: np.ndarray) -> dict[str, float]:
    n = min(len(v), len(q10), len(q100))
    if n < 4:
        return {name: np.nan for name in ["dq_mean", "dq_std", "dq_min", "dq_range", "dq_kurtosis"]}
    v, q10, q100 = v[:n], q10[:n], q100[:n]
    mask = np.isfinite(v) & np.isfinite(q10) & np.isfinite(q100)
    dq = q100[mask] - q10[mask]
    if dq.size < 4:
        return {name: np.nan for name in ["dq_mean", "dq_std", "dq_min", "dq_range", "dq_kurtosis"]}
    return {
        "dq_mean": float(np.mean(dq)),
        "dq_std": float(np.std(dq, ddof=1)),
        "dq_min": float(np.min(dq)),
        "dq_range": float(np.ptp(dq)),
        "dq_kurtosis": float(kurtosis(dq, fisher=True, bias=False)),
    }


def extract_feature_table(data_dir: Path, cache_path: Path) -> pd.DataFrame:
    """Summarize cycles 2--100 as one statistics-and-derived-variable row per cell."""
    rows: list[dict[str, Any]] = []
    for batch_name, filename in BATCH_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required raw file not found: {path}")
        batch_date = filename[:10]
        with h5py.File(path, "r") as file:
            batch = file["batch"]
            count = int(batch["summary"].shape[0])
            for index in range(count):
                def obj(field: str):
                    return _deref(file, batch[field][index, 0])

                life = _safe_life(obj("cycle_life"))
                policy = _decode_text(obj("policy_readable"))
                first_c, max_c = _extract_c_rates(policy)
                summary = obj("summary")
                cycles = _read_vector(file, _field(summary, "cycle"))
                qd = _read_vector(file, _field(summary, "QDischarge"))
                ir = _read_vector(file, _field(summary, "IR"))
                tavg = _read_vector(file, _field(summary, "Tavg"))
                chargetime = _read_vector(file, _field(summary, "chargetime"))
                n = min(map(len, [cycles, qd, ir, tavg, chargetime]))
                cycles, qd, ir, tavg, chargetime = (
                    values[:n] for values in [cycles, qd, ir, tavg, chargetime]
                )
                early = (cycles >= 2) & (cycles <= 100)
                ec, eq, ei, et, ect = cycles[early], qd[early], ir[early], tavg[early], chargetime[early]
                positive_q = eq[np.isfinite(eq) & (eq > 0)]
                q10, q100 = _at_cycle(ec, eq, 10), _at_cycle(ec, eq, 100)

                cycle_group = obj("cycles")
                qdlin = _field(cycle_group, "Qdlin")
                v = _read_vector(file, obj("Vdlin"))
                qd10 = _read_cycle_vector(file, qdlin, 9)
                qd100 = _read_cycle_vector(file, qdlin, 99)
                rows.append({
                    "cell_key": f"{batch_date}_cell_{index + 1}",
                    "cell_index": index,
                    "batch": batch_name,
                    "batch_date": batch_date,
                    "cycle_life": life,
                    "policy_readable": policy,
                    "first_stage_c_rate": first_c,
                    "max_c_rate": max_c,
                    "initial_QDischarge": float(positive_q[0]) if positive_q.size else np.nan,
                    "QDischarge_mean_100": float(np.nanmean(eq[eq > 0])) if np.any(eq > 0) else np.nan,
                    "QDischarge_cycle100_minus_cycle10": q100 - q10,
                    "IR_mean_100": float(np.nanmean(ei[ei > 0])) if np.any(ei > 0) else np.nan,
                    "temperature_slope_100": _safe_slope(ec, et),
                    "chargetime_mean_100": float(np.nanmean(ect[ect > 0])) if np.any(ect > 0) else np.nan,
                    **_delta_features(v, qd10, qd100),
                })

    frame = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


__all__ = ["BATCH_FILES", "FEATURE_SETS", "extract_feature_table"]
