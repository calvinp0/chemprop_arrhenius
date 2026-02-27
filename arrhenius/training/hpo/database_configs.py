# run_hpo/database_configs.py

from typing import Optional
import hashlib
import json
import math
import sqlite3
import time
import zlib
from contextlib import contextmanager


def _json(x):
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False)


def _hash_cfg(cfg: dict, split_sig: str) -> str:
    s = _json(cfg) + "|" + str(split_sig)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@contextmanager
def _conn(db_path):
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=60000;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS trials(
  trial_id      INTEGER,
  study_name    TEXT,
  cfg_json      TEXT,
  cfg_hash      TEXT,
  split_sig     TEXT,
  status        TEXT,
  started_ts    REAL,
  ended_ts      REAL,
  PRIMARY KEY(trial_id, study_name)
);

CREATE TABLE IF NOT EXISTS folds(
  trial_id   INTEGER,
  study_name TEXT,
  fold_id    INTEGER,
  metric     TEXT,
  value      REAL,
  PRIMARY KEY(trial_id, study_name, fold_id, metric)
);

CREATE TABLE IF NOT EXISTS metrics_agg(
  trial_id   INTEGER,
  study_name TEXT,
  metric     TEXT,
  mean       REAL,
  std        REAL,
  min        REAL,
  max        REAL,
  count      INTEGER,
  PRIMARY KEY(trial_id, study_name, metric)
);

CREATE TABLE IF NOT EXISTS scalers(
  trial_id    INTEGER,
  study_name  TEXT,
  fold_id     INTEGER,
  which       TEXT,   -- e.g. 'y', 'X_global'
  component   TEXT,   -- e.g. 't0','t1', or 'StandardScaler'
  param       TEXT,   -- 'mean','scale','var','lambdas', etc.
  summary     TEXT,   -- JSON with {mean, std, min, max, n}
  raw         BLOB,   -- optional compressed JSON (when --save-raw-scalers)
  PRIMARY KEY(trial_id, study_name, fold_id, which, component, param)
);

CREATE TABLE IF NOT EXISTS splits(
  trial_id   INTEGER,
  study_name TEXT,
  fold_id    INTEGER,
  split_name TEXT,     -- 'train','val','test','dev'
  idx_json   TEXT,     -- JSON array of indices
  PRIMARY KEY(trial_id, study_name, fold_id, split_name)
);

CREATE TABLE IF NOT EXISTS trial_seeds (
    trial_id INTEGER,
    fold_id INTEGER,
    seed_json TEXT,
    PRIMARY KEY (trial_id, fold_id)
);

CREATE TABLE IF NOT EXISTS final_summaries (
  study_name TEXT,
  tag        TEXT,
  payload    TEXT,
  created_ts REAL,
  PRIMARY KEY(study_name, tag)
);

CREATE TABLE IF NOT EXISTS predictions (
  trial_id    INTEGER,
  study_name  TEXT,
  fold_id     INTEGER,
  replicate   INTEGER,
  split       TEXT,
  sample_index INTEGER,
  payload     TEXT,
  PRIMARY KEY(trial_id, study_name, fold_id, replicate, split, sample_index)
);
"""


def _summarize_vec(v):
    if v is None:
        return None
    try:
        n = len(v)
        if n == 0:
            return {"n": 0}
        s = float(sum(v))
        m = s / n
        var = sum((x - m) * (x - m) for x in v) / n
        return {
            "n": n,
            "mean": m,
            "std": math.sqrt(var),
            "min": float(min(v)),
            "max": float(max(v)),
        }
    except Exception:
        return None


def _maybe_vec(x):
    # return Python list[float] if attribute exists, else None
    if x is None:
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(map(float, x))
    return None


def _extract_scaler_params(scaler, which: str):
    """
    Returns list of (component, param_name, summary_dict, raw_json_bytes_or_None).
    Handles sklearn ColumnTransformer (y) and StandardScaler/PowerTransformer (X).
    """
    rows = []
    if scaler is None:
        return rows

    def pack(component, name, vec):
        v = _maybe_vec(vec)
        summ = _summarize_vec(v)
        raw = None if v is None else zlib.compress(_json(v).encode("utf-8"))
        rows.append((component, name, summ, raw))

    # sklearn ColumnTransformer (for y)
    if hasattr(scaler, "named_transformers_"):
        for comp, tr in scaler.named_transformers_.items():
            if hasattr(tr, "mean_"):
                pack(comp, "mean", tr.mean_)
            if hasattr(tr, "scale_"):
                pack(comp, "scale", tr.scale_)
            if hasattr(tr, "var_"):
                pack(comp, "var", tr.var_)
            if hasattr(tr, "lambdas_"):
                pack(comp, "lambdas", tr.lambdas_)
            # PowerTransformer stores lambdas_; QuantileTransformer has quantiles_; RobustScaler has center_/scale_
            if hasattr(tr, "center_"):
                pack(comp, "center", tr.center_)
            if hasattr(tr, "quantiles_"):
                pack(comp, "quantiles", tr.quantiles_)
    else:
        comp = getattr(scaler, "__class__", type("X", (), {})).__name__
        if hasattr(scaler, "mean_"):
            pack(comp, "mean", scaler.mean_)
        if hasattr(scaler, "scale_"):
            pack(comp, "scale", scaler.scale_)
        if hasattr(scaler, "var_"):
            pack(comp, "var", scaler.var_)
        if hasattr(scaler, "lambdas_"):
            pack(comp, "lambdas", scaler.lambdas_)
        if hasattr(scaler, "center_"):
            pack(comp, "center", scaler.center_)
        if hasattr(scaler, "quantiles_"):
            pack(comp, "quantiles", scaler.quantiles_)
    return rows


class TrialLogger:
    def __init__(self, db_path: str, study_name: str, save_raw_scalers: bool = False):
        self.db, self.study_name, self.save_raw = db_path, study_name, save_raw_scalers
        with _conn(self.db) as c:
            c.executescript(SCHEMA)

    def start_trial(self, trial_id: int, cfg: dict, split_sig: str):
        with _conn(self.db) as c:
            c.execute(
                """INSERT OR REPLACE INTO trials
              (trial_id, study_name, cfg_json, cfg_hash, split_sig, status, started_ts, ended_ts)
              VALUES (?,?,?,?,?,?,?, COALESCE((SELECT ended_ts FROM trials WHERE trial_id=? AND study_name=?), NULL))""",
                (
                    trial_id,
                    self.study_name,
                    _json(cfg),
                    _hash_cfg(cfg, split_sig),
                    split_sig,
                    "running",
                    time.time(),
                    trial_id,
                    self.study_name,
                ),
            )

    def end_trial(self, trial_id: int, status: str):
        with _conn(self.db) as c:
            c.execute(
                "UPDATE trials SET status=?, ended_ts=? WHERE trial_id=? AND study_name=?",
                (status, time.time(), trial_id, self.study_name),
            )

    def log_fold_metrics(self, trial_id: int, fold_id: int, metrics: dict):
        rows = [(trial_id, self.study_name, fold_id, k, float(v)) for k, v in metrics.items()]
        with _conn(self.db) as c:
            c.executemany(
                "INSERT OR REPLACE INTO folds(trial_id,study_name,fold_id,metric,value) VALUES(?,?,?,?,?)",
                rows,
            )

    def log_seed(self, trial_id: int, fold_id: Optional[int], seed_dict: dict):
        import json

        with _conn(self.db) as c:
            c.execute(
                "INSERT OR REPLACE INTO trial_seeds(trial_id, fold_id, seed_json) VALUES (?,?,?)",
                (trial_id, fold_id, json.dumps(seed_dict)),
            )

    def log_scalers(self, trial_id: int, fold_id: int, which: str, scaler):
        store_raw = self.save_raw or which == "y"
        for comp, param, summ, raw in _extract_scaler_params(scaler, which):
            with _conn(self.db) as c:
                c.execute(
                    """INSERT OR REPLACE INTO scalers
                  (trial_id,study_name,fold_id,which,component,param,summary,raw)
                  VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        trial_id,
                        self.study_name,
                        fold_id,
                        which,
                        comp,
                        param,
                        _json(summ) if summ else None,
                        raw if (store_raw and raw is not None) else None,
                    ),
                )

    def log_split_indices(self, trial_id: int, fold_id: int, **splits_arrays):
        rows = []
        for name, arr in splits_arrays.items():
            if arr is None:
                continue
            if hasattr(arr, "tolist"):
                arr = arr.tolist()
            rows.append((trial_id, self.study_name, fold_id, name, _json(list(map(int, arr)))))
        if rows:
            with _conn(self.db) as c:
                c.executemany(
                    "INSERT OR REPLACE INTO splits(trial_id,study_name,fold_id,split_name,idx_json) VALUES(?,?,?,?,?)",
                    rows,
                )

    def log_predictions(
        self,
        trial_id: int,
        fold_id: int,
        replicate: int,
        split_name: str,
        sample_indices,
        payloads: dict,
    ):
        """
        Store per-sample predictions (scaled and unscaled) for a split.
        payloads keys mirror _stack_prediction_outputs: y_pred_raw, y_true_raw, y_pred_s, y_true_s, lnk_pred, lnk_true, temps.
        """
        temps = payloads.get("temps")

        def _row(arr, pos):
            if arr is None or pos >= len(arr):
                return None
            v = arr[pos]
            return v.tolist() if hasattr(v, "tolist") else v

        rows = []
        for pos, idx in enumerate(sample_indices):
            entry = {
                "y_pred_raw": _row(payloads.get("y_pred_raw"), pos),
                "y_true_raw": _row(payloads.get("y_true_raw"), pos),
                "y_pred_s": _row(payloads.get("y_pred_s"), pos),
                "y_true_s": _row(payloads.get("y_true_s"), pos),
                "lnk_pred": _row(payloads.get("lnk_pred"), pos),
                "lnk_true": _row(payloads.get("lnk_true"), pos),
                "temps": temps,
            }
            rows.append(
                (
                    trial_id,
                    self.study_name,
                    fold_id,
                    int(replicate),
                    split_name,
                    int(idx),
                    _json(entry),
                )
            )

        if rows:
            with _conn(self.db) as c:
                c.executemany(
                    "INSERT OR REPLACE INTO predictions(trial_id,study_name,fold_id,replicate,split,sample_index,payload) "
                    "VALUES (?,?,?,?,?,?,?)",
                    rows,
                )

    def aggregate_and_store(self, trial_id: int):
        # compute mean/std/min/max/count per metric over folds and store into metrics_agg
        with _conn(self.db) as c:
            cur = c.execute(
                """SELECT metric, value FROM folds WHERE trial_id=? AND study_name=? ORDER BY metric""",
                (trial_id, self.study_name),
            )
            by_metric = {}
            for m, v in cur.fetchall():
                by_metric.setdefault(m, []).append(float(v))
            rows = []
            for m, vals in by_metric.items():
                n = len(vals)
                mu = sum(vals) / n
                var = sum((x - mu) * (x - mu) for x in vals) / n
                rows.append(
                    (trial_id, self.study_name, m, mu, math.sqrt(var), min(vals), max(vals), n)
                )
            c.executemany(
                """INSERT OR REPLACE INTO metrics_agg
              (trial_id,study_name,metric,mean,std,min,max,count) VALUES (?,?,?,?,?,?,?,?)""",
                rows,
            )

    def store_summary(self, tag: str, payload: dict):
        with _conn(self.db) as c:
            c.execute(
                "INSERT OR REPLACE INTO final_summaries(study_name, tag, payload, created_ts) VALUES (?,?,?,?)",
                (self.study_name, tag, _json(payload), time.time()),
            )


def fetch_split_indices(
    db_path: str, study_name: str, trial_id: int, fold_id: Optional[int] = None
) -> dict:
    """
    Return stored split indices for a given trial and optional fold.

    When `fold_id` is None the result is a mapping:
        {fold_id: {split_name: [indices]}}
    Otherwise, the mapping for the specified fold is returned directly.
    """
    with _conn(db_path) as c:
        params: tuple = (trial_id, study_name)
        query = "SELECT fold_id, split_name, idx_json FROM splits WHERE trial_id=? AND study_name=?"
        if fold_id is not None:
            query += " AND fold_id=?"
            params += (fold_id,)
        rows = c.execute(query, params).fetchall()

    result: dict[int, dict[str, list[int]]] = {}
    for f_id, name, idx_json in rows:
        result.setdefault(int(f_id), {})[str(name)] = json.loads(idx_json)

    if fold_id is not None:
        return result.get(int(fold_id), {})
    return result


def fetch_trial_seeds(
    db_path: str, study_name: str, trial_id: int, fold_id: Optional[int] = None
) -> dict:
    """
    Fetch stored seed metadata for the given trial.
    Returns {fold_id: seed_dict}.
    """
    with _conn(db_path) as c:
        params: tuple = (trial_id,)
        query = "SELECT fold_id, seed_json FROM trial_seeds WHERE trial_id=?"
        if fold_id is not None:
            query += " AND fold_id=?"
            params += (fold_id,)
        rows = c.execute(query, params).fetchall()

    result: dict[int, dict[str, int]] = {}
    for f_id, payload in rows:
        result[int(f_id)] = json.loads(payload)
    if fold_id is not None:
        return result.get(int(fold_id), {})
    return result
