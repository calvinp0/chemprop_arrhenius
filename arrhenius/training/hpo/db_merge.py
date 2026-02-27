import argparse
import os
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

from arrhenius.training.hpo.database_configs import SCHEMA  # reuse canonical schema

DEFAULT_TABLES = (
    "trials",
    "folds",
    "metrics_agg",
    "scalers",
    "splits",
    "trial_seeds",
    "final_summaries",
)


def _attach(conn: sqlite3.Connection, db_path: str, alias: str) -> None:
    path = db_path.replace("'", "''")
    conn.execute(f"ATTACH DATABASE '{path}' AS {alias}")


def _detach(conn: sqlite3.Connection, alias: str) -> None:
    conn.execute(f"DETACH DATABASE {alias}")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def _merge_single(conn: sqlite3.Connection, src_path: str, tables: Sequence[str]) -> None:
    alias = "src"
    _attach(conn, src_path, alias)
    try:
        for table in tables:
            conn.execute(f"INSERT OR REPLACE INTO {table} SELECT * FROM {alias}.{table}")
    finally:
        _detach(conn, alias)


def merge_databases(
    inputs: Iterable[str], output: str, tables: Sequence[str] = DEFAULT_TABLES
) -> None:
    inputs = list(inputs)
    if not inputs:
        raise ValueError("No input databases provided.")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    with sqlite3.connect(output, timeout=60.0) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=60000;")
        _ensure_schema(conn)
        for path in inputs:
            if not Path(path).is_file():
                print(f"[WARN] Skipping missing database: {path}")
                continue
            print(f"[MERGE] Incorporating {path}")
            _merge_single(conn, path, tables)
        conn.commit()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge multiple Chemprop HPO SQLite databases.")
    p.add_argument("--inputs", nargs="+", required=True, help="List of input database paths.")
    p.add_argument("--output", required=True, help="Destination database path.")
    p.add_argument(
        "--tables",
        nargs="+",
        choices=DEFAULT_TABLES,
        default=list(DEFAULT_TABLES),
        help="Optional subset of tables to merge (defaults to all).",
    )
    return p


def main():
    args = build_parser().parse_args()
    merge_databases(args.inputs, args.output, args.tables)
    print(f"[DONE] Merged {len(args.inputs)} database(s) into {args.output}")


if __name__ == "__main__":
    main()
