"""Crea app/demo.db con datos sintéticos deterministas."""

import argparse

import _bootstrap  # noqa: F401
from sqlocal.config import DB_PATH
from sqlocal.db import build_demo_db, execute_sql

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = build_demo_db(DB_PATH, seed=args.seed)
    print(f"Base creada en {path}")
    for table in ("authors", "books", "users", "loans"):
        n = execute_sql(f"SELECT COUNT(*) FROM {table}").rows[0][0]
        print(f"  {table:<8} {n:>5} filas")
