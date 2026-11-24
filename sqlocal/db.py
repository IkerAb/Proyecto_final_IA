"""Base de datos SQLite de demostración: creación, ejecución segura y comparación de resultados."""

from __future__ import annotations

import random
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .config import DB_PATH, MAX_ROWS, SCHEMA_PATH, SQL_TIMEOUT_S

# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Alice", "Bruno", "Carla", "Daniel", "Elena", "Felix", "Grace", "Hugo", "Irene", "James",
    "Kara", "Liam", "Maya", "Noah", "Olivia", "Peter", "Quinn", "Rosa", "Samuel", "Tara",
    "Umar", "Vera", "Walter", "Ximena", "Yusuf", "Zoe", "Adrian", "Bianca", "Caleb", "Diana",
]
LAST_NAMES = [
    "Anderson", "Brooks", "Carter", "Dawson", "Ellis", "Fisher", "Gardner", "Hayes", "Iverson",
    "Jensen", "Keller", "Lawson", "Morgan", "Nolan", "Owens", "Parker", "Quincy", "Reed",
    "Sawyer", "Turner", "Underwood", "Vaughn", "Walsh", "Young", "Zimmerman", "Bishop", "Cole",
]
COUNTRIES = ["USA", "UK", "Canada", "Ireland", "Australia", "India", "Nigeria", "France", "Spain", "Japan"]
GENRES = [
    "Fiction", "Mystery", "Science Fiction", "Fantasy", "History",
    "Biography", "Poetry", "Science", "Romance", "Horror",
]
CITIES = ["Boston", "Chicago", "Denver", "Seattle", "Austin", "Portland", "Miami", "Atlanta"]
MEMBERSHIPS = ["basic", "premium", "student"]
TITLE_ADJ = [
    "Silent", "Hidden", "Broken", "Golden", "Distant", "Burning", "Frozen", "Lost", "Secret",
    "Wild", "Crimson", "Endless", "Quiet", "Forgotten", "Shining", "Restless", "Hollow", "Bright",
]
TITLE_NOUN = [
    "River", "Garden", "Empire", "Shadow", "Harbor", "Mountain", "Letter", "Kingdom", "Forest",
    "Machine", "Island", "Promise", "Winter", "City", "Voyage", "Mirror", "Storm", "Orchard",
]

N_AUTHORS = 60
N_BOOKS = 300
N_USERS = 150
N_LOANS = 2000
LOAN_START = date(2023, 1, 1)
LOAN_END = date(2024, 12, 31)


def build_demo_db(path: Path = DB_PATH, seed: int = 42) -> Path:
    """Crea demo.db con datos sintéticos deterministas.

    Algunas columnas (pages, birth_year, joined_date) tienen valores únicos para que las
    consultas con ORDER BY ... LIMIT no tengan empates y su resultado sea inequívoco.
    """
    rng = random.Random(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    full_names = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    rng.shuffle(full_names)
    author_names = full_names[:N_AUTHORS]
    user_names = full_names[N_AUTHORS : N_AUTHORS + N_USERS]

    titles = [f"The {a} {n}" for a in TITLE_ADJ for n in TITLE_NOUN]
    rng.shuffle(titles)
    titles = titles[:N_BOOKS]

    birth_years = rng.sample(range(1900, 2001), N_AUTHORS)
    pages_values = rng.sample(range(80, 1200), N_BOOKS)
    join_days = rng.sample(range(0, (date(2024, 12, 31) - date(2018, 1, 1)).days), N_USERS)

    con = sqlite3.connect(path)
    con.executescript(SCHEMA_PATH.read_text())

    con.executemany(
        "INSERT INTO authors VALUES (?, ?, ?, ?)",
        [(i + 1, author_names[i], rng.choice(COUNTRIES), birth_years[i]) for i in range(N_AUTHORS)],
    )
    con.executemany(
        "INSERT INTO books VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                i + 1,
                titles[i],
                rng.randint(1, N_AUTHORS),
                rng.choice(GENRES),
                rng.randint(1950, 2023),
                pages_values[i],
            )
            for i in range(N_BOOKS)
        ],
    )
    con.executemany(
        "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
        [
            (
                i + 1,
                user_names[i],
                rng.choice(CITIES),
                rng.choices(MEMBERSHIPS, weights=[5, 3, 2])[0],
                (date(2018, 1, 1) + timedelta(days=join_days[i])).isoformat(),
            )
            for i in range(N_USERS)
        ],
    )
    span = (LOAN_END - LOAN_START).days
    loans = []
    for i in range(N_LOANS):
        loan_date = LOAN_START + timedelta(days=rng.randint(0, span))
        returned = rng.random() > 0.12
        return_date = (loan_date + timedelta(days=rng.randint(3, 40))).isoformat() if returned else None
        loans.append((i + 1, rng.randint(1, N_BOOKS), rng.randint(1, N_USERS), loan_date.isoformat(), return_date))
    con.executemany("INSERT INTO loans VALUES (?, ?, ?, ?, ?)", loans)
    con.commit()
    con.close()
    return path


# ---------------------------------------------------------------------------
# Ejecución segura
# ---------------------------------------------------------------------------

_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


@dataclass
class ExecResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0


def execute_sql(sql: str, db_path: Path = DB_PATH) -> ExecResult:
    """Ejecuta una consulta en modo solo lectura, con tiempo límite y tope de filas."""
    sql = (sql or "").strip().rstrip(";").strip()
    if not sql:
        return ExecResult(ok=False, error="Consulta vacía")
    if not _ALLOWED_START.match(sql):
        return ExecResult(ok=False, error="Solo se permiten consultas SELECT")
    if ";" in sql:
        return ExecResult(ok=False, error="Solo se permite una sentencia")

    con = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    deadline = time.monotonic() + SQL_TIMEOUT_S
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    start = time.perf_counter()
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(MAX_ROWS)
        cols = [d[0] for d in cur.description or []]
        return ExecResult(ok=True, columns=cols, rows=rows, elapsed_ms=(time.perf_counter() - start) * 1000)
    except sqlite3.Error as exc:
        msg = "Tiempo límite excedido" if "interrupted" in str(exc) else str(exc)
        return ExecResult(ok=False, error=msg, elapsed_ms=(time.perf_counter() - start) * 1000)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Comparación (Execution Accuracy)
# ---------------------------------------------------------------------------


def _norm_value(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, str):
        return v.strip()
    return v


def _norm_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_norm_value(v) for v in r) for r in rows]


def _sorted_cells(row: tuple) -> tuple:
    return tuple(sorted(row, key=lambda x: (type(x).__name__, str(x))))


def results_match(pred: ExecResult, gold: ExecResult, ordered: bool) -> bool:
    """¿Devuelven las dos consultas el mismo resultado?

    - Si la consulta de referencia tiene ORDER BY, el orden de las filas importa.
    - Si no, se comparan como multiconjuntos (el orden da igual).
    - Se tolera un orden distinto de columnas (p. ej. `COUNT(*), genre` frente a `genre, COUNT(*)`).
    - Los nombres de columna (alias) no cuentan.
    """
    if not (pred.ok and gold.ok):
        return False
    p, g = _norm_rows(pred.rows), _norm_rows(gold.rows)
    if len(p) != len(g):
        return False
    if ordered:
        return p == g or [_sorted_cells(r) for r in p] == [_sorted_cells(r) for r in g]
    return Counter(p) == Counter(g) or Counter(map(_sorted_cells, p)) == Counter(map(_sorted_cells, g))


def is_ordered(sql: str) -> bool:
    return bool(re.search(r"\bORDER\s+BY\b", sql, re.IGNORECASE))
