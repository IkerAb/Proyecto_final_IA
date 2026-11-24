"""Genera el dataset pregunta–SQL (en inglés) y lo valida por ejecución contra demo.db.

Cada par se conserva solo si la SQL de referencia se ejecuta sin error y devuelve un
resultado no vacío (y, en agregaciones, distinto de 0/NULL), para que acertar por
casualidad con una consulta vacía no cuente como acierto.

División train/val/test (--split-mode):
  template    (por defecto) las plantillas de test NO aparecen en train: mide si el
              modelo generaliza a estructuras de consulta nuevas.
  paraphrase  todas las plantillas en train, pero las formulaciones de test son nuevas.
  random      división aleatoria por ejemplo (tiende a inflar la métrica).
"""

from __future__ import annotations

import argparse
import calendar
import json
import random
from datetime import date, timedelta
from collections import Counter, defaultdict

import _bootstrap  # noqa: F401
from sqlocal.config import DATA_DIR, DB_PATH
from sqlocal.db import build_demo_db, execute_sql
from sqlocal.templates import TEMPLATES, Template


def load_values() -> dict[str, list]:
    def col(sql):
        return [r[0] for r in execute_sql(sql).rows]

    return {
        "genre": col("SELECT DISTINCT genre FROM books"),
        "country": col("SELECT DISTINCT country FROM authors"),
        "city": col("SELECT DISTINCT city FROM users"),
        "membership": col("SELECT DISTINCT membership FROM users"),
        "title": col("SELECT title FROM books"),
        "author": col("SELECT name FROM authors"),
        "user": col("SELECT name FROM users"),
        "join_year": [str(y) for y in range(2018, 2025)],
        "loan_year": ["2023", "2024"],
        "year": list(range(1960, 2021, 5)),
        "birth_year": list(range(1920, 1991, 10)),
        "pages": [300, 400, 500, 600, 800, 1000],
        "n": [3, 5, 10],
        "k": [25, 28, 30, 32, 35],
        "month": [f"{m:02d}" for m in range(1, 13)],
        # Umbrales para HAVING, elegidos según la distribución real de demo.db
        "k_author": [6, 7, 8, 10],
        "k_user": [15, 17, 18, 20],
        "k_city": [3, 4, 5, 8, 10],
        "avg_pages": [560, 600, 620, 640, 660],
        "k_book": [10, 11, 12, 13],
        "k_country": [20, 25, 30, 35],
        "days": [15, 20, 25, 30, 35],
        "quarter": [1, 2, 3, 4],
    }


QUARTERS = {1: ("01-01", "03-31"), 2: ("04-01", "06-30"), 3: ("07-01", "09-30"), 4: ("10-01", "12-31")}
QUARTER_NAMES = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def _date_text(d: date) -> str:
    return f"{calendar.month_name[d.month]} {d.day}, {d.year}"


def sample_params(tpl: Template, values: dict, rng: random.Random) -> dict:
    derived = {"year2", "city2", "genre2", "d1"}
    params = {}
    for name in tpl.params:
        if name not in derived:
            params[name] = rng.choice(values[name])
    if "year2" in tpl.params:
        params["year2"] = params["year"] + rng.choice([10, 15, 20, 30])
    for name, base in (("city2", "city"), ("genre2", "genre")):
        if name in tpl.params:  # un segundo valor distinto del primero
            params[name] = rng.choice([v for v in values[base] if v != params[base]])
    if "d1" in tpl.params:  # rango de fechas dentro del periodo de préstamos
        start = date(2023, 1, 1) + timedelta(days=rng.randint(0, 600))
        end = start + timedelta(days=rng.choice([7, 14, 30, 45, 60, 90]))
        params.update(d1=start.isoformat(), d2=end.isoformat(), d1_text=_date_text(start), d2_text=_date_text(end))
    if "month" in params:
        params["month_name"] = calendar.month_name[int(params["month"])]
    if "quarter" in params:
        q = params["quarter"]
        params.update(q_start=QUARTERS[q][0], q_end=QUARTERS[q][1], quarter_name=QUARTER_NAMES[q])
    return params


def is_useful(result) -> bool:
    if not result.ok or not result.rows:
        return False
    if len(result.rows) == 1 and len(result.rows[0]) == 1:
        return result.rows[0][0] not in (0, None)
    return True


def generate(per_template: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    values = load_values()
    examples, seen = [], set()
    for tpl in TEMPLATES:
        target = per_template if tpl.params else len(tpl.questions)
        attempts = 0
        kept = 0
        while kept < target and attempts < target * 20:
            attempts += 1
            params = sample_params(tpl, values, rng)
            q_idx = (kept if not tpl.params else rng.randrange(len(tpl.questions)))
            question = tpl.questions[q_idx].format(**params)
            sql = tpl.sql.format(**params)
            if question in seen:
                continue
            if not is_useful(execute_sql(sql)):
                continue
            seen.add(question)
            examples.append(
                {
                    "template_id": tpl.id,
                    "paraphrase_id": q_idx,
                    "category": tpl.category,
                    "question": question,
                    "sql": sql,
                }
            )
            kept += 1
    return examples


def split(examples: list[dict], mode: str, seed: int, ratios=(0.8, 0.1, 0.1)) -> dict[str, list]:
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    if mode == "random":
        ex = examples[:]
        rng.shuffle(ex)
        n = len(ex)
        a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
        out["train"], out["val"], out["test"] = ex[:a], ex[a:b], ex[b:]
    elif mode == "paraphrase":
        # La última formulación de cada plantilla va a test, la penúltima a val (si hay ≥3).
        for e in examples:
            n_q = len(next(t for t in TEMPLATES if t.id == e["template_id"]).questions)
            if e["paraphrase_id"] == n_q - 1:
                out["test"].append(e)
            elif e["paraphrase_id"] == n_q - 2 and n_q >= 3:
                out["val"].append(e)
            else:
                out["train"].append(e)
    elif mode == "template":
        # Estratificado por categoría: de cada categoría se reserva ~10% de plantillas para
        # test y ~10% para val (al menos una de cada si la categoría tiene ≥ 3).
        by_cat = defaultdict(list)
        for t in TEMPLATES:
            by_cat[t.category].append(t.id)
        assign = {}
        for ids in by_cat.values():
            ids = ids[:]
            rng.shuffle(ids)
            n_test = max(1, round(len(ids) * ratios[2])) if len(ids) >= 3 else 0
            n_val = max(1, round(len(ids) * ratios[1])) if len(ids) >= 3 else 0
            for i, tid in enumerate(ids):
                assign[tid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        for e in examples:
            out[assign[e["template_id"]]].append(e)
    else:
        raise ValueError(mode)
    for name, rows in out.items():
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            row["id"] = f"{name}-{i:04d}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-template", type=int, default=20, help="ejemplos por plantilla con parámetros")
    parser.add_argument("--split-mode", choices=["template", "paraphrase", "random"], default="template")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not DB_PATH.exists():
        build_demo_db()

    examples = generate(args.per_template, args.seed)
    splits = split(examples, args.split_mode, args.seed)

    DATA_DIR.mkdir(exist_ok=True)
    for name, rows in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps({"id": row.pop("id"), **row}, ensure_ascii=False) + "\n")

    stats = {
        "split_mode": args.split_mode,
        "seed": args.seed,
        "total": len(examples),
        "templates": len(TEMPLATES),
        "splits": {
            name: {
                "n": len(rows),
                "templates": len({r["template_id"] for r in rows}),
                "by_category": dict(Counter(r["category"] for r in rows)),
            }
            for name, rows in splits.items()
        },
        "avg_sql_chars": round(sum(len(e["sql"]) for e in examples) / len(examples), 1),
    }
    (DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"{stats['total']} ejemplos válidos de {stats['templates']} plantillas (modo {args.split_mode})")
    for name, s in stats["splits"].items():
        print(f"  {name:<5} {s['n']:>4} ejemplos · {s['templates']:>2} plantillas · {s['by_category']}")


if __name__ == "__main__":
    main()
