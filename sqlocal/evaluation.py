"""Evaluación: Execution Accuracy y Valid SQL Rate. Lo usan scripts/evaluate.py y la demo."""

from __future__ import annotations

import json
import platform
from collections import defaultdict
from datetime import datetime
from typing import Callable, Iterable

from .config import DATA_DIR, RESULTS_DIR, results_path
from .db import execute_sql, is_ordered, results_match
from .templates import HARD_CATEGORIES


def load_split(name: str = "test") -> list[dict]:
    with (DATA_DIR / f"{name}.jsonl").open() as f:
        return [json.loads(line) for line in f]


def _preview(res, n: int = 5) -> dict:
    return {"columns": res.columns, "rows": [list(r) for r in res.rows[:n]], "n_rows": len(res.rows)}


def evaluate_one(model, example: dict) -> dict:
    gen = model.generate(example["question"])
    pred = execute_sql(gen.sql)
    gold = execute_sql(example["sql"])
    correct = results_match(pred, gold, ordered=is_ordered(example["sql"]))
    return {
        **example,
        "pred_sql": gen.sql,
        "raw_output": gen.raw,
        "valid": pred.ok,
        "correct": correct,
        "error": pred.error,
        "latency_s": round(gen.latency_s, 3),
        "pred_result": _preview(pred) if pred.ok else None,
        "gold_result": _preview(gold),
    }


def summarize(records: list[dict]) -> dict:
    n = len(records)
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    def rate(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 4) if rows else 0.0

    return {
        "n": n,
        "execution_accuracy": rate(records, "correct"),
        "valid_sql_rate": rate(records, "valid"),
        "easy_execution_accuracy": rate([r for r in records if r["category"] not in HARD_CATEGORIES], "correct"),
        "hard_execution_accuracy": rate([r for r in records if r["category"] in HARD_CATEGORIES], "correct"),
        "avg_latency_s": round(sum(r["latency_s"] for r in records) / n, 3) if n else 0.0,
        "by_category": {
            cat: {"n": len(rows), "execution_accuracy": rate(rows, "correct"), "valid_sql_rate": rate(rows, "valid")}
            for cat, rows in sorted(by_cat.items())
        },
    }


def run_evaluation(
    model,
    examples: Iterable[dict],
    split: str = "test",
    progress: Callable[[int, int, dict], None] | None = None,
    save: bool = True,
) -> dict:
    examples = list(examples)
    records = []
    for i, ex in enumerate(examples):
        rec = evaluate_one(model, ex)
        records.append(rec)
        if progress:
            progress(i + 1, len(examples), rec)
    report = {
        "model": model.info,
        "split": split,
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": f"{platform.system()} {platform.machine()}",
        "summary": summarize(records),
        "examples": records,
    }
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        results_path(model.variant).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def load_report(variant: str) -> dict | None:
    path = results_path(variant)
    return json.loads(path.read_text()) if path.exists() else None
