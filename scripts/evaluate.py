"""Evalúa el modelo base y/o el fine-tuneado sobre el set de test.

    python scripts/evaluate.py --variant base
    python scripts/evaluate.py --variant both --limit 20

Guarda resultados/eval_base.json y resultados/eval_finetuned.json con las métricas
(Execution Accuracy, Valid SQL Rate, latencia, desglose por categoría) y cada ejemplo.
Funciona sin conexión: la guardia de red bloquea cualquier llamada externa.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from sqlocal.privacy import enable_offline_mode, guard_status, install_network_guard

enable_offline_mode()
install_network_guard()

from sqlocal.config import adapter_available, results_path  # noqa: E402
from sqlocal.evaluation import load_split, run_evaluation  # noqa: E402
from sqlocal.model import SQLModel  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["base", "finetuned", "both"], default="both")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None, help="evaluar solo los N primeros ejemplos")
    p.add_argument("--4bit", dest="four_bit", action="store_true", help="cargar en 4-bit (solo CUDA)")
    args = p.parse_args()

    variants = ["base", "finetuned"] if args.variant == "both" else [args.variant]
    if "finetuned" in variants and not adapter_available():
        print("Aviso: no hay adaptador en adapter/; se evalúa solo el modelo base.")
        variants = [v for v in variants if v != "finetuned"]

    examples = load_split(args.split)[: args.limit]
    for variant in variants:
        model = SQLModel(variant, load_in_4bit=args.four_bit)
        print(f"\n== {variant} · {model.info['device']} · {len(examples)} ejemplos ==")

        def progress(i, n, rec):
            mark = "✓" if rec["correct"] else ("·" if rec["valid"] else "✗")
            print(f"[{i:>3}/{n}] {mark} {rec['latency_s']:.2f}s  {rec['question'][:70]}")

        s = run_evaluation(model, examples, split=args.split, progress=progress)["summary"]
        print(f"\nExecution Accuracy: {s['execution_accuracy']:.1%} · Valid SQL Rate: {s['valid_sql_rate']:.1%} "
              f"· latencia media: {s['avg_latency_s']:.2f}s → {results_path(variant)}")
        del model

    blocked = guard_status()["conexiones_externas_bloqueadas"]
    print(f"\nConexiones externas bloqueadas durante la evaluación: {len(blocked)}")


if __name__ == "__main__":
    main()
