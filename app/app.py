"""SQLocal · demo Gradio 100 % local.

    python app/app.py                 # http://127.0.0.1:7860
    python app/app.py --host 0.0.0.0  # dentro de Docker

Pestañas:
  1. Probar el modelo  pregunta en inglés → SQL → resultados sobre demo.db
  2. Desempeño         métricas base vs. fine-tuneado, por categoría, curva de pérdida
                       y explorador de aciertos/errores; permite lanzar una evaluación
  3. Dataset           esquema, estadísticas y ejemplos
  4. Privacidad        evidencia de que la app no hace llamadas externas
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlocal.privacy import enable_offline_mode, guard_status, install_network_guard  # noqa: E402

enable_offline_mode()
install_network_guard()

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402

from sqlocal.config import (  # noqa: E402
    BASE_MODEL_ID, DATA_DIR, RESULTS_DIR, VARIANTS, adapter_available,
)
from sqlocal.db import execute_sql  # noqa: E402
from sqlocal.evaluation import load_report, load_split, run_evaluation  # noqa: E402
from sqlocal.prompting import get_schema  # noqa: E402
from sqlocal.templates import CATEGORY_LABELS, SHORT_LABELS  # noqa: E402

# ---------------------------------------------------------------------------
# Modelos: se cargan una sola vez por variante
# ---------------------------------------------------------------------------

_models: dict = {}
_model_lock = threading.Lock()


def get_model(variant: str):
    with _model_lock:
        if variant not in _models:
            from sqlocal.model import SQLModel

            _models[variant] = SQLModel(variant)
        return _models[variant]


def available_variants() -> list[tuple[str, str]]:
    out = [(VARIANTS["base"], "base")]
    if adapter_available():
        out.append((VARIANTS["finetuned"], "finetuned"))
    return out


def _safe_load_split(name: str) -> list[dict]:
    try:
        return load_split(name)
    except FileNotFoundError:
        return []


TEST = _safe_load_split("test")
GOLD_BY_QUESTION = {r["question"].strip().lower(): r for s in ("train", "val", "test") for r in _safe_load_split(s)}


def cat_label(c: str) -> str:
    return CATEGORY_LABELS.get(c, c)


def rows_df(columns: list[str], rows: list) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame()
    # Columnas duplicadas (p. ej. dos COUNT(*)) romperían el DataFrame.
    seen, cols = {}, []
    for c in columns:
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c} ({seen[c]})")
    return pd.DataFrame([list(r) for r in rows], columns=cols)


# ---------------------------------------------------------------------------
# Pestaña 1 · Probar el modelo
# ---------------------------------------------------------------------------


def ask(question: str, variant: str):
    question = (question or "").strip()
    if not question:
        return "", pd.DataFrame(), "Escribe una pregunta en inglés."
    try:
        model = get_model(variant)
    except Exception as exc:  # modelo no descargado, adaptador ausente, etc.
        return "", pd.DataFrame(), (
            f"**No se pudo cargar el modelo:** `{exc}`\n\n"
            "¿Descargaste el modelo base? → `python scripts/download_model.py`"
        )
    gen = model.generate(question)
    res = execute_sql(gen.sql)
    info = [f"**{VARIANTS[variant]}** · {model.info['device']} · generación en **{gen.latency_s:.2f} s**"]
    if res.ok:
        info.append(f"✅ SQL válida · {len(res.rows)} fila(s) en {res.elapsed_ms:.1f} ms")
    else:
        info.append(f"❌ La SQL no se pudo ejecutar: `{res.error}`")

    gold = GOLD_BY_QUESTION.get(question.lower())
    if gold:
        from sqlocal.db import is_ordered, results_match

        ok = results_match(res, execute_sql(gold["sql"]), ordered=is_ordered(gold["sql"]))
        info.append(
            f"\nEsta pregunta está en el dataset ({gold['id']}). Referencia:\n```sql\n{gold['sql']}\n```\n"
            + ("🎯 **Mismo resultado que la referencia**" if ok else "⚠️ **El resultado no coincide con la referencia**")
        )
    if " ".join(gen.raw.split()) != gen.sql:
        info.append(f"<details><summary>Salida cruda del modelo</summary>\n\n```\n{gen.raw}\n```\n</details>")
    return gen.sql, rows_df(res.columns, res.rows) if res.ok else pd.DataFrame(), "\n\n".join(info)


def run_manual_sql(sql: str):
    res = execute_sql(sql)
    if not res.ok:
        return pd.DataFrame(), f"❌ `{res.error}`"
    return rows_df(res.columns, res.rows), f"✅ {len(res.rows)} fila(s) en {res.elapsed_ms:.1f} ms"


# ---------------------------------------------------------------------------
# Pestaña 2 · Desempeño
# ---------------------------------------------------------------------------

FILTERS = [
    "Todos",
    "Aciertos",
    "Errores (SQL válida, resultado distinto)",
    "SQL inválida",
    "Mejorados por el fine-tuning",
    "Empeorados por el fine-tuning",
]


def load_reports() -> dict:
    return {v: r for v in ("base", "finetuned") if (r := load_report(v))}


def summary_markdown(reports: dict) -> str:
    if not reports:
        return (
            "### Aún no hay resultados\n"
            "Lanza una evaluación desde el panel de abajo o con\n"
            "`python scripts/evaluate.py --variant base`.\n\n"
            "Cuando entrenes el adaptador LoRA, evalúa también `--variant finetuned` para comparar."
        )
    lines = [
        "| Modelo | Execution Accuracy | Fáciles | Difíciles | Valid SQL Rate | Latencia media | Ejemplos | Dispositivo |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for v, r in reports.items():
        s = r["summary"]
        lines.append(
            f"| {VARIANTS[v]} | **{s['execution_accuracy']:.1%}** | {s.get('easy_execution_accuracy', 0):.1%} | "
            f"{s.get('hard_execution_accuracy', 0):.1%} | {s['valid_sql_rate']:.1%} | "
            f"{s['avg_latency_s']:.2f} s | {s['n']} | {r['model']['device']} |"
        )
    if len(reports) == 2:
        delta = reports["finetuned"]["summary"]["execution_accuracy"] - reports["base"]["summary"]["execution_accuracy"]
        lines.append(f"\n**Mejora del fine-tuning en Execution Accuracy: {delta:+.1%}**")
    lines.append("\n_Difíciles = JOIN de 3-4 tablas, HAVING, lógica de fechas, varias condiciones y subconsultas._")
    if any(r["summary"]["n"] < len(TEST) for r in reports.values()):
        lines.append(f"\n> ⚠️ Algún modelo se evaluó con menos ejemplos que el test completo ({len(TEST)}).")
    return "\n".join(lines)




def category_df(reports: dict, metric: str) -> pd.DataFrame:
    # Una barra por (categoría, modelo): Gradio apila las barras que comparten x,
    # así que la x combina categoría y modelo para que queden una al lado de la otra.
    rows = []
    for cat in SHORT_LABELS:
        for v, r in reports.items():
            s = r["summary"]["by_category"].get(cat)
            if s is None:
                continue
            short = "FT" if v == "finetuned" else "Base"
            rows.append({"Categoría": f"{SHORT_LABELS[cat]} · {short}",
                         "Modelo": "Fine-tuneado" if v == "finetuned" else "Base",
                         "Valor (%)": round(s[metric] * 100, 1), "n": s["n"]})
    return pd.DataFrame(rows, columns=["Categoría", "Modelo", "Valor (%)", "n"])


def loss_df() -> pd.DataFrame:
    path = RESULTS_DIR / "train_log.json"
    if not path.exists():
        return pd.DataFrame(columns=["Paso", "Pérdida", "Conjunto"])
    rows = []
    for h in json.loads(path.read_text()).get("history", []):
        if "loss" in h:
            rows.append({"Paso": h["step"], "Pérdida": h["loss"], "Conjunto": "train"})
        if "eval_loss" in h:
            rows.append({"Paso": h["step"], "Pérdida": h["eval_loss"], "Conjunto": "val"})
    return pd.DataFrame(rows, columns=["Paso", "Pérdida", "Conjunto"])


def examples_table(reports: dict, filt: str, variant: str):
    if not reports:
        return pd.DataFrame(), []
    variant = variant if variant in reports else next(iter(reports))
    base = {e["id"]: e for e in reports.get("base", {}).get("examples", [])}
    ft = {e["id"]: e for e in reports.get("finetuned", {}).get("examples", [])}
    main = base if variant == "base" else ft

    def keep(eid, e):
        if filt == "Aciertos":
            return e["correct"]
        if filt.startswith("Errores"):
            return e["valid"] and not e["correct"]
        if filt == "SQL inválida":
            return not e["valid"]
        if filt.startswith("Mejorados"):
            return eid in base and eid in ft and ft[eid]["correct"] and not base[eid]["correct"]
        if filt.startswith("Empeorados"):
            return eid in base and eid in ft and base[eid]["correct"] and not ft[eid]["correct"]
        return True

    rows, ids = [], []
    for eid, e in main.items():
        if not keep(eid, e):
            continue
        status = "✅" if e["correct"] else ("⚠️" if e["valid"] else "❌")
        row = {"": status, "id": eid, "Categoría": cat_label(e["category"]), "Pregunta": e["question"]}
        if base and ft:
            row["Base"] = "✅" if base.get(eid, {}).get("correct") else "—"
            row["FT"] = "✅" if ft.get(eid, {}).get("correct") else "—"
        rows.append(row)
        ids.append(eid)
    return pd.DataFrame(rows), ids


def _result_md(res: dict | None) -> str:
    if not res:
        return "_sin resultado_"
    more = f"\n\n_… {res['n_rows']} filas en total_" if res["n_rows"] > len(res["rows"]) else ""
    if not res["rows"]:
        return "_vacío_"
    cell = lambda v: str(round(v, 4) if isinstance(v, float) else v).replace("|", "\\|")  # noqa: E731
    lines = ["| " + " | ".join(res["columns"]) + " |", "|" + "---|" * len(res["columns"])]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in res["rows"]]
    return "\n".join(lines) + more


def example_detail(reports: dict, eid: str) -> str:
    parts = []
    for v, r in reports.items():
        e = next((x for x in r["examples"] if x["id"] == eid), None)
        if e is None:
            continue
        if not parts:
            parts.append(
                f"### {e['question']}\n`{eid}` · plantilla {e['template_id']} · {cat_label(e['category'])}\n\n"
                f"**SQL de referencia**\n```sql\n{e['sql']}\n```\n**Resultado esperado**\n\n{_result_md(e['gold_result'])}"
            )
        verdict = "✅ acierto" if e["correct"] else ("⚠️ SQL válida, resultado distinto" if e["valid"] else f"❌ SQL inválida: `{e['error']}`")
        parts.append(
            f"---\n#### {VARIANTS[v]} — {verdict} · {e['latency_s']:.2f} s\n```sql\n{e['pred_sql'] or '(vacía)'}\n```\n"
            f"{_result_md(e['pred_result']) if e['valid'] else ''}"
        )
    return "\n\n".join(parts) or "Selecciona una fila de la tabla."


def refresh_performance(filt: str, variant: str, metric: str):
    reports = load_reports()
    table, ids = examples_table(reports, filt, variant)
    metric_key = "execution_accuracy" if metric == "Execution Accuracy" else "valid_sql_rate"
    return (
        summary_markdown(reports),
        category_df(reports, metric_key),
        loss_df(),
        table,
        ids,
        "Selecciona una fila de la tabla para ver el detalle.",
    )


def on_select(evt: gr.SelectData, ids: list[str]):
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not ids or row is None or row >= len(ids):
        return "Selecciona una fila de la tabla."
    return example_detail(load_reports(), ids[row])


def launch_evaluation(variant: str, limit: int, progress=gr.Progress()):
    try:
        model = get_model(variant)
    except Exception as exc:
        return f"❌ No se pudo cargar el modelo: `{exc}`"
    examples = TEST[: int(limit)]
    progress(0, desc="Cargando…")

    def cb(i, n, rec):
        progress(i / n, desc=f"{i}/{n} · {rec['question'][:50]}")

    s = run_evaluation(model, examples, progress=cb)["summary"]
    return (f"✅ {VARIANTS[variant]}: Execution Accuracy **{s['execution_accuracy']:.1%}**, "
            f"Valid SQL **{s['valid_sql_rate']:.1%}**, {s['avg_latency_s']:.2f} s/consulta. Pulsa «Actualizar».")


# ---------------------------------------------------------------------------
# Pestaña 3 · Dataset
# ---------------------------------------------------------------------------


def dataset_overview():
    stats_path = DATA_DIR / "stats.json"
    if not stats_path.exists():
        return "Genera el dataset con `python scripts/generate_dataset.py`.", pd.DataFrame()
    st = json.loads(stats_path.read_text())
    cats = sorted({c for s in st["splits"].values() for c in s["by_category"]})
    table = pd.DataFrame(
        [{"Categoría": cat_label(c), **{n: s["by_category"].get(c, 0) for n, s in st["splits"].items()}} for c in cats]
    )
    md = (
        f"**{st['total']} pares** pregunta–SQL validados por ejecución, de **{st['templates']} plantillas** · "
        f"división `{st['split_mode']}` · longitud media de la SQL: {st['avg_sql_chars']} caracteres\n\n"
        + " · ".join(f"**{n}**: {s['n']} ({s['templates']} plantillas)" for n, s in st["splits"].items())
    )
    if st["split_mode"] == "template":
        md += "\n\nLas plantillas de test no aparecen en train: se mide la generalización a consultas nuevas."
    return md, table


def dataset_sample(split: str) -> pd.DataFrame:
    rows = _safe_load_split(split)
    return pd.DataFrame(
        [{"id": r["id"], "Categoría": cat_label(r["category"]), "Pregunta": r["question"], "SQL": r["sql"]} for r in rows]
    )


# ---------------------------------------------------------------------------
# Pestaña 4 · Privacidad
# ---------------------------------------------------------------------------


def privacy_report() -> str:
    st = guard_status()
    blocked = st["conexiones_externas_bloqueadas"]
    lines = [
        f"- Guardia de red activa: **{'sí' if st['guardia_activa'] else 'no'}**",
        f"- Conexiones locales (loopback) permitidas: **{st['conexiones_locales_permitidas']}**",
        f"- Conexiones externas bloqueadas: **{len(blocked)}**",
    ]
    if blocked:
        lines.append("\n| Hora | Destino | Puerto |\n|---|---|---|")
        lines += [f"| {b['hora']} | {b['destino']} | {b['puerto']} |" for b in blocked]
        lines.append("\nEstos intentos los hicieron las librerías (p. ej. comprobaciones de versión) y **no salieron del equipo**.")
    lines.append("\n**Variables de entorno de modo sin conexión**\n")
    lines += [f"- `{k}={v}`" for k, v in st["variables_entorno"].items()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------

# Fuentes del sistema: el tema por defecto descargaría fuentes de Google Fonts.
THEME = gr.themes.Soft(
    primary_hue="blue",
    font=[gr.themes.Font(f) for f in ("ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "sans-serif")],
    font_mono=[gr.themes.Font(f) for f in ("ui-monospace", "SFMono-Regular", "Menlo", "monospace")],
)

EXAMPLE_QUESTIONS = [
    "How many books are there in each genre?",
    "List the 5 longest books.",
    "Which authors are from Japan?",
    "How many loans were made in March 2024?",
    "How many loans were made by users from Seattle?",
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SQLocal", theme=THEME, analytics_enabled=False) as demo:
        gr.Markdown(
            f"# SQLocal\nGeneración de SQL en inglés con un LLM que corre **100 % en local** · "
            f"modelo base `{BASE_MODEL_ID}` · adaptador LoRA: **{'disponible' if adapter_available() else 'aún no entrenado'}**"
        )

        with gr.Tab("Probar el modelo"):
            with gr.Row():
                with gr.Column(scale=3):
                    question = gr.Textbox(label="Pregunta (en inglés)", placeholder="How many books were borrowed in March 2024?", lines=2)
                    gr.Examples(EXAMPLE_QUESTIONS + [r["question"] for r in TEST[:3]], inputs=question, label="Ejemplos")
                with gr.Column(scale=1):
                    variant = gr.Radio(available_variants(), value="base", label="Modelo")
                    ask_btn = gr.Button("Generar SQL", variant="primary")
            sql_out = gr.Code(label="SQL generada", language="sql", interactive=False)
            info_out = gr.Markdown()
            result_out = gr.Dataframe(label="Resultado sobre demo.db", interactive=False, wrap=True)
            ask_btn.click(ask, [question, variant], [sql_out, result_out, info_out])
            question.submit(ask, [question, variant], [sql_out, result_out, info_out])

            with gr.Accordion("Ejecutar SQL a mano (solo lectura)", open=False):
                manual_sql = gr.Code(label="SQL", language="sql", value="SELECT * FROM books LIMIT 10;", interactive=True)
                manual_btn = gr.Button("Ejecutar")
                manual_info = gr.Markdown()
                manual_res = gr.Dataframe(interactive=False, wrap=True)
                manual_btn.click(run_manual_sql, manual_sql, [manual_res, manual_info])

        with gr.Tab("Desempeño"):
            refresh_btn = gr.Button("Actualizar", size="sm")
            summary_md = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    metric = gr.Radio(["Execution Accuracy", "Valid SQL Rate"], value="Execution Accuracy", label="Métrica por categoría")
                    cat_plot = gr.BarPlot(x="Categoría", y="Valor (%)", color="Modelo", y_lim=[0, 100], sort=None,
                                          x_label_angle=-45, title="Desempeño por tipo de consulta", height=420)
                with gr.Column():
                    loss_plot = gr.LinePlot(x="Paso", y="Pérdida", color="Conjunto",
                                            title="Curva de pérdida (tras entrenar)", height=360)
            gr.Markdown("### Explorador de ejemplos del test")
            with gr.Row():
                filt = gr.Dropdown(FILTERS, value="Todos", label="Filtro")
                view_variant = gr.Radio([("Base", "base"), ("Fine-tuneado", "finetuned")], value="base", label="Ver resultados de")
            ids_state = gr.State([])
            ex_table = gr.Dataframe(interactive=False, wrap=True, max_height=380)
            detail = gr.Markdown()

            with gr.Accordion("Lanzar una evaluación desde aquí", open=False):
                gr.Markdown("En CPU cada consulta tarda unos segundos; empieza con pocos ejemplos.")
                with gr.Row():
                    eval_variant = gr.Radio(available_variants(), value="base", label="Modelo")
                    eval_limit = gr.Slider(1, max(len(TEST), 1), value=min(20, max(len(TEST), 1)), step=1, label="N.º de ejemplos del test")
                eval_btn = gr.Button("Evaluar", variant="primary")
                eval_msg = gr.Markdown()
                eval_btn.click(launch_evaluation, [eval_variant, eval_limit], eval_msg)

            perf_outputs = [summary_md, cat_plot, loss_plot, ex_table, ids_state, detail]
            perf_inputs = [filt, view_variant, metric]
            for trigger in (refresh_btn.click, filt.change, view_variant.change, metric.change, demo.load):
                trigger(refresh_performance, perf_inputs, perf_outputs)
            ex_table.select(on_select, ids_state, detail)

        with gr.Tab("Dataset"):
            ds_md = gr.Markdown()
            ds_table = gr.Dataframe(label="Ejemplos por categoría y split", interactive=False)
            ds_split = gr.Radio(["train", "val", "test"], value="test", label="Ver ejemplos de")
            ds_rows = gr.Dataframe(interactive=False, wrap=True, max_height=400)
            gr.Code(get_schema(), language="sql", label="Esquema (DDL en inglés)", interactive=False)
            demo.load(dataset_overview, None, [ds_md, ds_table])
            demo.load(dataset_sample, ds_split, ds_rows)
            ds_split.change(dataset_sample, ds_split, ds_rows)

        with gr.Tab("Privacidad"):
            gr.Markdown(
                "La app instala una **guardia de red** que bloquea toda conexión saliente que no sea a la propia "
                "máquina, y activa el modo sin conexión de Hugging Face y Gradio. El esquema, las preguntas y los datos "
                "nunca salen del equipo. Alcance: se garantiza a nivel de proceso; para aislar a nivel de sistema "
                "usa `docker run --network none` (ver README)."
            )
            priv_md = gr.Markdown()
            gr.Button("Actualizar", size="sm").click(privacy_report, None, priv_md)
            demo.load(privacy_report, None, priv_md)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_ui().queue().launch(server_name=args.host, server_port=args.port, share=False)
