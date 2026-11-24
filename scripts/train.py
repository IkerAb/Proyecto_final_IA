"""Fine-tuning QLoRA de Qwen2.5-Coder-1.5B-Instruct para Text-to-SQL.

Requiere GPU NVIDIA con CUDA (≥ 6 GB de VRAM; p. ej. Colab T4) y requirements-train.txt.

    python scripts/train.py                 # entrena y guarda el adaptador en adapter/
    python scripts/train.py --dry-run       # solo comprueba datos y longitudes (sin GPU)

La pérdida solo se calcula sobre la respuesta (la SQL), no sobre el prompt.
El historial de pérdidas se guarda en resultados/train_log.json (lo muestra la demo).
"""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
from sqlocal.config import ADAPTER_DIR, BASE_MODEL_ID, DATA_DIR, RESULTS_DIR, ROOT
from sqlocal.prompting import build_messages

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"  # marcador del chat template de Qwen


def load_split(name: str) -> list[dict]:
    with (DATA_DIR / f"{name}.jsonl").open() as f:
        return [json.loads(line) for line in f]


def to_text(rows: list[dict], tokenizer) -> list[dict]:
    return [
        {"text": tokenizer.apply_chat_template(build_messages(r["question"], r["sql"]), tokenize=False)}
        for r in rows
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--eval-steps", type=int, default=10)
    p.add_argument("--patience", type=int, default=3, help="early stopping (evaluaciones sin mejora)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows, val_rows = load_split("train"), load_split("val")
    train_text, val_text = to_text(train_rows, tokenizer), to_text(val_rows, tokenizer)

    lengths = [len(tokenizer(x["text"]).input_ids) for x in train_text]
    steps_per_epoch = -(-len(train_rows) // (args.batch_size * args.grad_accum))
    print(f"train={len(train_rows)} val={len(val_rows)} · tokens/ejemplo: max={max(lengths)} "
          f"media={sum(lengths) / len(lengths):.0f} · pasos/época≈{steps_per_epoch}")
    if max(lengths) > args.max_seq_length:
        raise SystemExit(f"Hay ejemplos de más de {args.max_seq_length} tokens: sube --max-seq-length")
    if args.dry_run:
        print("\nEjemplo formateado:\n" + train_text[0]["text"])
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, EarlyStoppingCallback
    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("QLoRA (bitsandbytes 4-bit) necesita una GPU CUDA. Usa Colab T4 o una GPU NVIDIA.")

    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        ),
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    output_dir = ROOT / "checkpoints"
    config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=2,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
    )

    trainer_kwargs = dict(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(train_text),
        eval_dataset=Dataset.from_list(val_text),
        peft_config=lora,
        data_collator=DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    # trl renombró `tokenizer` a `processing_class`; se admite cualquiera de los dos.
    key = "processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters else "tokenizer"
    trainer_kwargs[key] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    # Guardar el mejor adaptador en adapter/
    if ADAPTER_DIR.exists():
        for child in ADAPTER_DIR.iterdir():
            if child.name != ".gitkeep":
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    trainer.model.save_pretrained(str(ADAPTER_DIR))

    RESULTS_DIR.mkdir(exist_ok=True)
    log = {
        "base_model": BASE_MODEL_ID,
        "hyperparameters": {k: v for k, v in vars(args).items() if k != "dry_run"},
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "best_eval_loss": trainer.state.best_metric,
        "history": trainer.state.log_history,
    }
    (RESULTS_DIR / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"Adaptador guardado en {ADAPTER_DIR} · mejor eval_loss={trainer.state.best_metric:.4f}")


if __name__ == "__main__":
    main()
