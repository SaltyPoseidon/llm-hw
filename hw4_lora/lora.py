import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
TRAIN_SIZE = 20_000
TEST_SIZE = 1000
MMLU_SIZE = 1000
SAMPLES = 10
SYSTEM = "You translate questions into SQL. Use only the tables and columns from the schema. Reply with a single SQL query only."

ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
LORA = dict(method="lora", lr=2e-4, r=16, alpha=32, dropout=0.05, targets=ATTENTION + MLP)
EXPERIMENTS = {
    "base": dict(method="none"),
    "full_ft": dict(method="full", lr=1e-5),
    "lora": LORA,
    "lora_seed43": {**LORA, "seed": 43},
    "lora_r4": {**LORA, "r": 4},
    "lora_r64": {**LORA, "r": 64},
    "lora_alpha8": {**LORA, "alpha": 8},
    "lora_alpha128": {**LORA, "alpha": 128},
    "lora_dropout0": {**LORA, "dropout": 0.0},
    "lora_dropout0.2": {**LORA, "dropout": 0.2},
    "lora_qv": {**LORA, "targets": ["q_proj", "v_proj"]},
    "lora_attention": {**LORA, "targets": ATTENTION},
    "lora_mlp": {**LORA, "targets": MLP},
}


def load_sql(train_size, test_size):
    frame = load_dataset("b-mc2/sql-create-context", split="train").to_pandas()
    frame = frame.drop_duplicates(subset=["question", "context"])
    data = Dataset.from_pandas(frame, preserve_index=False).shuffle(seed=SEED)
    test = data.select(range(test_size))
    train = data.select(range(test_size, test_size + train_size))
    return train, test


def prompt(row):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Schema:\n{row['context']}\n\nQuestion: {row['question']}"},
    ]


def extract_sql(text):
    block = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if block:
        text = block.group(1)
    return text.strip().split(";")[0].strip()


def normalize(sql):
    sql = sql.replace('"', "'").lower()
    sql = re.sub(r"\s+", " ", sql).strip()
    return re.sub(r"\s*([(),=<>*])\s*", r"\1", sql)


def is_valid(sql, context):
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(context)
        connection.execute(sql).fetchall()
        return True
    except sqlite3.Error:
        return False
    finally:
        connection.close()


@torch.no_grad()
def generate(model, tokenizer, conversations, batch_size=32, max_new_tokens=128):
    tokenizer.padding_side = "left"
    outputs = []
    for start in range(0, len(conversations), batch_size):
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations[start:start + batch_size]
        ]
        batch = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        tokens = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        outputs += tokenizer.batch_decode(tokens[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    return outputs


def evaluate_sql(model, tokenizer, test):
    outputs = generate(model, tokenizer, [prompt(row) for row in test])
    predictions = [extract_sql(output) for output in outputs]
    exact = [normalize(p) == normalize(row["answer"]) for p, row in zip(predictions, test)]
    valid = [is_valid(p, row["context"]) for p, row in zip(predictions, test)]
    samples = [
        dict(question=row["question"], context=row["context"], answer=row["answer"], output=output, exact=e)
        for row, output, e in list(zip(test, outputs, exact))[:SAMPLES]
    ]
    return dict(sql_exact_match=sum(exact) / len(exact), sql_valid=sum(valid) / len(valid), samples=samples)


@torch.no_grad()
def evaluate_mmlu(model, tokenizer, size, batch_size=16):
    data = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=SEED).select(range(size))
    letters = ["A", "B", "C", "D"]
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in letters]
    tokenizer.padding_side = "left"
    correct = 0
    for start in range(0, size, batch_size):
        rows = data.select(range(start, min(start + batch_size, size)))
        texts = []
        for row in rows:
            options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, row["choices"]))
            question = f"{row['question']}\n{options}\nAnswer with the letter of the correct option only."
            texts.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True,
            ))
        batch = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        logits = model(**batch).logits[:, -1, letter_ids]
        correct += (logits.argmax(-1).cpu() == torch.tensor(rows["answer"])).sum().item()
    return correct / size


def train(model, tokenizer, train_data, config, output_dir):
    dataset = train_data.map(
        lambda row: {"prompt": prompt(row), "completion": [{"role": "assistant", "content": row["answer"]}]},
        remove_columns=train_data.column_names,
    )
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=config["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        max_length=512,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        seed=config.get("seed", SEED),
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    return dict(
        train_seconds=time.perf_counter() - started,
        peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
        train_loss=[x["loss"] for x in trainer.state.log_history if "loss" in x],
    )


def run(name, smoke=False):
    config = EXPERIMENTS[name]
    set_seed(config.get("seed", SEED))
    train_size, test_size, mmlu_size = (200, 20, 32) if smoke else (TRAIN_SIZE, TEST_SIZE, MMLU_SIZE)
    train_data, test = load_sql(train_size, test_size)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    dtype = torch.float32 if config["method"] == "full" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to("cuda")
    if config["method"] == "lora":
        model = get_peft_model(model, LoraConfig(
            r=config["r"], lora_alpha=config["alpha"], lora_dropout=config["dropout"],
            target_modules=config["targets"], bias="none", task_type="CAUSAL_LM",
        ))
    result = dict(
        name=name,
        config=config,
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters()),
    )
    if config["method"] != "none":
        result.update(train(model, tokenizer, train_data, config, f"/tmp/out/{name}"))
        if config["method"] == "full":
            model = model.to(torch.bfloat16)
    model.eval()
    result.update(evaluate_sql(model, tokenizer, test))
    result["mmlu"] = evaluate_mmlu(model, tokenizer, mmlu_size)
    return result


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = Path(args[1] if len(args) > 1 else "results")
    out.mkdir(parents=True, exist_ok=True)
    names = list(EXPERIMENTS) if args[0] == "all" else args[0].split(",")
    for experiment in names:
        result = run(experiment, smoke="--smoke" in sys.argv)
        (out / f"{experiment}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(experiment, {k: result[k] for k in ("sql_exact_match", "sql_valid", "mmlu")}, flush=True)
