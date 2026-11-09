import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

MODEL = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DATA = "Team-ACE/ToolACE"
DATA_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
SEED = 42
CONFIGS = {
    "exact": ("grpo", "group", False),
    "partial": ("grpo", "group", True),
    "dr-grpo": ("dr_grpo", "none", True),
}
TOOL_CALL = re.compile(r"<tool_call>\s*<function=([^>]+)>\s*(.*?)</function>\s*</tool_call>", re.S)
PARAMETER = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.S)


def reference_calls(answer, names):
    if not answer.startswith("["):
        return []
    for index, name in sorted(enumerate(names), key=lambda pair: -len(pair[1])):
        pattern = r"(?<=[\[,])\s*" + re.escape(name) + r"(?=\s*\()"
        answer = re.sub(pattern, f" tool_{index}", answer)
    expression = ast.parse(answer, mode="eval").body
    if not isinstance(expression, ast.List):
        raise ValueError("Expected a list of calls")
    calls = []
    for call in expression.elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValueError("Expected a named call")
        if call.args or not re.fullmatch(r"tool_\d+", call.func.id):
            raise ValueError("Unknown tool or positional arguments")
        arguments = {arg.arg: ast.literal_eval(arg.value) for arg in call.keywords}
        if None in arguments or len(arguments) != len(call.keywords):
            raise ValueError("Invalid arguments")
        calls.append((call.func.id, arguments))
    return calls


def predicted_calls(answer):
    if not answer.strip():
        return None
    calls = []
    for match in TOOL_CALL.finditer(answer):
        name, body = match.groups()
        args = {}
        for key, raw in PARAMETER.findall(body):
            if key in args:
                return None
            try:
                args[key] = json.loads(raw)
            except ValueError:
                args[key] = raw
        if PARAMETER.sub("", body).strip():
            return None
        calls.append((name.strip(), args))
    remainder = TOOL_CALL.sub("", answer)
    if any(tag in remainder for tag in ("<tool_call", "</tool_call", "<function=", "<parameter=")):
        return None
    return calls


def reward(answer, target, partial):
    predicted = predicted_calls(answer)
    if predicted is None:
        return 0.0
    expected = json.loads(target)
    encode = lambda call: json.dumps(call, sort_keys=True)
    actual_calls = Counter(map(encode, predicted))
    expected_calls = Counter(map(encode, expected))
    if actual_calls == expected_calls:
        return 1.0
    if not partial or not predicted or not expected:
        return 0.0
    actual_names = Counter(name for name, _ in predicted)
    expected_names = Counter(name for name, _ in expected)
    actual_args = Counter(encode([name, key, value]) for name, args in predicted for key, value in args.items())
    expected_args = Counter(encode([name, key, value]) for name, args in expected for key, value in args.items())
    name_overlap = 2 * sum((actual_names & expected_names).values()) / (len(predicted) + len(expected))
    arg_total = actual_args.total() + expected_args.total()
    arg_overlap = 2 * sum((actual_args & expected_args).values()) / arg_total if arg_total else name_overlap
    return 0.45 * name_overlap + 0.45 * arg_overlap


def convert_dialogue(row, tokenizer):
    raw_tools = json.JSONDecoder().raw_decode(row["system"][row["system"].index("[{"):])[0]
    names = [tool["name"] for tool in raw_tools]
    tools = []
    for index, tool in enumerate(raw_tools):
        schema = dict(tool["parameters"])
        if schema.get("type") == "dict":
            schema["type"] = "object"
        tools.append({"type": "function", "function": {
            "name": f"tool_{index}", "description": f'{tool["name"]}: {tool.get("description", "")}',
            "parameters": schema,
        }})

    examples = []
    messages = [{"role": "system", "content": "Use the available tools when appropriate. Ask for missing information when needed."}]
    for message in row["conversations"]:
        role, content = message["from"], message["value"].strip()
        if role == "assistant":
            calls = reference_calls(content, names)
            if messages[-1]["role"] in {"user", "tool"} and (calls or content):
                prompt = tokenizer.apply_chat_template(
                    messages, tools=tools, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
                examples.append({"prompt": prompt, "target": json.dumps(calls)})
            if calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": name, "arguments": arguments}}
                    for name, arguments in calls
                ]})
                continue
        if role not in {"user", "assistant", "tool"}:
            raise ValueError("Unknown dialogue role")
        messages.append({"role": role, "content": content})
    return examples


def prepare(tokenizer):
    from datasets import Dataset, load_dataset

    dialogues = load_dataset(DATA, revision=DATA_REVISION, split="train")
    splits = dialogues.train_test_split(test_size=0.1, seed=SEED)
    prepared = []
    stats = {}
    for name in ("train", "test"):
        rows = []
        skipped = 0
        for dialogue in splits[name]:
            try:
                rows.extend(convert_dialogue(dialogue, tokenizer))
            except (KeyError, TypeError, ValueError, SyntaxError):
                skipped += 1
        lengths = [len(tokens) for tokens in tokenizer(
            [row["prompt"] for row in rows], add_special_tokens=False
        )["input_ids"]]
        kept = [row for row, length in zip(rows, lengths) if length <= 3072]
        stats[name] = {
            "dialogues": len(splits[name]),
            "skipped_dialogues": skipped,
            "examples": len(kept),
            "filtered_long": len(rows) - len(kept),
            "reference_call_counts": dict(Counter(len(json.loads(row["target"])) for row in kept)),
            "mean_prompt_tokens": sum(length for length in lengths if length <= 3072) / len(kept),
        }
        prepared.append(Dataset.from_list(kept).shuffle(seed=SEED))
    return *prepared, stats


def main():
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    parser = argparse.ArgumentParser()
    parser.add_argument("config", choices=CONFIGS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    loss, scale, partial = CONFIGS[args.config]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, padding_side="left")
    train, validation, stats = prepare(tokenizer)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data_stats.json").write_text(json.dumps(stats, indent=2))
    config = GRPOConfig(
        output_dir=str(args.output / "training"),
        max_steps=args.steps,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=256,
        temperature=0.8,
        loss_type=loss,
        scale_rewards=scale,
        beta=0.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=max(1, args.steps // 3),
        save_strategy="no",
        report_to="none",
        seed=SEED,
    )
    (args.output / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    def score(completions, target, **kwargs):
        return [reward(answer, expected, partial) for answer, expected in zip(completions, target)]

    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16)
    trainer = GRPOTrainer(
        model=model, args=config, processing_class=tokenizer, reward_funcs=score,
        train_dataset=train, eval_dataset=validation.select(range(min(4 if args.steps <= 2 else 64, len(validation)))),
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    trainer.train()
    (args.output / "history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    merged = trainer.accelerator.unwrap_model(trainer.model).merge_and_unload()
    full = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision=MODEL_REVISION, dtype=torch.bfloat16,
    )
    full.model.language_model.load_state_dict(merged.model.state_dict(), strict=True)
    full.lm_head.load_state_dict(merged.lm_head.state_dict(), strict=True)
    full.save_pretrained(args.output / "final")
    tokenizer.save_pretrained(args.output / "final")


if __name__ == "__main__":
    main()
