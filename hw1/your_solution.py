import json
import math
import os
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
)
import torch
import time

# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000

TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/gpt2-1b-russian',
    'optim': 'adamw_torch_fused',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 32,
    'learning_rate': 5e-5,
    'lr_scheduler_type': 'linear',
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'logging_steps': 10,
    'eval_strategy': 'no',
    'save_strategy': 'no',
    'save_only_model': True,
    'bf16': True,
    'tf32': True,
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': 1,
    'dataloader_num_workers': 4,
    'torch_compile': True,
    'report_to': 'none',
    'seed': 42,
}

PROMPTS = [
    'Москва — столица',
    'Учёные обнаружили, что',
    'В 1812 году',
    'Нейронные сети',
]

class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                # Include the final weights in best-checkpoint selection.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control

class ElapsedCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, **kwargs):
        state.log_history[-1]['elapsed'] = round(time.time() - self.start_time, 1)

def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer

def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
    )
    tokenized[LABELS] = [
        [token if keep else -100 for token, keep in zip(ids, mask)]
        for ids, mask in zip(tokenized[INPUT_IDS], tokenized[ATTENTION_MASK])
    ]
    return {key: tokenized[key] for key in (INPUT_IDS, ATTENTION_MASK, LABELS)}

def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    os.makedirs(output_dir, exist_ok=True)
    for index in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        shard.to_parquet(f'{output_dir}/{index:05d}.parquet')

def prepare_dataset(output_dir=OUTPUT_DIR):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenizer = prepare_tokenizer()
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer},
        remove_columns=dataset.column_names,
        num_proc=16,
    )
    save_as_parquets(tokenized, output_dir)

def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    files = sorted(str(path) for path in Path(data_dir).glob('*.parquet'))
    return load_dataset('parquet', data_files=files)['train']

def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset

def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )

    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16
    )

    print(f"Model pad token id: {model.config.pad_token_id}")

    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")

    return model

def generate_samples(model, tokenizer, prompts=PROMPTS):
    model.eval()
    torch.manual_seed(42)
    samples = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        samples.append({'prompt': prompt, 'text': tokenizer.decode(output[0], skip_special_tokens=True)})
    return samples

def train_model(data_dir=OUTPUT_DIR, output_dir=None, overrides=None):
    config = {**TRAINING_CONFIG, **(overrides or {})}
    if output_dir:
        config['output_dir'] = output_dir
    print(json.dumps(config, indent=2))

    tokenizer = prepare_tokenizer()
    train_dataset, eval_dataset = split_dataset(load_tokenized_dataset(data_dir))
    set_seed(config['seed'])
    model = create_model(tokenizer)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)] # dont change
        )
    trainer.add_callback(ElapsedCallback())
    train_result = trainer.train()
    eval_results = next((row for row in reversed(trainer.state.log_history) if 'eval_loss' in row), None)
    if eval_results is None:
        print("Running final evaluation...")
        eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    trainer.save_state()

    try:
        samples = generate_samples(trainer.model, tokenizer)
    except RuntimeError as error:
        samples = [{'error': str(error)}]

    eval_loss = eval_results['eval_loss']
    return {
        'eval_loss': eval_loss,
        'perplexity': math.exp(min(eval_loss, 50)),
        'global_step': trainer.state.global_step,
        'train_runtime': train_result.metrics['train_runtime'],
        'pad_fraction': 1 - sum(map(sum, eval_dataset[ATTENTION_MASK])) / (len(eval_dataset) * MAX_LENGTH),
        'config': config,
        'log_history': trainer.state.log_history,
        'samples': samples,
    }

if __name__ == "__main__":
    # Step 1: Prepare the dataset (run once)
    # prepare_dataset()

    # Step 2: Train the model
    train_model()
