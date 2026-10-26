import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
MAIN = ['base', 'full_ft', 'lora']
ORDER = MAIN + [
    'lora_seed43', 'lora_r4', 'lora_r64', 'lora_alpha8', 'lora_alpha128',
    'lora_dropout0', 'lora_dropout0.2', 'lora_qv', 'lora_attention', 'lora_mlp',
]


def load(folder):
    results = {}
    for path in folder.glob('*.json'):
        result = json.loads(path.read_text())
        results[result['name']] = result
    return dict(sorted(results.items(), key=lambda item: ORDER.index(item[0])))


def millions(n):
    return f'{n / 1e6:.1f}M'


def minutes(r):
    return f"{r['train_seconds'] / 60:.1f}" if r.get('train_seconds') else '-'


def gib(r):
    return f"{r['peak_memory_gib']:.1f}" if r.get('peak_memory_gib') else '-'


def scores(r):
    return f"{r['sql_exact_match']:.3f} | {r['sql_valid']:.3f} | {r['mmlu']:.3f}"


def table(results):
    lines = [
        '| модель | обучаемые параметры | доля | время обучения, мин | пик памяти, GiB | SQL EM | SQL valid | MMLU |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for name in MAIN:
        r = results[name]
        share = f"{100 * r['trainable_params'] / r['total_params']:.2f}%" if r['config']['method'] != 'none' else '-'
        trainable = millions(r['trainable_params']) if r['config']['method'] != 'none' else '-'
        lines.append(f"| {name} | {trainable} | {share} | {minutes(r)} | {gib(r)} | {scores(r)} |")
    lines += [
        '',
        '| эксперимент | r | alpha | dropout | target_modules | обучаемые параметры | время, мин | пик памяти, GiB | SQL EM | SQL valid | MMLU |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for name, r in results.items():
        c = r['config']
        if c['method'] != 'lora':
            continue
        targets = ','.join(t.removesuffix('_proj') for t in c['targets'])
        seed = f" seed {c['seed']}" if 'seed' in c else ''
        lines.append(
            f"| {name}{seed} | {c['r']} | {c['alpha']} | {c['dropout']} | {targets} | {millions(r['trainable_params'])} "
            f"| {minutes(r)} | {gib(r)} | {scores(r)} |"
        )
    return '\n'.join(lines) + '\n'


def samples(results):
    lines = []
    names = [n for n in MAIN if n in results]
    for i, row in enumerate(results[names[0]]['samples']):
        lines += [f"## {i + 1}. {row['question']}", '', f"схема: `{row['context']}`", '', f"эталон: `{row['answer']}`", '']
        for name in names:
            s = results[name]['samples'][i]
            assert s['question'] == row['question']
            output = ' '.join(re.sub(r'```(?:sql)?', '', s['output']).split())
            lines.append(f"- {name}{' EM' if s['exact'] else ''}: `{output}`")
        lines.append('')
    return '\n'.join(lines)


def plot(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, names in zip(axes, [['full_ft', 'lora'], [n for n, r in results.items() if r['config']['method'] == 'lora']]):
        for name in names:
            loss = results[name].get('train_loss') or []
            ax.plot([25 * (i + 1) for i in range(len(loss))], loss, label=name)
        ax.set_xlabel('шаг')
        ax.set_ylabel('train loss')
        ax.set_ylim(0.01, 0.08)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title('Full FT и LoRA')
    axes[1].set_title('эксперименты LoRA')
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == '__main__':
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results'
    results = load(out)
    (out / 'table.md').write_text(table(results))
    (out / 'samples.md').write_text(samples(results))
    plot(results, out / 'loss.png')
    print((out / 'table.md').read_text())
