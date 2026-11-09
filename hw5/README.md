# ДЗ 5. RL для вызова функций

Qwen3.5-2B на ToolACE, сид 42. Ревизия ToolACE указана ниже. Ревизия модели в train.py. Делю целые диалоги на обучение и валидацию в пропорции 90/10. Из реплик получаю примеры для обучения. Запуск в окружении из Dockerfile

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
python3 evaluate.py Qwen/Qwen3.5-2B --output results/base
for config in exact partial dr-grpo; do
    python3 train.py "$config" "results/$config" --steps 120
    python3 evaluate.py "results/$config/final" --output "results/$config"
done
```

- exact использует GRPO с точной наградой
- partial использует GRPO с частичной наградой
- dr-grpo меняет нормировки при той же частичной награде

В results/ у каждого варианта свои metrics.json, predictions.json и predictions.jsonl с оценкой на 350 примерах BFCL. Для обученных моделей там же лежат настройки в config.json, статистика данных в data_stats.json и логи в history.json

Все три обучения идут по 120 шагов. BFCL считаю на последнем чекпоинте

Формулы наград, графики и выводы в REPORT.pdf

Ревизия ToolACE

```text
6bda777c88d21e5a204703c1ee45597a8fa4f734
```
