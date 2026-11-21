#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0

case "${1:-base}" in
    base) exec bash serve.sh --host 127.0.0.1 ;;
    seq32) exec bash serve.sh --host 127.0.0.1 --max-num-seqs 32 ;;
    batch8192) exec bash serve.sh --host 127.0.0.1 --max-num-batched-tokens 8192 ;;
    ngram) exec bash serve.sh --host 127.0.0.1 --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}' ;;
    *) exit 2 ;;
esac
