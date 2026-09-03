#!/usr/bin/env python3
"""Run Qwen3-32B on the selected KELM evaluation samples."""

import os
import sys

from prompted_llm_kelm import main


if __name__ == "__main__":
    os.environ.setdefault("USE_HF", "1")
    sys.argv = [sys.argv[0],
        "--backend", "vllm",
        "--model", "Qwen/Qwen3-32B",
        "--model-name", "Qwen3-32B-Prompt",
        "--model-type", "qwen3",
        "--template", "qwen3",
        "--dtype", "bfloat16",
        "--batch-size", "60",
        "--max-num-seqs", "60",
        "--max-model-len", "4096",
        "--max-tokens", "1024",
        "--gpu-memory-utilization", "0.92",
        "--disable-thinking",
        *sys.argv[1:],
    ]
    main()
