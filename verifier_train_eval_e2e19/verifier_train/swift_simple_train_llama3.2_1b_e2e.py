#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Triple Error Detection on Llama-3.2-1B
"""
import torch
from pathlib import Path
from swift.llm import sft_main, TrainArguments
from swift.utils import seed_everything
from swift_e2e_dataset_corrected import register_triple_error_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_JSON = PROJECT_ROOT / 'verifier_train_eval_e2e19/data/train_joint_e2e_webnlg_all_samples.json'
VAL_JSON = PROJECT_ROOT / 'verifier_train_eval_e2e19/data/dev_joint_e2e_webnlg_all_samples.json'
TEST_JSON = PROJECT_ROOT / 'verifier_train_eval_e2e19/data/test_all_samples.json'
OUTPUT_DIR = PROJECT_ROOT / 'checkpoints/checkpoints_e2e19/output_llama3_1b_e2e19'


def train():
    """Train Llama-3.2-1B for triple error detection"""

    seed_everything(2023)

    missing_files = [p for p in [TRAIN_JSON, VAL_JSON, TEST_JSON] if not p.is_file()]
    if missing_files:
        missing_str = '\n'.join(str(p) for p in missing_files)
        raise FileNotFoundError(f"Dataset files not found:\n{missing_str}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Register datasets
    register_triple_error_dataset(
        train_json=str(TRAIN_JSON),
        val_json=str(VAL_JSON),
        test_json=str(TEST_JSON),
        shuffle_triples_train=True,
        shuffle_text_prob_train=0.3,
        seed=2023
    )

    print("\nTraining Llama-3.2-1B for Triple Error Detection\n")

    # Print GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_mem:.1f} GB\n")

    result = sft_main(TrainArguments(
        # ===== Model Configuration =====
        model='LLM-Research/Llama-3.2-1B-Instruct',  # Llama-3.2 1B
        template='llama3_2',
        system="Identify extra, missing and incorrect triples precisely.",
        train_type='lora',

        # ===== Dataset =====
        dataset=['triple_error_train'],
        val_dataset=['triple_error_val'],

        # ===== Training Hyperparameters =====
        num_train_epochs=16,
        per_device_train_batch_size=8,  # 1B: can use 8 or 16
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # adjust accordingly
        # Effective batch = 32

        # ===== Optimization =====
        learning_rate=5e-5,  # 1e-6, 1e-5, 5e-5, 1e-4
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim='adamw_torch',

        # ===== Learning Rate Scheduler =====
        lr_scheduler_type='cosine_with_restarts',
        warmup_ratio=1/16,
        lr_scheduler_kwargs={'num_cycles': 15},

        # ===== LoRA Configuration =====
        # 1B: rank 8 is a practical starting point
        lora_rank=8,  # 8 is enough
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=['all-linear'],

        # ===== Evaluation & Checkpointing =====
        eval_strategy='steps',
        eval_steps=279,  # 44611/32/5 ≈ 279 (evaluate every 1/5 epoch)
        save_strategy='steps',
        save_steps=279,
        save_total_limit=3,
        save_only_model=False,

        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        early_stop_interval=10,

        # ===== Sequence Length =====
        max_length=1024,
        truncation_strategy='right',

        # ===== Logging =====
        logging_steps=93,
        logging_first_step=True,
        report_to=['tensorboard'],

        # ===== System Configuration =====
        output_dir=str(OUTPUT_DIR),

        bf16=True,  # most GPUs support this
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},

        seed=2023,
        data_seed=2023,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        label_smoothing_factor=0.0,
        remove_unused_columns=False,
        lazy_tokenize=True,
    ))

    print("\nTraining completed!")
    # print(f"Best checkpoint: {result.best_model_checkpoint}\n")

    return result


if __name__ == '__main__':
    train()
