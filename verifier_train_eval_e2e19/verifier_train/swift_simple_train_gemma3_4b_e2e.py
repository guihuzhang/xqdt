#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Triple Error Detection on Gemma-3-4B
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
OUTPUT_DIR = PROJECT_ROOT / 'checkpoints/checkpoints_e2e19/output_gemma3_4b_e2e19'


def train():
    """Train Gemma-3-4B for triple error detection"""

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

    print("\nTraining Gemma-3-4B for Triple Error Detection\n")

    # Print GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_mem:.1f} GB\n")

    result = sft_main(TrainArguments(
        # ===== Model Configuration =====
        model='google/gemma-3-4b-it',  # Gemma-3 4B
        template='gemma3_text',
        train_type='lora',

        # ===== Dataset =====
        dataset=['triple_error_train'],
        val_dataset=['triple_error_val'],

        # ===== Training Hyperparameters =====
        num_train_epochs=16,
        per_device_train_batch_size=4,  # 4B on 2 GPUs: 4 per device is a practical default
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # adjust accordingly
        # Effective batch size with torchrun --nproc_per_node=2: 4 * 4 * 2 = 32

        # ===== Optimization =====
        learning_rate=5e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim='adamw_torch',

        # ===== Learning Rate Scheduler =====
        lr_scheduler_type='cosine_with_restarts',
        warmup_ratio=1/16,
        lr_scheduler_kwargs={'num_cycles': 15},

        # ===== LoRA Configuration =====
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['all-linear'],

        # ===== Evaluation & Checkpointing =====
        eval_strategy='steps',
        eval_steps=279,  # 44611/32/5 ≈ 279 (for 2-GPU global batch 32)
        save_strategy='steps',
        save_steps=279,
        save_total_limit=3,
        save_only_model=False,
        early_stop_interval=10,

        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,

        # ===== Sequence Length =====
        max_length=1024,
        truncation_strategy='right',

        # ===== Logging =====
        logging_steps=93,
        logging_first_step=True,
        report_to=['tensorboard'],

        # ===== System Configuration =====
        output_dir=str(OUTPUT_DIR),

        bf16=True,  # Keep enabled for modern GPUs

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},

        seed=2023,
        data_seed=2023,

        dataloader_num_workers=4,
        dataloader_pin_memory=True,

        label_smoothing_factor=0.0,
        remove_unused_columns=False,
        # lazy_tokenize=True,
    ))

    print("\nTraining completed!")
    # print(f"Best checkpoint: {result.best_model_checkpoint}\n")

    return result


if __name__ == '__main__':
    train()
