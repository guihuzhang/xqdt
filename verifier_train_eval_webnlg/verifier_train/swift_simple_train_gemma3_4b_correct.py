#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Triple Error Detection on Gemma-3-4b
Ultra-fast training with small model
"""
import torch
from swift.llm import sft_main, TrainArguments
from swift.utils import seed_everything
from swift_webnlg_dataset_corrected import register_triple_error_dataset
from verifier_train_eval_webnlg.path_config import CHECKPOINTS_WEBNLG_DIR


def train():
    """Train Gemma-3-4b for triple error detection"""

    seed_everything(2023)

    # Register datasets
    register_triple_error_dataset(
        shuffle_triples_train=True,
        shuffle_text_prob_train=0.3,
        seed=2023
    )

    print("\nTraining Gemma-3-4b for Triple Error Detection\n")

    # Print GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_mem:.1f} GB\n")

    result = sft_main(TrainArguments(
        # ===== Model Configuration =====
        model='google/gemma-3-4b-it',  # Gemma-3
        template='gemma3_text',
        train_type='lora',

        # ===== Dataset =====
        dataset=['triple_error_train'],
        val_dataset=['triple_error_val'],

        # ===== Training Hyperparameters =====
        num_train_epochs=16,
        per_device_train_batch_size=4,  # 270M: can use 8 or 16
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # adjust accordingly
        # Effective batch = 32 (two devices)

        # ===== Optimization =====
        learning_rate=5e-5,  # small models can use larger LR
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
        eval_steps=277,  # 8832/32/2 ≈ 138 (every half epoch)
        save_strategy='steps',
        save_steps=277,
        save_total_limit=3,
        save_only_model=False,
        early_stop_interval=8,

        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,

        # ===== Sequence Length =====
        max_length=1024,
        truncation_strategy='right',

        # ===== Logging =====
        logging_steps=100,
        logging_first_step=True,
        report_to=['tensorboard'],

        # ===== System Configuration =====
        output_dir=str(CHECKPOINTS_WEBNLG_DIR / 'output_gemma3_4b_correct'),

        bf16=True,  # most GPUs support this

        gradient_checkpointing=True,  # 270M may not need it, but safe to enable
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
