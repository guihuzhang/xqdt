#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for Triple Error Detection on Qwen3-8b
"""
import torch
from swift.llm import sft_main, TrainArguments
from swift.utils import seed_everything
from swift_webnlg_dataset_corrected import register_triple_error_dataset
from verifier_train_eval_webnlg.path_config import CHECKPOINTS_WEBNLG_DIR


def get_device_config():
    """Detect device and return optimal configuration"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        supports_bf16 = torch.cuda.get_device_capability(0)[0] >= 8

        print(f"GPU: {gpu_name}")
        print(f"Using {'bfloat16' if supports_bf16 else 'float16'}")

        return {
            'bf16': supports_bf16,
            'fp16': not supports_bf16,
        }
    else:
        print("No GPU detected, using CPU")
        return {
            'bf16': False,
            'fp16': False,
        }


def train():
    """Train Qwen3-8B for triple error detection"""

    device_config = get_device_config()
    seed_everything(2023)

    # Register datasets
    register_triple_error_dataset(
        shuffle_triples_train=True,
        shuffle_text_prob_train=0.5,
        seed=2023
    )

    print("\nTraining Qwen3-8B for Triple Error Detection\n")

    result = sft_main(TrainArguments(
        # ===== Model Configuration =====
        model='Qwen/Qwen3-8B',
        template='qwen3',  # Qwen template
        train_type='lora',  # 'lora', 'full'
        system="Identify extra, missing and incorrect triples precisely.",

        # ===== Dataset =====
        dataset=['triple_error_train'],
        val_dataset=['triple_error_val'],

        # ===== Training Hyperparameters =====
        num_train_epochs=16,  # 2, 3, 5, 10
        per_device_train_batch_size=4,  # 2, 4, 8, 16
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,  # 2, 4, 8
        # Effective batch size = 16 * 2 = 32

        # ===== Optimization =====
        learning_rate=3e-5,  # 1e-6, 1e-5, 5e-5, 1e-4
        weight_decay=0.01,  # 0.0, 0.01, 0.1
        max_grad_norm=1.0,  # 0.5, 1.0, 5.0

        optim='adamw_torch',  # 'adamw_torch', 'adamw_8bit'

        # ===== Learning Rate Scheduler =====
        lr_scheduler_type='cosine_with_restarts',  # 'cosine', 'linear', 'constant', 'cosine_with_restarts'
        warmup_ratio=1/16,  # 0.0, 0.05, 0.1, 0.2
        lr_scheduler_kwargs={'num_cycles': 15},

        # ===== LoRA Configuration =====
        lora_rank=16,  # 4, 8, 16, 32, 64
        lora_alpha=32,  # typically 2 * rank
        lora_dropout=0.05,  # 0.0, 0.05, 0.1
        target_modules=['all-linear'],

        # ===== Evaluation & Checkpointing =====
        eval_strategy='steps',  # 'steps', 'epoch', 'no'
        eval_steps=277,  # 100, 200, 500
        save_strategy='steps',
        save_steps=277,
        save_total_limit=3,
        save_only_model=False,

        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',  # 'loss', 'eval_loss', 'accuracy'
        greater_is_better=False,
        early_stop_interval=8,

        # early_stopping_patience=5,  # Uncomment to enable early stopping

        # ===== Sequence Length =====
        max_length=1024,  # 1024, 2048, 4096
        truncation_strategy='right',

        # ===== Logging =====
        logging_steps=92,  # 20, 50, 100
        logging_first_step=True,
        report_to=['tensorboard'],  # ['tensorboard'], ['wandb'], ['none']

        # ===== System Configuration =====
        output_dir=str(CHECKPOINTS_WEBNLG_DIR / 'output_qwen3_8b_correct'),

        bf16=device_config['bf16'],
        fp16=device_config['fp16'],

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},

        seed=2023,
        data_seed=2023,

        dataloader_num_workers=4,  # 0, 2, 4, 8
        dataloader_pin_memory=True,

        remove_unused_columns=False,
        lazy_tokenize=True,  # Speed up data loading
    ))

    print("\nTraining completed!")
    # print(f"Best checkpoint: {result.best_model_checkpoint}\n")

    return result


if __name__ == '__main__':
    train()
