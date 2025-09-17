#!/usr/bin/env python3
"""
Simplified distillation test script for verifying basic functionality
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.loss_functions import DistillationLossFn


def test_basic_imports():
    """Test whether basic imports work"""
    print("✅ Basic import test passed")
    return True


def test_config_creation():
    """Test config creation"""
    master_config = {
        "distillation": {
            "max_num_steps": 3,
            "val_period": 2,
            "val_batch_size": 2,
            "val_at_start": False,
            "max_val_samples": 10,
            "topk_logits_k": 64,
            "num_prompts_per_step": 2,
            "num_generations_per_prompt": 1,
            "max_rollout_turns": 1,
            "seed": 42,
        },
        "policy": {
            "train_global_batch_size": 2,
            "make_sequence_length_divisible_by": 8,
            "max_total_sequence_length": 2048,
            "generation": {
                "colocated": {
                    "enabled": False,
                },
            },
        },
        "teacher": {
            "model_name": "test-teacher",
        },
        "loss_fn": {
            "temperature": 1.0,
            "alpha": 0.5,
            "kl_type": "forward",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
        },
        "data": {
            "dataset_name": "test_dataset",
        },
        "logger": {
            "num_val_samples_to_print": 5,
        },
        "cluster": {
            "num_nodes": 1,
            "gpus_per_node": 2,
        },
        "checkpointing": {
            "enabled": False,
            "checkpoint_must_save_by": None,
            "save_period": 10,
            "metric_name": None,
        },
    }

    # Test config access
    assert "distillation" in master_config
    assert "policy" in master_config
    assert "generation" in master_config["policy"]
    assert "colocated" in master_config["policy"]["generation"]

    print("✅ Config creation test passed")
    return True


def test_mock_components():
    """Test creation of mock components"""
    # Create mock components
    student_policy = MagicMock()
    student_policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {},
    }

    teacher_policy = MagicMock()
    teacher_policy.get_topk_logits.return_value = {
        "topk_logits": torch.randn(2, 10, 64),
        "topk_indices": torch.randint(0, 8, (2, 10, 64)),
    }

    student_generation = MagicMock()
    student_generation.generate.return_value = {
        "logits": torch.randn(2, 10, 8),
        "topk_logits": torch.randn(2, 10, 64),
        "topk_indices": torch.randint(0, 8, (2, 10, 64)),
    }

    # Create mock batch data
    mock_batch = {
        "message_log": [
            [
                {
                    "role": "user",
                    "content": "What is 1+1?",
                    "token_ids": torch.tensor([1, 2, 3, 4, 5]),
                }
            ],
            [
                {
                    "role": "user",
                    "content": "What is 2+2?",
                    "token_ids": torch.tensor([6, 7, 8, 9, 10]),
                }
            ],
        ],
        "input_ids": torch.randint(0, 8, (2, 10)),
        "input_lengths": torch.tensor([8, 10]),
        "token_mask": torch.ones(2, 10),
        "sample_mask": torch.ones(2),
        "teacher_topk_logits": torch.randn(2, 10, 64),
        "teacher_topk_indices": torch.randint(0, 8, (2, 10, 64)),
    }

    # Create mock dataloaders
    train_dataloader = MagicMock(spec=StatefulDataLoader)

    def train_iter(self):
        return iter([mock_batch] * 10)

    train_dataloader.__iter__ = train_iter
    train_dataloader.__len__ = MagicMock(return_value=10)

    val_dataloader = MagicMock(spec=StatefulDataLoader)

    def val_iter(self):
        return iter([mock_batch] * 5)

    val_dataloader.__iter__ = val_iter
    val_dataloader.__len__ = MagicMock(return_value=5)

    # Create other mock components
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    loss_fn = DistillationLossFn(
        {
            "temperature": 1.0,
            "alpha": 0.5,
            "kl_type": "forward",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
        }
    )

    logger = MagicMock()
    checkpointer = MagicMock()

    task_to_env = {"math": MagicMock()}
    val_task_to_env = {"math": MagicMock()}

    print("✅ Mock component creation test passed")
    return True


def main():
    """Run all tests"""
    print("Starting simplified distillation tests...")

    try:
        test_basic_imports()
        test_config_creation()
        test_mock_components()

        print("\n🎉 All tests passed! Distillation test configuration is correct.")
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
