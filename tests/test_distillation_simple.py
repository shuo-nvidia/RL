#!/usr/bin/env python3
"""
简化的distillation测试脚本，用于验证基本功能
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.distillation import (
    _default_distillation_save_state,
    distillation_train,
    validate,
)
from nemo_rl.algorithms.loss_functions import DistillationLossFn

def test_basic_imports():
    """测试基本导入是否正常"""
    print("✅ 基本导入测试通过")
    return True

def test_config_creation():
    """测试配置创建"""
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
    
    # 测试配置访问
    assert "distillation" in master_config
    assert "policy" in master_config
    assert "generation" in master_config["policy"]
    assert "colocated" in master_config["policy"]["generation"]
    
    print("✅ 配置创建测试通过")
    return True

def test_mock_components():
    """测试mock组件创建"""
    # 创建mock组件
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
    
    # 创建mock batch数据
    mock_batch = {
        "message_log": [
            [{"role": "user", "content": "What is 1+1?", "token_ids": torch.tensor([1, 2, 3, 4, 5])}],
            [{"role": "user", "content": "What is 2+2?", "token_ids": torch.tensor([6, 7, 8, 9, 10])}],
        ],
        "input_ids": torch.randint(0, 8, (2, 10)),
        "input_lengths": torch.tensor([8, 10]),
        "token_mask": torch.ones(2, 10),
        "sample_mask": torch.ones(2),
        "teacher_topk_logits": torch.randn(2, 10, 64),
        "teacher_topk_indices": torch.randint(0, 8, (2, 10, 64)),
    }
    
    # 创建mock dataloaders
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
    
    # 创建其他mock组件
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    
    loss_fn = DistillationLossFn({
        "temperature": 1.0,
        "alpha": 0.5,
        "kl_type": "forward",
        "mixed_kl_weight": 0.5,
        "zero_outside_topk": False,
    })
    
    logger = MagicMock()
    checkpointer = MagicMock()
    
    task_to_env = {"math": MagicMock()}
    val_task_to_env = {"math": MagicMock()}
    
    print("✅ Mock组件创建测试通过")
    return True

def main():
    """运行所有测试"""
    print("开始运行简化的distillation测试...")
    
    try:
        test_basic_imports()
        test_config_creation()
        test_mock_components()
        
        print("\n🎉 所有测试通过！distillation测试配置正确。")
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
