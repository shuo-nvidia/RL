# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.
import os
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import ray
import torch
import numpy as np
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from nemo_rl.algorithms.loss_functions import (
    DistillationLossConfig,
    DistillationLossDataDict,
    DistillationLossFn,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, rl_collate_fn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
)
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.timer import Timer, TimeoutChecker
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import run_multi_turn_rollout, run_async_multi_turn_rollout
from nemo_rl.algorithms.grpo import _should_use_async_rollouts
from nemo_rl.utils.nsys import maybe_gpu_profile_step

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class DistillationConfig(TypedDict):
    teacher_model_path: str           # Teacher model path (for loading weights)
    generate_strategy: dict[str, Any] # Generation strategy parameters
    
    # Training configuration
    max_num_steps: int
    eval_steps: int
    save_steps: int
    logging_steps: int


class MasterConfig(TypedDict):
    """Main configuration structure"""
    policy: PolicyConfig             # Student model configuration
    teacher: PolicyConfig            # Teacher model configuration
    loss_fn: DistillationLossConfig  # Loss function configuration
    env: dict[str, Any]              # Environment configuration
    data: DataConfig                 # Data configuration
    distillation: DistillationConfig    # Distillation configuration
    logger: LoggerConfig                # Logger configuration
    cluster: ClusterConfig              # Cluster configuration
    checkpointing: CheckpointingConfig  # Checkpointing configuration


class DistillationSaveState(TypedDict):
    step: int
    val_reward: NotRequired[float]
    consumed_samples: int
    # epoch: int


def _default_distillation_save_state() -> DistillationSaveState:
    return {
        "step": 0,
        "consumed_samples": 0,
        # "epoch": 0,
    }


# ===============================================================================
# Setup Functions
# ===============================================================================
def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,     # student_policy
    Optional[GenerationInterface],  # student_generation
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    TokenizerType,                  # tokenizer
    DistillationLossFn,
    Logger,
    CheckpointManager,
    DistillationSaveState,
    MasterConfig,
]:
    """Main entry point for distillation algorithm
    
    Returns:
        tuple of student_policy, student_generation, 
        (train_cluster, inference_cluster), train_dataloader, val_dataloader, 
        loss_fn, logger, checkpointer, distillation_save_state, master_config
    """
    # Extract configuration
    policy_config = master_config["policy"]
    teacher_config = master_config["teacher"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    distillation_config = master_config["distillation"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for distillation"
    )

    # Set random seed
    set_seed(42)  

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    distillation_save_state: Optional[DistillationSaveState] = cast(
        Optional[DistillationSaveState], 
        checkpointer.load_training_info(last_checkpoint_path)
    )
    if distillation_save_state is None:
        distillation_save_state = _default_distillation_save_state()

    # ==========================
    #           Data
    # ==========================
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=distillation_config["num_prompts_per_step"],  
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
    )
    
    if last_checkpoint_path:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)

    # Validation dataset
    val_dataloader: Optional[StatefulDataLoader] = None
    if val_dataset is not None:
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distillation_config["val_batch_size"],  
            shuffle=False,
            collate_fn=rl_collate_fn,
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    colocated_inference = generation_config["colocated"]["enabled"]

    if colocated_inference:
        cluster = RayVirtualCluster(
            name="distillation_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]] * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1
            if generation_config["backend"] == "megatron"
            else 3,
        )
        train_cluster = cluster
        inference_cluster = cluster
        print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")
    
    else:
        assert generation_config["backend"] != "megatron", (
            "Non-colocated inference is not supported for Megatron generation backends. "
            "Please use vLLM backend for generation."
        )

        # train resources will be updated through overall and inference resources below
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        # validate and configure resources
        if cluster_config["num_nodes"] == 1:
            if inference_gpus_per_node is None:
                inference_gpus_per_node = cluster_config["gpus_per_node"] // 2
            if inference_nodes is None:
                inference_nodes = 1
        else:
            if inference_gpus_per_node is None:
                inference_gpus_per_node = cluster_config["gpus_per_node"]
            if inference_nodes is None:
                inference_nodes = cluster_config["num_nodes"] // 2

        # validate resources
        if inference_gpus_per_node > cluster_config["gpus_per_node"]:
            raise ValueError(
                f"Inference GPUs per node ({inference_gpus_per_node}) cannot be greater than "
                f"total GPUs per node ({cluster_config['gpus_per_node']})"
            )
        if inference_nodes > cluster_config["num_nodes"]:
            raise ValueError(
                f"Inference nodes ({inference_nodes}) cannot be greater than "
                f"total nodes ({cluster_config['num_nodes']})"
            )

        # update train resources
        train_gpus_per_node = cluster_config["gpus_per_node"] - inference_gpus_per_node
        train_nodes = cluster_config["num_nodes"] - inference_nodes

        # create clusters
        train_cluster = RayVirtualCluster(
            name="distillation_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        inference_cluster = RayVirtualCluster(
            name="distillation_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        print(f"  ✓ Separate clusters created: train={train_nodes}x{train_gpus_per_node}GPUs, inference={inference_nodes}x{inference_gpus_per_node}GPUs")

    # ==========================
    #      Student Policy
    # ==========================
    print("\n▶ Setting up models...")
    
    # Checkpoint paths
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
    else:
        weights_path = None
        optimizer_path = None

    student_policy = Policy(
        name_prefix="student",
        cluster=train_cluster,  
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,  
    )

    # ==========================
    #      Teacher Policy
    # ==========================
    print("\n▶ Setting up models...")
    
    # Checkpoint paths
    weights_path = None
    optimizer_path = None

    teacher_policy = Policy(
        name_prefix="teacher",
        cluster=train_cluster,  
        config=teacher_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=False,
        init_reference_model=False,  
    )

    # ==========================
    #    Generation Interface
    # ==========================
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "megatron":
        student_generation = None
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        student_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        student_generation.finish_generation()

    if student_generation is not None:
        state_dict_info = student_policy.prepare_refit_info()
        student_generation.prepare_refit_info(state_dict_info)

    # ==========================
    #        Loss Function
    # ==========================
    loss_fn = DistillationLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        student_policy,
        teacher_policy,
        student_generation,
        train_dataloader,
        val_dataloader,
        tokenizer,  
        loss_fn,
        logger,
        checkpointer,
        distillation_save_state,
        master_config,
    )


# ===============================================================================
# Core Algorithm Functions
# ===============================================================================

def refit_student_generation(
    student_policy: ColocatablePolicyInterface,
    student_generation: GenerationInterface,
    colocated_inference: bool,
    _refit_buffer_size_gb: Optional[int] = None,
    timer: Optional[Timer] = None,
    generation_config: Optional[dict] = None,
    master_config: Optional[dict] = None,
) -> None:
    """Refit the student generation interface with the latest policy weights.
    
    Args:
        student_policy: The student policy model
        student_generation: The student generation interface
        colocated_inference: Whether to use colocated inference
        _refit_buffer_size_gb: Buffer size in GB
        timer: Timer for performance measurement
        generation_config: Generation configuration dictionary
        master_config: Master configuration dictionary for parameters like max_total_sequence_length
    """
    if colocated_inference:
        student_policy.offload_before_refit()
        student_generation.prepare_for_generation(tags=["weights"])
        

    # Create a context manager that does nothing when timer is None
    timer_context = (
        timer.time("prepare_for_generation/transfer_and_update_weights")
        if timer is not None
        else nullcontext()
    )
    with timer_context:
        # Update weights
        update_success = False
        if colocated_inference:
            # Get model parameter keys, grouped by size
            grouped_param_keys = student_policy.prepare_weights_for_ipc(
                _refit_buffer_size_gb=_refit_buffer_size_gb
            )
            
            # Execute updates
            for keys in grouped_param_keys:
                ipc_handles = student_policy.get_weights_ipc_handles(keys)
                update_success = student_generation.update_weights_from_ipc_handles(ipc_handles)
                if not update_success:
                    break
        else:
            # Update weights through NCCL
            futures_train = student_policy.broadcast_weights_for_collective()
            futures_inference = student_generation.update_weights_from_collective()
            # Wait for all futures to complete
            ray.get(futures_train)
            results = ray.get(futures_inference)
            update_success = all(result for result in results if result is not None)

        # Check if update was successful
        if not update_success:
            error_tag = "cuda-ipc" if colocated_inference else "nccl"
            error_message = (
                "❌ Error: Updating weights for the student generation policy failed during refit.\n"
                f"This often indicates an issue with {error_tag} or "
                "a problem within the generation backend (e.g., vLLM worker).\n"
            )
            raise RuntimeError(error_message)

    if colocated_inference:
        student_policy.offload_after_refit()
        student_generation.prepare_for_generation(tags=["kv_cache"])


# ===============================================================================
# Training & Validation
# ===============================================================================


def distillation_train(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    student_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: DistillationLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: DistillationSaveState,
    master_config: MasterConfig,
) -> None:
    """Run GRPO training algorithm."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    # If policy_generation is None, use the policy as the generation interface (megatron framework backend)
    if student_generation is None:
        student_generation = student_policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert student_generation is not None  # for mypy type check

    # common config/state itmes
    step = distillation_save_state["step"]
    consumed_samples = distillation_save_state["consumed_samples"]
    # epoch = distillation_save_state["epoch"]
    val_period = master_config["distillation"]["val_period"]
    val_at_start = master_config["distillation"]["val_at_start"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    # max_num_epochs = master_config["distillation"].get("max_num_epochs", 1)

    # Run validation at the start if configured
    if val_at_start and step == 0:
        print("\n🔍 Running initial validation...")
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_student_generation(student_policy, student_generation, colocated_inference)
            POLICY_GENERATION_STALE = False
        else:
            student_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            student_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=0,
            master_config=master_config,
        )
        student_generation.finish_generation()
        logger.log_metrics(val_metrics, step, prefix="validation")
        logger.log_metrics(validation_timings, step, prefix="timing/validation")

    # Run grpo training (multi-epoch until reaching max_num_steps)
    batch: BatchedDataDict[DatumSpec]
    max_steps = master_config["distillation"]["max_num_steps"]

    while step < max_steps:
        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {step + 1}/{max_steps} {'=' * 25}"
            )
            maybe_gpu_profile_step(student_policy, step + 1)
            if student_policy != student_generation:
                maybe_gpu_profile_step(student_generation, step + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # Prepare batch
                print("▶ Preparing batch...")
                with timer.time("data_processing"):
                    # Repeat batch items
                    repeated_batch: BatchedDataDict[DatumSpec] = batch.repeat_interleave(
                        master_config["distillation"]["num_generations_per_prompt"]
                    )
                    # Convert LLMMessageLogType to FlatMessagesType for generation
                    batched_flat, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                    )
                    input_ids = batched_flat["token_ids"]

                # Generate responses - this updates the LLMMessageLogType in repeated_batch
                print(f"▶ Generating responses for batch of size {repeated_batch.size}...")
                with timer.time("prepare_for_generation"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_student_generation(
                            student_policy, student_generation, colocated_inference, timer=timer
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()

                with timer.time("generation"):
                    # Use async rollouts if vLLM async engine is enabled
                    if _should_use_async_rollouts(master_config):
                        (
                            repeated_batch,
                            rollout_metrics,
                        ) = run_async_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["distillation"]["max_rollout_turns"],
                            greedy=False,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["distillation"]["max_rollout_turns"],
                            greedy=False,
                        )
                    student_generation.finish_generation()

                with timer.time("data_processing"):
                    # Add loss mask and advantages to each message in LLMMessageLogType
                    for i, message_log in enumerate(repeated_batch["message_log"]):
                        for j, message in enumerate(message_log):
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(
                                    message["token_ids"]
                                )
                            else:
                                message["token_loss_mask"] = torch.zeros_like(
                                    message["token_ids"]
                                )
                            if "generation_logprobs" not in message:
                                message["generation_logprobs"] = torch.zeros_like(
                                    message["token_ids"], dtype=torch.float32
                                )

                    # Convert updated LLMMessageLogType to FlatMessagesType for training
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    # Create training data from flattened messages
                    train_data = BatchedDataDict[DistillationLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    # this will be mini-batched inside the policy, so maintain the packed multimodal structure
                    train_data.update(flat_messages.get_multimodal_dict(as_tensors=False))
                    train_data.to("cpu")

                print("▶ Preparing for teacher logprob inference...")
                with timer.time("teacher_logprob_inference_prep"):
                    teacher_policy.prepare_for_lp_inference()

                print("▶ Computing teacher logprobs...")
                with timer.time("teacher_logprob_inference"):
                    teacher_topk = teacher_policy.get_topk_logits(train_data, k=master_config["distillation"]["topk_logits_k"])
                    train_data["teacher_topk_logits"] = teacher_topk["topk_logits"]
                    train_data["teacher_topk_indices"] = teacher_topk["topk_indices"]

                print("▶ Preparing for logprob inference...")
                with timer.time("logprob_inference_prep"):
                    teacher_policy.offload_after_refit()

                print("▶ Preparing for training...")
                with timer.time("training_prep"):
                    teacher_policy.offload_after_refit()
                    student_policy.prepare_for_training()  # set model train and reload optim to GPU
                    POLICY_GENERATION_STALE = True

                print("▶ Training policy...")
                with timer.time("policy_training"):
                    train_results = student_policy.train(train_data, loss_fn)

                # is_last_step = step + 1 == min(
                #     master_config["distillation"]["max_num_steps"], len(dataloader)
                # )
                is_last_step = step + 1 == master_config["distillation"]["max_num_steps"] 

                # Run validation if it's a validation step
                if val_period > 0 and (step + 1) % val_period == 0:
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_student_generation(
                            student_policy, student_generation, colocated_inference
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        student_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=step + 1,
                        master_config=master_config,
                    )
                    student_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, step + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(val_metrics, step + 1, prefix="validation")

                ## Checkpointing
                consumed_samples += master_config["distillation"]["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (step + 1) % master_config["checkpointing"]["save_period"] == 0
                )
                # +1 because step is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    student_policy.prepare_for_training()

                    distillation_save_state["step"] = step + 1
                    if val_metrics is not None:
                        distillation_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in distillation_save_state:
                        del distillation_save_state["val_reward"]
                    distillation_save_state["consumed_samples"] = consumed_samples

                    if master_config["checkpointing"]["metric_name"] is not None:
                        if (
                            master_config["checkpointing"]["metric_name"]
                            not in distillation_save_state
                        ):
                            warnings.warn(
                                f"You asked to save checkpoints based on {master_config['checkpointing']['metric_name']} but the metric is not found in the save state. "
                                "Saving most recent k checkpoints instead."
                            )
                            master_config["checkpointing"]["metric_name"] = None

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {step + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            step + 1, distillation_save_state, master_config
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Logging
            # Log training data
            log_data = {"content": flat_messages["content"]}
            # log_data["teacher_logprobs"] = train_data["teacher_logprobs"].tolist()
            log_data["input_lengths"] = input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(log_data, f"train_data_step{step}.jsonl")

            metrics = {
                "loss": train_results["loss"].numpy(),
                "grad_norm": train_results["grad_norm"].numpy(),
                "mean_prompt_length": repeated_batch["length"].numpy(),
                "total_num_tokens": input_lengths.numpy(),
            }
            metrics.update(train_results["all_mb_metrics"])
            for k, v in metrics.items():
                if k in {
                    "lr",
                    "wd",
                    "reward",
                    "global_valid_seqs",
                    "global_valid_toks",
                    "mean_prompt_length",
                }:
                    metrics[k] = np.mean(v).item()
                else:
                    metrics[k] = np.sum(v).item()
            metrics.update(rollout_metrics)

            timing_metrics: dict[str, float] = timer.get_timing_metrics(reduction_op="sum")  # type: ignore
            # # track example with high token mult prob error above 1.05
            # if metrics["token_mult_prob_error"] > 1.05:
            #     logger.log_plot_token_mult_prob_error(
            #         {
            #             "prompt_lengths": repeated_batch["length"],
            #             "full_lengths": input_lengths,
            #             "generation_logprobs": train_data["generation_logprobs"],
            #             "prev_logprobs": train_data["prev_logprobs"],
            #             "token_mask": train_data["token_mask"],
            #             "sample_mask": train_data["sample_mask"],
            #         },
            #         step + 1,
            #         name="train/token_mult_prob_error_plot_sample",
            #     )

            print("\n📊 Training Results:")

            print(f"  • Loss: {metrics['loss']:.4f}")
            print(
                f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
            )
            if "total_flops" in train_results:
                total_tflops = (
                    train_results["total_flops"] / timing_metrics["policy_training"] / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)"
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%"
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops

            print("\n⏱️  Timing:")
            # Display total time first, separately
            total_time = timing_metrics.get("total_step_time", 0)

            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            metrics.update(
                {
                    "tokens_per_sec_per_gpu": metrics["total_num_tokens"]
                    / total_time
                    / total_num_gpus
                }
            )

            print(f"  • Total step time: {total_time:.2f}s")

            # Display all other timing metrics
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            logger.log_metrics(metrics, step + 1, prefix="train")
            logger.log_metrics(timing_metrics, step + 1, prefix="timing/train")

            timer.reset()
            step += 1
            if step >= max_steps:
                break


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        total_rewards = []
        total_lengths = []
        all_message_logs = []  # Collect all message logs

        max_batches = (
            master_config["distillation"]["max_val_samples"]
            // master_config["distillation"]["val_batch_size"]
        )
        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            # Use async rollouts if vLLM async engine is enabled
            if _should_use_async_rollouts(master_config):
                val_batch, gen_metrics = run_async_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["distillation"]["max_rollout_turns"],
                    greedy=False,
                )
            else:
                val_batch, gen_metrics = run_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=master_config["policy"]["max_total_sequence_length"],
                    max_rollout_turns=master_config["distillation"]["max_rollout_turns"],
                    greedy=False,
                )
            rewards = val_batch["total_reward"]

            total_rewards.extend(rewards.tolist())
            total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

            # Collect message logs for later display
            to_env = [
                get_keys_from_message_log(
                    val_batch["message_log"][i], ["role", "content"]
                )
                for i in range(len(val_batch["message_log"]))
            ]

            all_message_logs.extend(to_env)

        # Calculate validation metrics
        accuracy = sum(total_rewards) / len(total_rewards)
        avg_length = sum(total_lengths) / len(total_lengths)

        val_metrics = {
            "accuracy": accuracy,
            "avg_length": avg_length,
        }

        # Print sample conversations only once at the end of validation
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...")

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}")

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics
