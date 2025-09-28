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
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

custom_parallel_plan: dict[str, ParallelStyle] = {
    "model.layers.*.self_attn": PrepareModuleInput(
        input_kwarg_layouts={"attention_mask": Replicate()},
        desired_input_kwarg_layouts={"attention_mask": Replicate()},
    ),
    "model.embed_tokens": RowwiseParallel(
        input_layouts=Replicate(), output_layouts=Replicate(), use_local_output=True
    ),
    "model.layers.*.self_attn.q_proj": ColwiseParallel(use_local_output=False),
    "model.layers.*.self_attn.k_proj": ColwiseParallel(use_local_output=False),
    "model.layers.*.self_attn.v_proj": ColwiseParallel(use_local_output=False),
    "model.layers.*.self_attn.o_proj": RowwiseParallel(
        output_layouts=Replicate(), use_local_output=True
    ),
    "model.layers.*.self_attn.rotary_emb": PrepareModuleOutput(
        output_layouts=(Replicate(), Replicate()),
        desired_output_layouts=(Replicate(), Replicate()),
        use_local_output=False,
    ),
    "model.layers.*.mlp.up_proj": ColwiseParallel(),
    "model.layers.*.mlp.gate_proj": ColwiseParallel(),
    "model.layers.*.mlp.down_proj": RowwiseParallel(
        output_layouts=Replicate(), use_local_output=True
    ),
    "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
}
