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

import json
import warnings
from typing import Any, Callable, Union

from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


class PreservingDataset:
    """A dataset wrapper that preserves original dict structure without None-filling.

    Unlike HuggingFace's Dataset class which enforces schema uniformity across all samples
    (filling missing keys with None), this class maintains the exact structure of each sample.
    This is critical for heterogeneous data like tool calls where different samples may have
    different argument structures.
    """

    def __init__(self, data: list[dict[str, Any]]):
        """Initialize the dataset with a list of dictionaries.

        Args:
            data: List of dictionary samples, each can have different keys
        """
        self.data = data
        self.features = None  # For compatibility with HF Dataset interface

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: Union[int, slice, list]
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """Support integer indexing, slicing, and list indexing."""
        if isinstance(idx, slice):
            return [self.data[i] for i in range(*idx.indices(len(self.data)))]
        elif isinstance(idx, int):
            # Handle negative indices
            if idx < 0:
                idx = len(self.data) + idx
            if idx < 0 or idx >= len(self.data):
                raise IndexError(
                    f"Index {idx} out of range for dataset of size {len(self.data)}"
                )
            return self.data[idx]
        elif isinstance(idx, list):
            return [self.data[i] for i in idx]
        else:
            raise TypeError(
                f"Indices must be integers, slices, or lists, not {type(idx)}"
            )

    def __iter__(self):
        return iter(self.data)

    def map(self, function: Callable, *args, **kwargs) -> "PreservingDataset":
        """Apply a function to each sample in the dataset.

        Args:
            function: Function to apply to each sample
            with_indices: If True, pass index as second argument to function

        Returns:
            New PreservingDataset with transformed samples
        """
        if kwargs.get("with_indices", False):
            mapped_data = [function(item, i) for i, item in enumerate(self.data)]
        else:
            mapped_data = [function(item) for item in self.data]
        return PreservingDataset(mapped_data)


class OpenAIFormatDataset:
    """This class is used to load an SFT dataset in the OpenAI format.

    The dataset should be in the following format:
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
    }

    Args:
        train_ds_path: Path to the training dataset JSON file
        val_ds_path: Path to the validation dataset JSON file
        chat_key: Key for the messages list in the dataset (default: "messages")
        system_key: Optional key for system prompt in the dataset
        system_prompt: Optional system prompt to add if not in the dataset
        tool_key: Key for tools in the dataset (default: "tools")
        use_preserving_dataset: If True, uses PreservingDataset to maintain
            heterogeneous schemas (e.g., for tool calls with varying argument
            structures). If False, uses standard HuggingFace dataset loading.
            Default is False for backward compatibility.

    Notes:
        - system_key and system_prompt are optional. If provided, it will be added
          to the beginning of the dataset.
        - chat_key should be the key of the messages list. Multi-turn conversations
          are supported.
        - The last message in the conversation must be from the assistant.
        - When use_preserving_dataset=True, the dataset preserves the exact structure
          of each sample without None-filling for missing keys, which is useful for
          heterogeneous tool argument schemas.
    """

    def __init__(
        self,
        train_ds_path: str,
        val_ds_path: str,
        chat_key: str = "messages",
        system_key: str | None = None,
        system_prompt: str | None = None,
        tool_key: str | None = "tools",
        use_preserving_dataset: bool = False,
    ):
        self.chat_key = chat_key
        self.system_key = system_key
        self.system_prompt = system_prompt
        self.tool_key = tool_key

        if not use_preserving_dataset:
            # Use the standard HuggingFace approach (faster and more standard)
            train_original_dataset = load_dataset("json", data_files=train_ds_path)[
                "train"
            ]
            val_original_dataset = load_dataset("json", data_files=val_ds_path)["train"]

            formatted_train_dataset = train_original_dataset.map(self.add_messages_key)
            formatted_val_dataset = val_original_dataset.map(self.add_messages_key)

            print(
                f"Loaded dataset using standard approach (train: {len(formatted_train_dataset)}, val: {len(formatted_val_dataset)})"
            )

            # Warn if tools are present in the dataset
            if self.tool_key and any(
                self.tool_key in sample for sample in formatted_train_dataset
            ):
                warnings.warn(
                    "Tools detected in dataset. Set use_preserving_dataset=True to preserve heterogeneous tool schemas. "
                    "Current mode may add None values for missing tool arguments, making samples invalid.",
                    UserWarning,
                    stacklevel=2,
                )

        else:
            # Use custom loading for heterogeneous schemas
            # Issue: When tool calls have varying argument structures across samples,
            # HuggingFace's Dataset.from_list enforces uniform schema by adding None
            # values for missing keys. Example:
            #   Sample 1: {"tools": [{"name": "search", "args": {"query": "x"}}]}
            #   Sample 2: {"tools": [{"name": "calc", "args": {"expr": "y", "precision": 2}}]}
            # Standard loading would add "precision: None" to Sample 1 and "query: None" to Sample 2.
            # PreservingDataset maintains exact structure without None-filling.
            print(
                "Using PreservingDataset to preserve heterogeneous tool argument schemas without None-filling."
            )

            # Load JSON files directly
            with open(train_ds_path, "r") as f:
                train_data = [json.loads(line) for line in f]

            with open(val_ds_path, "r") as f:
                val_data = [json.loads(line) for line in f]

            # Apply transformations
            formatted_train_data = [self.add_messages_key(item) for item in train_data]
            formatted_val_data = [self.add_messages_key(item) for item in val_data]

            # Use PreservingDataset to maintain exact structure
            formatted_train_dataset = PreservingDataset(formatted_train_data)
            formatted_val_dataset = PreservingDataset(formatted_val_data)

            print(
                f"Loaded dataset using PreservingDataset (train: {len(formatted_train_dataset)}, val: {len(formatted_val_dataset)})"
            )

        self.formatted_ds = {
            "train": formatted_train_dataset,
            "validation": formatted_val_dataset,
        }

        self.task_spec = TaskDataSpec(
            "json_dataset",
        )

    def add_messages_key(
        self,
        example: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        messages = [message for message in example[self.chat_key]]
        if self.system_key is not None and self.system_key in example:
            messages = [
                {"role": "system", "content": example[self.system_key]}
            ] + messages
        elif self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}] + messages
        assert messages[-1]["role"] == "assistant"

        # Preserve tools if they exist in the data
        result = {"messages": messages}
        if self.tool_key and self.tool_key in example:
            result["tools"] = example[self.tool_key]

        return result
