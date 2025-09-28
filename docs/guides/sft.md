# Supervised Fine-Tuning in NeMo RL

This document explains how to perform SFT within NeMo RL. It outlines key operations, including initiating SFT runs, managing experiment configurations using YAML, and integrating custom datasets that conform to the required structure and attributes.

## Launch an SFT Run

The script, [examples/run_sft.py](../../examples/run_sft.py), can be used to launch an experiment. This script can be launched either locally or via Slurm. For details on how to set up Ray and launch a job using Slurm, refer to the [cluster documentation](../cluster.md).

Be sure to launch the job using `uv`. The command to launch an SFT job is as follows:

```bash
uv run examples/run_sft.py --config <PATH TO YAML CONFIG> <OVERRIDES>
```

If not specified, `config` will default to [examples/configs/sft.yaml](../../examples/configs/sft.yaml).

## Example Configuration File

NeMo RL allows users to configure experiments using `yaml` config files. An example SFT configuration file can be found [here](../../examples/configs/sft.yaml).

To override a value in the config, either update the value in the `yaml` file directly, or pass the override via the command line. For example:

```bash
uv run examples/run_sft.py \
    cluster.gpus_per_node=1 \
    logger.wandb.name="sft-dev-1-gpu"
```

**Reminder**: Don't forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Datasets

SFT datasets in NeMo RL are encapsulated using classes. Each SFT data class is expected to have the following attributes:
  1. `formatted_ds`: The dictionary of formatted datasets. This dictionary should contain `train` and `validation` splits, and each split should conform to the format described below.
  2. `task_spec`: The `TaskDataSpec` for this dataset. This should specify the name you choose for this dataset.

SFT datasets are expected to follow the HuggingFace chat format. Refer to the [chat dataset document](../design-docs/chat-datasets.md) for details. If your data is not in the correct format, simply write a preprocessing script to convert the data into this format. [response_datasets/squad.py](../../nemo_rl/data/datasets/response_datasets/squad.py) has an example:

```python
def format_squad(data):
    return {
        "messages": [
            {
                "role": "system",
                "content": data["context"],
            },
            {
                "role": "user",
                "content": data["question"],
            },
            {
                "role": "assistant",
                "content": data["answers"]["text"][0],
            },
        ]
    }
```

NeMo RL SFT uses HuggingFace chat templates to format the individual examples. Three types of chat templates are supported, which can be configured via `tokenizer.chat_template` in your yaml config (see [sft.yaml](../../examples/configs/sft.yaml) for an example):

1. Apply the tokenizer's default chat template. To use the tokenizer's default, either omit `tokenizer.chat_template` from the config altogether, or set `tokenizer.chat_template="default"`.
2. Use a "passthrough" template which simply concatenates all messages. This is desirable if the chat template has been applied to your dataset as an offline preprocessing step. In this case, you should set `tokenizer.chat_template` to None as follows:
    ```yaml
    tokenizer:
      chat_template: NULL
    ```
3. Use a custom template: If you would like to use a custom template, create a string template in [jinja format](https://huggingface.co/docs/transformers/v4.34.0/en/chat_templating#how-do-i-create-a-chat-template), and add that string to the config. For example,

    ```yaml
    tokenizer:
    custom_template: "{% for message in messages %}{%- if message['role'] == 'system'  %}{{'Context: ' + message['content'].strip()}}{%- elif message['role'] == 'user'  %}{{' Question: ' + message['content'].strip() + ' Answer: '}}{%- elif message['role'] == 'assistant'  %}{{message['content'].strip()}}{%- endif %}{% endfor %}"
    ```

By default, NeMo RL has support for [OpenAssistant](../../nemo_rl/data/datasets/response_datasets/oasst.py), [Squad](../../nemo_rl/data/datasets/response_datasets/squad.py) and [OpenMathInstruct-2](../../nemo_rl/data/datasets/response_datasets/openmathinstruct2.py) datasets. All of these datasets are downloaded from HuggingFace and preprocessed on-the-fly, so there's no need to provide a path to any datasets on disk.

We provide a [ResponseDataset](../../nemo_rl/data/datasets/response_datasets/response_dataset.py) class that is compatible with jsonl-formatted response datasets for loading datasets from local path or HuggingFace. You can use `input_key`, `output_key` to specify which fields in your data correspond to the question and answer respectively. Here's an example configuration:
```yaml
data:
  dataset_name: ResponseDataset
  train_data_path: <PathToTrainingDataset>  # e.g., /path/to/local/dataset.jsonl or hf_org/hf_dataset_name (HuggingFace)
  val_data_path: <PathToValidationDataset>
  input_key: <QuestionKey>, default is "input"
  output_key: <AnswerKey>, default is "output"
  train_split: <TrainSplit>, default is None  # used for HuggingFace datasets
  val_split: <ValSplit>, default is None  # used for HuggingFace datasets
```

### OpenAI Format Datasets (with Tool Calling Support)

NeMo RL also supports datasets in the OpenAI conversation format, which is commonly used for chat models and function calling. This format is particularly useful for training models with tool-use capabilities.

#### Basic Usage

To use an OpenAI format dataset, configure your YAML as follows:

```yaml
data:
  dataset_name: openai_format
  train_data_path: "/path/to/train.jsonl"  # Path to training data
  val_data_path: "/path/to/val.jsonl"      # Path to validation data
  chat_key: "messages"                     # Key for messages in the data (default: "messages")
  system_key: null                         # Key for system message in the data (optional)
  system_prompt: null                      # Default system prompt if not in data (optional)
  tool_key: "tools"                        # Key for tools in the data (default: "tools")
  use_preserving_dataset: false            # Set to true for heterogeneous tool schemas (see below)
```

#### Data Format

Your JSONL files should contain one JSON object per line with the following structure:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": "I'll check the weather for you.", "tool_calls": [
      {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
    ]},
    {"role": "tool", "content": "22°C, sunny", "tool_call_id": "call_123"},
    {"role": "assistant", "content": "The weather in Paris is currently 22°C and sunny."}
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      }
    }
  ]
}
```

#### Tool Calling with Heterogeneous Schemas

When your dataset contains tools with different argument structures (heterogeneous schemas), you should enable `use_preserving_dataset: true` to avoid data corruption:

```yaml
data:
  dataset_name: openai_format
  ...
  use_preserving_dataset: true  # IMPORTANT: Enable this for tool calling datasets
```

**Why this matters:** Standard HuggingFace dataset loading enforces uniform schemas by adding `None` values for missing keys. For example:
- Tool A has arguments: `{"query": "search term"}`
- Tool B has arguments: `{"expression": "2+2", "precision": 2}`

Without `use_preserving_dataset: true`, the loader would incorrectly add:
- Tool A becomes: `{"query": "search term", "expression": None, "precision": None}`
- Tool B becomes: `{"query": None, "expression": "2+2", "precision": 2}`

This corrupts your training data and can lead to models generating invalid tool calls. The `PreservingDataset` mode maintains the exact structure of each tool call.


Adding a new dataset is a straightforward process.
As long as your custom dataset has the `formatted_ds` and `task_spec` attributes described above, it can serve as a drop-in replacement for Squad and OpenAssistant.

## Evaluate the Trained Model

Upon completion of the training process, you can refer to our [evaluation guide](eval.md) to assess model capabilities.
