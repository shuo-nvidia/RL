## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import random
import zipfile
from pathlib import Path
from typing import Any, Optional, Union

import requests
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm  # Using tqdm for progress bar, install with: pip install tqdm

from nemo_rl.data.hf_datasets.clevr import pil_to_base64
from nemo_rl.data.interfaces import TaskDataSpec


def download_and_unzip(url: str, target_directory: str, subdir_name: str = "."):
    """Downloads a zip file from a given URL to a target directory and unzips it into a specified subdirectory within the target directory, showing download progress.

    Args:
        url (str): The URL of the zip file to download.
        target_directory (str): The directory where the zip file will be downloaded
                                and unzipped.
        subdir_name (str): The name of the subdirectory within the target_directory
                           where the contents of the zip file will be unzipped.
                           Defaults to "train".
    """
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"Created target directory: {target_directory}")

    # Extract filename from URL
    filename = url.split("/")[-1]
    filepath = os.path.join(target_directory, filename)

    # Download the file with progress
    if not os.path.exists(filepath):
        print(f"Downloading {filename} from {url} to {filepath}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size_in_bytes = int(r.headers.get("content-length", 0))
                block_size = 8192  # 8 Kibibytes

                # Initialize tqdm progress bar
                progress_bar = tqdm(
                    total=total_size_in_bytes, unit="iB", unit_scale=True
                )

                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
                progress_bar.close()  # Close the progress bar

            print(f"Download complete: {filepath}")
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Error downloading file: {e}")
    else:
        print(f"File {filepath} already exists, skipping download.")

    # Define the unzipping directory
    unzip_dir = os.path.join(target_directory, subdir_name)
    if not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir)
        print(f"Created unzip directory: {unzip_dir}")

    # Unzip the file
    print(f"Unzipping {filepath} to {unzip_dir}...")
    try:
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            # You can add a progress bar for unzipping as well, but it's more complex
            # as zipfile doesn't directly expose progress for extractall.
            # For large files, consider iterating through namelist and extracting one by one.
            zip_ref.extractall(unzip_dir)
        print("Unzipping complete.")
    except zipfile.BadZipFile:
        raise zipfile.BadZipFile(f"Error: {filepath} is not a valid zip file.")
    except Exception as e:
        raise Exception(f"Error unzipping file: {e}")


def format_refcoco_dataset(
    example: dict[str, Any],
    width: int = 256,
    height: int = 256,
    caption_type: str = "random",
    prompt_file: Optional[str] = None,
) -> dict[str, Any]:
    """Format the RefCOCO dataset from huggingface.

    This should be replaced with our own curated RefCOCO/+/g dataset soon

    Args:
        example: The example to format.
        width: The width of the resized image.
        height: The height of the resized image.
        caption_type: The type of caption to use.
    """
    split = example["split"]
    if "val" in split:
        caption_type = "descriptive"

    # resize image for easy image processing across batches
    image = Image.open(example["image_path"])
    orig_width, orig_height = image.size
    resized_image = image.resize((width, height))

    # get caption from many types
    if caption_type == "random":
        caption = random.choice(example["captions"])
    elif caption_type == "first":
        caption = example["captions"][0]
    elif caption_type == "descriptive":  # choose the most descriptive caption
        caption = max(example["captions"], key=lambda x: len(x))
    elif caption_type == "brief":  # choose the briefest caption
        caption = min(example["captions"], key=lambda x: len(x))
    elif caption_type == "all":
        caption = " or ".join(example["captions"])
    else:
        raise ValueError(f"Invalid caption type: {caption_type}")

    # get normalized bounding box coordinates (top-left, bottom-right)
    bbox = example["bbox"]
    bbox = [
        bbox[0] / orig_width * 1000,
        bbox[1] / orig_height * 1000,
        bbox[2] / orig_width * 1000,
        bbox[3] / orig_height * 1000,
    ]
    bbox = [int(round(coord)) for coord in bbox]
    solution = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"

    user_content = [
        {
            "type": "image",
            "image": pil_to_base64(resized_image),
        },
        {
            "type": "text",
            "text": f"Please provide the bounding box coordinate of the region described by the following phrase: {caption}",
        },
    ]

    ret = {
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": solution,
            },
        ],
        "task_name": "refcoco",
    }
    return ret


# contain different variants of the CLEVR dataset
def prepare_refcoco_dataset(
    split: str = "default",
    task_name: Optional[str] = None,
    path_to_coco_images: Optional[Union[str, Path]] = None,
):
    if task_name is None:
        task_name = "refcoco"

    tr_dataset = load_dataset("jxu124/refcoco")["train"]
    val_dataset = load_dataset("jxu124/refcoco")["validation"]

    # format - disable features to avoid schema conflicts
    tr_dataset = tr_dataset.add_column("task_name", [task_name] * len(tr_dataset))
    val_dataset = val_dataset.add_column("task_name", [task_name] * len(val_dataset))

    if path_to_coco_images is None:
        print("No path to coco images provided, downloading images to ./coco_images")
        path_to_coco_images = Path("./coco_images")
        os.makedirs(path_to_coco_images, exist_ok=True)
    else:
        path_to_coco_images = Path(path_to_coco_images)

    # check for images
    if not os.path.exists(str(path_to_coco_images / "train2014")):
        print(f"Downloading train2014 images to {path_to_coco_images}")
        download_and_unzip(
            "http://images.cocodataset.org/zips/train2014.zip", str(path_to_coco_images)
        )
    if not os.path.exists(str(path_to_coco_images / "val2014")):
        print(f"Downloading val2014 images to {path_to_coco_images}")
        download_and_unzip(
            "http://images.cocodataset.org/zips/val2014.zip", str(path_to_coco_images)
        )

    # add image column
    tr_dataset = tr_dataset.map(
        lambda example: {
            **example,
            "image_path": str(example["image_path"]).replace(
                "coco/", str(path_to_coco_images) + "/"
            )
            if "image_path" in example
            else example.get("image_path"),
        }
    )
    val_dataset = val_dataset.map(
        lambda example: {
            **example,
            "image_path": str(example["image_path"]).replace(
                "coco/", str(path_to_coco_images) + "/"
            )
            if "image_path" in example
            else example.get("image_path"),
        }
    )

    return {
        "train": tr_dataset,
        "validation": val_dataset,
    }


class RefCOCODataset:
    def __init__(
        self,
        split: str = "default",
        prompt_file: Optional[str] = None,
        download_dir: Optional[str] = None,
    ):
        """Simple wrapper around the RefCOCO dataset.

        Args:
            split: The split of the dataset to use (currently only 'default' is supported)
            prompt_file: The file containing the prompt for the dataset.
        """
        VALID_SPLITS = ["default"]
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {split}. Please use one of {VALID_SPLITS}."
            )
        self.task_name = "refcoco"

        self.formatted_ds = prepare_refcoco_dataset(
            split=split,
            task_name=self.task_name,
            path_to_coco_images=download_dir,
        )
        self.task_spec = TaskDataSpec(
            task_name="RefCOCO",
            prompt_file=prompt_file,
        )
