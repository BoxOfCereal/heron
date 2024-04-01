# Copyright 2023 Turing Inc. Team. All rights reserved.
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


from base64 import b64decode
from io import BytesIO

import cv2
import requests
import datasets
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

from .base_datasets import IGNORE_INDEX, ResilientDataset

# added
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import base64

def load_image(input_image):
    """
    Load an image from various sources such as URL, file path, base64 string, PIL Image object,
    or numpy array (tensor).

    Parameters:
        input_image (str, bytes, PIL.Image.Image, numpy.ndarray): The input image. It can be
            a URL, file path, base64 string, PIL Image object, or numpy array (tensor) in channels,
            height, width format.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.

    Raises:
        ValueError: If the input type is not supported or the shape of the image tensor is invalid.
    """
    if isinstance(input_image, str):
        # If input is a string, check if it's a URL or file path
        if input_image.startswith(('http://', 'https://')):
            # Load image from URL
            response = requests.get(input_image, stream=True)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # Load image from file path
            image = Image.open(input_image).convert("RGB")
    elif isinstance(input_image, bytes):
        # If input is bytes, assume it's a base64 string
        image_data = BytesIO(input_image)
        image = Image.open(image_data).convert("RGB")
    elif isinstance(input_image, Image.Image):
        # If input is already an image object, use it directly
        image = input_image.convert("RGB")
    elif isinstance(input_image, np.ndarray):
        # If input is a numpy array (tensor), assume it's in channels, height, width format
        if input_image.ndim == 3 and input_image.shape[2] in [1, 3]:
            # Convert to PIL image
            image = Image.fromarray(input_image)
        else:
            raise ValueError("Unsupported shape for image tensor. Expected (height, width, channels) or (height, width).")
    else:
        raise ValueError("Unsupported input type. Supported types are URL, file path, base64 string, PIL Image object, or numpy array (tensor).")
    
    return np.array(image)

HFProcessor = "HFProcessor"


class VGDatasets(ResilientDataset):
    """Dataset for M3IT Dataset learning"""

    def __init__(
        self,
        loaded_dataset: ConcatDataset,
        processor: HFProcessor,
        max_length: int,
        is_inference: bool = False,
    ):
        super(VGDatasets, self).__init__(is_inference)
        self.loaded_dataset = loaded_dataset
        self.max_length = max_length
        self.processor = processor
        self.is_inference = is_inference

    @classmethod
    def create(
        cls,
        dataset_config: dict,
        processor: HFProcessor,
        max_length: int,
        split: str = "train",
        is_inference: bool = False,
    ):
        # dataset_list = [
        #     datasets.load_dataset("MMInstruction/M3IT", i, num_proc=16)
        #     for i in dataset_config["dataset_names"]
        # ]
        dataset_list = [
            datasets.load_dataset(name, num_proc=16)
            for name in dataset_config["dataset_names"]
        ]

        # some dataset have no validation
        target_dataset_list = []
        for d in dataset_list:
            try:
                target_dataset_list.append(d[split])
            except KeyError:
                print(f"{d['train']._info.config_name} has no {split} set.")
        target_dataframe = ConcatDataset(target_dataset_list)

        return cls(target_dataframe, processor, max_length, is_inference)

    def preprocess_image(self, images):
        return self.processor(images=images, return_tensors="pt")["pixel_values"][0]

    def tokenize(self, text):
        if self.is_inference:
            kwargs = {}
        else:
            kwargs = {"padding": "max_length", "max_length": self.max_length, "truncation": True}
        return self.processor.tokenizer(text=text, return_tensors="pt", **kwargs)

    def __len__(self) -> int:
        return len(self.loaded_dataset)

    def _get_item_train(self, index):
         # cf: https://huggingface.co/datasets/ArthurFischel/smw_10k
        row = self.loaded_dataset[index]

        # imageのロード
        image_url = row["img_url"]  # url str
        # load image from url
        image = load_image(image_url)
        image = np.array(image)
        images = [image]
        prompt = row["prompt"]

        tokenized = self.tokenize(prompt)
        tokenized_prompt = tokenized["input_ids"][0]
        labels = torch.full_like(tokenized_prompt, IGNORE_INDEX)
        prompt_attn_mask = tokenized["attention_mask"][0]

        index_ignore_loss = prompt_attn_mask.sum().item() + 1
        labels[:index_ignore_loss] = tokenized_prompt[:index_ignore_loss]

        return_dict = {
            "input_ids": tokenized_prompt,
            "labels": labels,
            "attention_mask": prompt_attn_mask,
            "pixel_values": self.preprocess_image(images),
        }
        return return_dict

    def _get_item_inference(self, index):
        """I don't know where this function gets called"""
        # cf: https://huggingface.co/datasets/ArthurFischel/smw_10k
        row = self.loaded_dataset[index]

        text = row["prompt"]  # str

       # imageのロード
        # Right now there is no Error handling
        image_url = row["img_url"]  # url str
        # load image from url
        image = load_image(image_url)
        image = np.array(image)
        images = [image]

        inputs = self.processor(
            text,
            images,
            return_tensors="pt",
        )
        inputs["labels"] = None
        return inputs, image, #answer
