#!/usr/bin/env python
# coding=utf-8
# Apache v2 license
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

"""
Fine-tuning vision-language models (e.g., Gemma3 VLM) on Gaudi HPUs with LoRA.

This script supports multimodal datasets with image and text inputs for instruction tuning.
It handles vision-text preprocessing, assistant-only label masking, and LoRA fine-tuning.

Key features:
- Subset dataset BEFORE normalization for efficiency
- Robust assistant-only labels via prompt-length masking
- GaudiConfig support via gaudi_config_name in training args
- Single and multi-card training via gaudi_spawn.py
"""

import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry

from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
from optimum.habana.utils import set_seed

try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


IGNORE_INDEX = -100

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

check_min_version("4.49.0")
check_optimum_habana_min_version("1.18.0.dev0")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/processor we are going to fine-tune.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub. "
                "This option should only be set to `True` for repositories you trust and in which you have read the "
                "code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "Setting it to True will benefit LLM loading time and RAM consumption."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set. Samples are selected BEFORE normalization."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    sample_strategy: str = field(
        default="first",
        metadata={"help": "How to select subset of samples: 'first' or 'random'"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    save_last_ckpt: bool = field(
        default=True,
        metadata={"help": "Whether to save the last checkpoint at the end of training."},
    )


@dataclass
class LoRAArguments:
    """
    Arguments for LoRA configuration.
    """

    lora_rank: int = field(
        default=16,
        metadata={"help": "LoRA rank"},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter"},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"},
    )
    lora_target_modules: Optional[str] = field(
        default="all-linear",
        metadata={"help": "Target modules for LoRA (comma-separated or 'all-linear')"},
    )


def to_1d_ids(obj, processor):
    """
    Normalize whatever apply_chat_template returns into a 1-D LongTensor (T,).
    Handles: Tensor (T) / (1,T), BatchEncoding, dict, list/np, str.
    """
    if hasattr(obj, "input_ids"):
        t = obj.input_ids
    elif isinstance(obj, dict) and "input_ids" in obj:
        t = obj["input_ids"]
    else:
        if isinstance(obj, torch.Tensor):
            t = obj
        elif isinstance(obj, str):
            t = processor.tokenizer(obj, return_tensors="pt").input_ids
        else:
            t = torch.as_tensor(obj)

    if isinstance(t, torch.Tensor):
        if t.ndim == 2:
            t = t.squeeze(0)
        elif t.ndim > 2:
            t = t.view(-1)
    else:
        t = torch.as_tensor(t)
        if t.ndim == 2:
            t = t.squeeze(0)

    return t.long()


def normalize_split(ds):
    """
    Normalize different dataset schemas to a common format: {image, question, answer}
    """

    def _map(ex):
        img = ex.get("image")
        if img is None:
            raise ValueError("Missing 'image' column.")

        if "query" in ex and "label" in ex:
            q = (ex["query"] or "").strip()
            lab = ex["label"]
            a = lab[0] if isinstance(lab, list) and lab else ""
        elif "question" in ex and "answer" in ex:
            q = (ex["question"] or "").strip()
            a = (ex["answer"] or "").strip()
        else:
            raise ValueError("Unknown schema; expected ['query','label'] or ['question','answer'].")

        return {"image": img.convert("RGB"), "question": q, "answer": a}

    return ds.map(_map, remove_columns=ds.column_names, desc="Normalize dataset schema")


def process_vision_info(messages):
    """
    Extract images from message content for vision-language processing.
    """
    imgs = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
        for elem in content:
            if isinstance(elem, dict) and ("image" in elem or elem.get("type") == "image"):
                im = elem.get("image", elem)
                imgs.append(im.convert("RGB"))
    return imgs


def make_collate_fn(processor):
    """
    Create a collate function for batching vision-language examples.
    Implements assistant-only label masking by measuring prompt length.
    """
    tok = processor.tokenizer
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pad_id = tok.pad_token_id
    boi_id = tok.convert_tokens_to_ids(tok.special_tokens_map.get("boi_token", ""))
    eoi_id = tok.convert_tokens_to_ids(tok.special_tokens_map.get("eoi_token", ""))
    LEGACY_IMAGE_TOKEN_ID = 262144

    def collate_fn(batch):
        full_texts, full_images, prompt_ids_list = [], [], []

        for ex in batch:
            q = ex["question"].strip()
            a = (ex.get("answer") or "").strip()

            # Full conversation (user + assistant)
            full_msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex["image"]},
                        {"type": "text", "text": q},
                    ],
                }
            ]
            if a:
                full_msgs.append({"role": "assistant", "content": [{"type": "text", "text": a}]})

            full_texts.append(
                processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False).strip()
            )
            full_images.append(process_vision_info(full_msgs))

            # Prompt (user only) for label masking
            prompt_msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex["image"]},
                        {"type": "text", "text": q},
                    ],
                }
            ]

            # Force batch-of-1 to stabilize return type across versions
            prompt_enc = processor.apply_chat_template(
                [prompt_msgs], add_generation_prompt=True, tokenize=True, return_tensors="pt"
            )

            p_ids = to_1d_ids(prompt_enc, processor)
            prompt_ids_list.append(p_ids)

        enc = processor(text=full_texts, images=full_images, return_tensors="pt", padding=True)

        # Assistant-only labels
        labels = enc["input_ids"].clone()
        labels[labels == pad_id] = IGNORE_INDEX
        if isinstance(boi_id, int) and boi_id != tok.unk_token_id:
            labels[labels == boi_id] = IGNORE_INDEX
        if isinstance(eoi_id, int) and eoi_id != tok.unk_token_id:
            labels[labels == eoi_id] = IGNORE_INDEX
        labels[labels == LEGACY_IMAGE_TOKEN_ID] = IGNORE_INDEX

        # Mask prompt by measured length
        for i, p_ids in enumerate(prompt_ids_list):
            labels[i, : p_ids.numel()] = IGNORE_INDEX

        enc["labels"] = labels
        return enc

    return collate_fn


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, LoRAArguments, GaudiTrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, lora_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # CRITICAL: Disable unused column removal so collate_fn can access image, question, answer
    training_args.remove_unused_columns = False

    send_example_telemetry("run_vlm_lora", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, "
        f"bf16 training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Discover distributed rank/world size (gaudi_spawn.py sets RANK and WORLD_SIZE)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    # Load dataset
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        raise ValueError("You must specify a dataset_name")

    if "train" not in raw_datasets:
        raise ValueError("Training requires a train split")

    train_dataset = raw_datasets["train"]

    # Subset dataset BEFORE normalization (global subset so all ranks agree on same N examples)
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        if data_args.sample_strategy == "random":
            train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(max_train_samples))
        else:
            train_dataset = train_dataset.select(range(max_train_samples))

        if is_main_process(training_args.local_rank):
            logger.info(
                f"Using {len(train_dataset)} training samples (strategy={data_args.sample_strategy}, seed={training_args.seed})"
            )

    # Shard dataset for distributed training (each rank processes its own shard)
    if world_size > 1:
        train_dataset = train_dataset.shard(num_shards=world_size, index=rank)
        logger.info(f"[rank {rank}/{world_size}] Sharded train size: {len(train_dataset)}")

    # Normalize dataset schema
    train_dataset = normalize_split(train_dataset)

    # Filter out examples with empty answers
    train_dataset = train_dataset.filter(
        lambda ex: isinstance(ex["answer"], str) and ex["answer"].strip() != ""
    )

    if is_main_process(training_args.local_rank):
        logger.info(f"[rank {rank}] Final train shard size after normalize+filter: {len(train_dataset)}")

    # Load processor and model
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )

    model.config.use_cache = False

    # LoRA configuration
    target_modules = lora_args.lora_target_modules
    if target_modules != "all-linear" and isinstance(target_modules, str):
        target_modules = [s.strip() for s in target_modules.split(",") if s.strip()]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    lora_model = get_peft_model(model, peft_config)

    if training_args.bf16:
        lora_model = lora_model.to(torch.bfloat16)

    lora_model.print_trainable_parameters()

    # Data collator
    collate_fn = make_collate_fn(processor)

    # Load Gaudi configuration
    gaudi_config = GaudiConfig.from_pretrained(
        training_args.gaudi_config_name or "Habana/gpt2",
    )

    # Derive max_steps from sample count if user didn't set it
    if training_args.max_steps == -1 and data_args.max_train_samples:
        eff_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        steps_per_epoch = math.ceil(data_args.max_train_samples / max(1, eff_bs))
        training_args.max_steps = int(steps_per_epoch * max(1.0, training_args.num_train_epochs))
        logger.info(
            f"Derived max_steps={training_args.max_steps} from N={data_args.max_train_samples}, "
            f"bs={training_args.per_device_train_batch_size}, ga={training_args.gradient_accumulation_steps}, "
            f"epochs={training_args.num_train_epochs}"
        )

    # Initialize Trainer
    trainer = GaudiTrainer(
        model=lora_model,
        gaudi_config=gaudi_config,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        processing_class=processor,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if data_args.save_last_ckpt:
            trainer._save_checkpoint(trainer.model, None)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Save processor
    if training_args.do_train:
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

