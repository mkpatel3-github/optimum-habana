<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Vision-Language Model Fine-tuning with LoRA

Fine-tuning vision-language models using Low-Rank Adaptation (LoRA) on Intel Gaudi HPUs. This approach provides parameter-efficient training of multimodal models while maintaining high performance. The script handles vision-text preprocessing, automatic dataset normalization, and supports both single-card and distributed multi-card training.

Examples can run on datasets hosted on the [Hugging Face Hub](https://huggingface.co/datasets) or your own vision-text datasets. The framework supports datasets with image-text pairs in various formats and automatically handles train/validation splitting.

## Requirements

First, install the requirements:

```bash
pip install -r requirements.txt
```

## Fine-tuning

### Single Card (Lazy Mode)

```bash
PT_HPU_LAZY_MODE=1 python run_lora_vlm.py \
    --model_name_or_path google/gemma-3-12b-it \
    --dataset_name HuggingFaceM4/ChartQA \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --output_dir ./output_gemma3 \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --lora_rank 16 \
    --max_train_samples 128 \
    --overwrite_output_dir \
    --bf16 \
    --use_lazy_mode \
    --gradient_checkpointing
```

### Multi-Card (8 HPUs)

```bash
PT_HPU_LAZY_MODE=1 python ../gaudi_spawn.py \
    --world_size 8 \
    --use_mpi \
    run_lora_vlm.py \
    --model_name_or_path google/gemma-3-12b-it \
    --dataset_name HuggingFaceM4/ChartQA \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --output_dir ./output_gemma3 \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --lora_rank 16 \
    --max_train_samples 4096 \
    --overwrite_output_dir \
    --bf16 \
    --logging_steps 50 \
    --log_level info \
    --save_steps 50 \
    --report_to tensorboard \
    --do_eval \
    --gradient_checkpointing
```

## Key Features

- **LoRA Fine-tuning**: Parameter-efficient training of vision-language models
- **Multi-modal Support**: Image and text input preprocessing with automatic schema detection
- **Assistant-only Label Masking**: Trains only on assistant responses, not prompts
- **Distributed Training**: Support for single-card and multi-card (8 HPUs) setups
- **Flexible Execution**: Both Lazy and Eager execution modes
- **Automatic Train/Val Split**: Auto-splits datasets without validation split
- **Evaluation**: Computes perplexity and validation metrics

## Supported Models

- Gemma3 VLM (google/gemma-3-12b-it)
- Other vision-language models may work with minor adjustments

## Tips for Memory/Performance

**OOM Issues**: Use eager mode and reduce batch size
```bash
PT_HPU_LAZY_MODE=0 --per_device_train_batch_size 1
```

**Lower Throughput**: Use lazy mode for better performance
```bash
PT_HPU_LAZY_MODE=1 --use_lazy_mode
```

**Effective Batch Size**: `batch_size × num_cards × gradient_accumulation_steps`

