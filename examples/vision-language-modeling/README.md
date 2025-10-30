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

# Vision Language Model Fine-tuning

Fine-tuning the library models for vision language modeling on a image + text dataset.
Gemma3 fine-tuned using a vision language modeling (VLM) loss. You can find more information about the differences between those objectives in our [model summary](https://huggingface.co/transformers/model_summary.html).

The following examples will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own
text files for training and validation. We give examples of both below.

## Requirements

First, you should install the requirements:
```bash
pip install -r requirements.txt
```

## Gemma3 vision language modeling

The following examples fine-tune Gemma3 on ChartQA.


### Single-card Fine-tuning (Gemma3)

```bash

```

### Multi-card Training (Gemma3)

```bash
PT_HPU_LAZY_MODE=0 python ../gaudi_spawn.py \
    --world_size 8 --use_mpi run_lora_vlm.py \
    --model_name_or_path google/gemma-3-12b-it \
    --dataset_name HuggingFaceM4/ChartQA \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --do_train \
    --output_dir ./output_gemma3 \
    --gaudi_config_name Habana/gpt2 \
    --use_habana \
    --use_lazy_mode \
    --lora_rank 16 \
    --max_train_samples 128 \
    --bf16 \    
    --throughput_warmup_steps 3
```
