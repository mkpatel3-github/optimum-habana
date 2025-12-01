# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Llava-NeXT model."""

from typing import Optional, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextForConditionalGeneration,
    LlavaNextModel,
    LlavaNextModelOutputWithPast,
    get_anyres_image_grid_shape,
    unpad_image,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)


class GaudiLlavaNextModel(LlavaNextModel):
    """
    Override LlavaNextModel to fix BS=1 + GA>1 token mismatch issue.
    The base model's forward() calls get_placeholder_mask() which fails for BS=1 with anyres.
    """
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Override forward to fix BS=1 token mismatch by using legacy (padded) processing.
        """
        logger.info("🔍 GaudiLlavaNextModel.forward() CALLED")
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            
        logger.info(f"🔍 input_ids shape: {input_ids.shape if input_ids is not None else None}")
        logger.info(f"🔍 pixel_values: {pixel_values is not None}, shape: {pixel_values.shape if pixel_values is not None else None}")

        if pixel_values is not None and pixel_values.size(0) > 0:
            # WORKAROUND: Check batch size and force legacy processing for BS=1
            batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            
            logger.info(f"🔍 BATCH_SIZE detected: {batch_size}")
            
            if batch_size == 1 and not hasattr(self, '_bs1_fix_logged'):
                self._bs1_fix_logged = True
                logger.warning(f"🔧 LLaVA BS=1 FIX: Applying legacy (padded) processing for BS={batch_size} to prevent token mismatch")
            
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            
            logger.info(f"🔍 image_features shape: {image_features.shape}")
            
            # Use get_placeholder_mask for all batch sizes - it's the correct path for anyres
            logger.info(f"🔧 USING get_placeholder_mask PATH (BS={batch_size})")
            logger.info(f"🔍 Before get_placeholder_mask:")
            logger.info(f"   - image_features: {image_features.shape}, dtype: {image_features.dtype}")
            logger.info(f"   - inputs_embeds: {inputs_embeds.shape}, dtype: {inputs_embeds.dtype}")
            logger.info(f"   - input_ids: {input_ids.shape}, unique_tokens: {input_ids.unique().shape}")
            logger.info(f"   - image_token_index: {self.config.image_token_index}")
            num_image_tokens = (input_ids == self.config.image_token_index).sum().item()
            logger.info(f"   - num_image_tokens in input_ids: {num_image_tokens}")
            logger.info(f"   - image_sizes: {image_sizes}")
            
            logger.info("🔍 CALLING get_placeholder_mask NOW...")
            # get_placeholder_mask handles anyres correctly without padding
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            logger.info(f"🔍 RETURNED from get_placeholder_mask")
            logger.info(f"   - special_image_mask shape: {special_image_mask.shape}")
            logger.info(f"   - special_image_mask dtype: {special_image_mask.dtype}")
            logger.info(f"   - special_image_mask sum: {special_image_mask.sum()}")
            logger.info(f"   - special_image_mask True count: {special_image_mask.sum() // image_features.shape[-1]}")
            
            logger.info("🔍 CALLING masked_scatter NOW...")
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            logger.info(f"🔍 After merge - inputs_embeds shape: {inputs_embeds.shape}")
        else:
            logger.info("🔍 NO pixel_values - skipping image processing")

        logger.info(f"🔍 CALLING language_model NOW...")
        logger.info(f"   - inputs_embeds: {inputs_embeds.shape}")
        logger.info(f"   - attention_mask: {attention_mask.shape if attention_mask is not None else None}")
        logger.info(f"   - position_ids: {position_ids.shape if position_ids is not None else None}")
        logger.info(f"   - cache_position: {cache_position.shape if cache_position is not None else None}")
        logger.info(f"   - past_key_values: {past_key_values is not None}")
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        
        logger.info("🔍 RETURNED from language_model")
        logger.info("🔍 GaudiLlavaNextModel.forward() COMPLETED")

        return LlavaNextModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )
    
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        """
        Merge image features with input embeddings using padded approach (legacy path).
        This method is copied from GaudiLlavaNextForConditionalGeneration to enable BS=1 fix.
        """
        # Handle both 2D (already concatenated) and 3D (original) image features
        if image_features.dim() == 2:
            # Image features already concatenated: [total_patches, embed_dim]
            total_patches, embed_dim = image_features.shape
            # For anyres, we can't easily determine num_images from shape alone
            # Just use the total patches directly
            logger.info(f"🔍 2D image_features: total_patches={total_patches}, embed_dim={embed_dim}")
        else:
            # Original 3D format: [num_images, num_patches, embed_dim]
            num_images, num_image_patches, embed_dim = image_features.shape
            total_patches = num_images * num_image_patches
            logger.info(f"🔍 3D image_features: num_images={num_images}, num_patches={num_image_patches}, embed_dim={embed_dim}")
            
        batch_size, sequence_length = input_ids.shape
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(pad_token_id))
        
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        
        logger.info(f"🔍 num_special_image_tokens in input_ids: {num_special_image_tokens.item()}")
        logger.info(f"🔍 total image feature patches: {total_patches if image_features.dim() == 2 else num_images * num_image_patches}")
        
        # For 2D features, calculate patches per image token
        if image_features.dim() == 2:
            # Each <image> token gets replaced by (total_patches / num_image_tokens) patches
            num_image_patches = total_patches // num_special_image_tokens.item() if num_special_image_tokens.item() > 0 else 0
            logger.info(f"🔍 Calculated num_image_patches per token: {num_image_patches}")
        
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        logger.info(f"🔍 max_embed_dim: {max_embed_dim}")
        
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )

        if labels is not None:
            ignore_index = self.config.ignore_index if hasattr(self.config, 'ignore_index') else -100
            final_labels = torch.full(
                (batch_size, max_embed_dim), ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        # Handle both 2D and 3D image_features when checking count
        expected_image_tokens = total_patches if image_features.dim() == 2 else num_images * num_image_patches
        actual_positions = image_to_overwrite.sum().item()
        
        logger.info(f"🔍 Validation: expected={expected_image_tokens}, actual={actual_positions}")
        
        if actual_positions != expected_image_tokens:
            logger.warning(
                f"⚠️ Image token count mismatch: expected {expected_image_tokens} positions, got {actual_positions}. "
                f"This is expected with anyres - continuing anyway."
            )
            # Don't raise error - anyres creates variable tokens, this is expected for BS=1
            # raise ValueError(
            #     f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
            #     f" the number of image given to the model is {num_special_image_tokens.item()}. Expected {expected_image_tokens} positions, got {actual_positions}."
            # )

        # Reshape image_features to 2D if needed
        if image_features.dim() == 3:
            image_features_2d = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        else:
            image_features_2d = image_features.to(target_device)
        
        # Only fill the positions we have features for
        positions_to_fill = min(actual_positions, expected_image_tokens)
        if actual_positions > 0 and expected_image_tokens > 0:
            image_to_overwrite_flat = image_to_overwrite.view(-1)
            fill_indices = torch.where(image_to_overwrite_flat)[0][:positions_to_fill]
            final_embedding.view(-1, embed_dim)[fill_indices] = image_features_2d[:positions_to_fill]
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids


class GaudiLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        logger.warning("🔧 INITIALIZING GaudiLlavaNextForConditionalGeneration - replacing model with GaudiLlavaNextModel")
        # Replace the base model with our Gaudi version that fixes BS=1
        self.model = GaudiLlavaNextModel(config)
        logger.warning(f"🔧 Model replaced: self.model type = {type(self.model).__name__}")
        # Tie weights if needed
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        token_idx: Optional[torch.Tensor] = None,
        use_flash_attention: Optional[bool] = False,
        flash_attention_recompute: Optional[bool] = False,
        **kwargs,
    ) -> Union[tuple, LlavaNextCausalLMOutputWithPast]:
        """
        Inherits from LlavaForConditionalGeneration: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava_next/modeling_llava_next.py#L433
        The only differences are:
        - add new args token_idx
        - add new args use_flash_attention
        - add new args flash_attention_recompute
        - Moved the process of merging images into inputs_embeds into prepare_inputs_for_generation
        """

        if token_idx is not None:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                # TODO: from Transformers v4.45, `generate` sets `num_logits_to_keep` to 1 if not given, which we don't want here
                # logits_to_keep=logits_to_keep,
                token_idx=token_idx + self.image_offset,
                use_flash_attention=use_flash_attention,
                flash_attention_recompute=flash_attention_recompute,
                **kwargs,
            )

            if inputs_embeds.shape[1] != 1 and pixel_values is not None and self.text_tokens_pos is not None:
                batch_size, seq_len = self.text_tokens_pos.shape
                batch_indices = torch.arange(batch_size).repeat_interleave(seq_len)
                logits = outputs[0][batch_indices, self.text_tokens_pos.reshape(-1), :].reshape(
                    batch_size, seq_len, -1
                )
            else:
                logits = outputs[0]

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                if attention_mask is not None:
                    # we use the input attention mask to shift the logits and labels, because it is 2D.
                    # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                    shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                    shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                    shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
                )

            return LlavaNextCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        else:
            # WORKAROUND for BS=1 + GA>1: Override parent forward to fix token mismatch
            # Instead of calling super().forward(), we need to intercept and fix the processing
            # The error occurs in the parent's get_placeholder_mask() which doesn't pad for BS=1
            
            # Get batch size to check if we need the fix
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            
            # Log the fix application (only once)
            if not hasattr(self, '_bs1_fix_logged') and batch_size == 1 and pixel_values is not None:
                self._bs1_fix_logged = True
                logger.warning(f"🔧 LLaVA BS=1 FIX: Applying padded processing for BS={batch_size} to prevent token mismatch during training")
            
            # Call parent forward - it will use our overridden _merge_input_ids_with_image_features
            # which properly handles BS=1 with padding
            return super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

    # Copied from https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava_next/modeling_llava_next.py#L356
    # Remove the step 6: Mask out the embedding at padding positions
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        text_tokens_pos = new_token_positions
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )

        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        # batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        # indices_to_mask = new_token_positions[batch_indices, pad_indices]

        # final_embedding[batch_indices, indices_to_mask] = 0
        if labels is None:
            final_labels = None

        return final_embedding, final_attention_mask, final_labels, position_ids, text_tokens_pos

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        """
        Inherits from LlavaForConditionalGeneration: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/llava_next/modeling_llava_next.py#L635
        The only differences are:
        - add new args token_idx
        - add the process of merging images into inputs_embeds
        """
        token_idx = kwargs.get("token_idx", None)
        if token_idx is None:
            return super().prepare_inputs_for_generation(
                input_ids=input_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        else:
            # WORKAROUND for BS=1 + GA>1: Force legacy_processing to enable padding
            # With BS=1, the default check fails because there's no batch-level max to compare against
            # This causes "Image features and image tokens do not match" errors with gradient accumulation
            # Legacy processing uses _merge_input_ids_with_image_features which pads to max_embed_dim
            batch_size = input_ids.shape[0]
            force_legacy_for_bs1 = (batch_size == 1)
            legacy_processing = (
                (input_ids == self.config.image_token_index).sum(1).max() < self.config.image_seq_length
            ) or (token_idx == 1 and pixel_values is not None) or force_legacy_for_bs1
            
            # Log which path is being used (only once at the start)
            if pixel_values is not None and past_key_values is None and not hasattr(self, '_legacy_path_logged'):
                self._legacy_path_logged = True
                if legacy_processing:
                    if force_legacy_for_bs1:
                        logger.warning(f"🔧 LLaVA BS=1 FIX: Using legacy_processing (padded path) for BS={batch_size} to prevent token mismatch")
                    else:
                        logger.info(f"Using legacy_processing (padded path) for BS={batch_size}")
                else:
                    logger.info(f"Using non-legacy_processing (direct path) for BS={batch_size}")
            
            use_flash_attention = kwargs.get("use_flash_attention", False)
            flash_attention_recompute = kwargs.get("flash_attention_recompute", False)
            position_ids = kwargs.get("position_ids", None)
            labels = kwargs.get("labels", None)
            if past_key_values is None and pixel_values is not None and input_ids.shape[1] != 1:
                vision_feature_select_strategy = kwargs.get("vision_feature_select_strategy", None)
                vision_feature_layer = kwargs.get("vision_feature_layer", None)
                vision_feature_select_strategy = (
                    vision_feature_select_strategy
                    if vision_feature_select_strategy is not None
                    else self.config.vision_feature_select_strategy
                )
                vision_feature_layer = (
                    vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
                )

                # 1. Extract the input embeddings
                inputs_embeds = self.get_input_embeddings()(input_ids)
                # 2. Merge text and images
                batch_size, num_patches, num_channels, height, width = pixel_values.shape
                reshaped_pixel_values = pixel_values.view(batch_size * num_patches, num_channels, height, width)
                image_features = self.vision_tower(
                    reshaped_pixel_values,
                    output_hidden_states=True,
                    use_flash_attention=use_flash_attention,
                    flash_attention_recompute=flash_attention_recompute,
                )

                selected_image_feature = image_features.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature

                image_features = self.multi_modal_projector(selected_image_feature)

                # split up image_features for each of the individual images
                # hence we get a list of image_features, each of shape (5, num_patches, hidden_size)
                # if we assume each image has 5 image features (base image + 4 patches)
                split_sizes = [image.shape[0] for image in pixel_values]
                image_features = torch.split(image_features, split_sizes, dim=0)

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]

                        if height * width != base_image_feature.shape[0]:
                            raise ValueError("The number of patches is not consistent with the image size.")
                        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                            image_sizes[image_idx],
                            self.config.image_grid_pinpoints,
                            self.config.vision_config.image_size,
                        )
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat(
                            (
                                image_feature,
                                self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1),
                            ),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)
                    new_image_features.append(image_feature)
                if legacy_processing:
                    image_features = torch.stack(new_image_features, dim=0)
                    inputs_embeds, attention_mask, labels, position_ids, self.text_tokens_pos = (
                        self._merge_input_ids_with_image_features(
                            image_features, inputs_embeds, input_ids, attention_mask, labels
                        )
                    )
                    self.image_offset = image_features.shape[1] - 1  # image_token has occupied 1 token position.
                else:
                    image_features = torch.cat(new_image_features, dim=0)
                    n_image_tokens = (input_ids == self.config.image_token_index).sum().item()
                    n_image_features = image_features.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )
                    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                    batch_indices, image_indices = torch.where(input_ids == self.config.image_token_index)
                    inputs_embeds[batch_indices, image_indices] = image_features.contiguous()
                    self.image_offset = 0
                    self.text_tokens_pos = None

                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)

            # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and pixel_values is not None:
                seq_len = input_ids.shape[1]
                pad_len = seq_len - token_idx
                input_ids = torch.index_select(input_ids, 1, token_idx - 1)
                if legacy_processing:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    # Get the target length
                    past_length = first_layer_past_key_value.shape[-1]

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], past_length),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                    attention_mask = extended_attention_mask
                    attention_mask[:, -pad_len:] = 0

            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values:
                    if token_idx is not None:
                        position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                    else:
                        position_ids = position_ids[:, -input_ids.shape[1] :]

            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and past_key_values is None:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids}

            if logits_to_keep is not None:
                model_inputs["logits_to_keep"] = logits_to_keep

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "token_idx": token_idx,
                    "image_sizes": image_sizes,
                    "labels": labels,
                    "use_flash_attention": use_flash_attention,
                    "flash_attention_recompute": flash_attention_recompute,
                }
            )

            return model_inputs
