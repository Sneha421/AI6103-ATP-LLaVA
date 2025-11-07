"""
ATP-enabled LLaMA model implementation.

This module provides LlamaModelWithATP and LlamaForCausalLMWithATP classes that
integrate Adaptive Token Pruning (ATP) modules between decoder layers.

ATP modules are inserted after specific decoder layers (4, 14, 24) to progressively
prune vision tokens based on redundancy and spatial importance scores.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaModel, LlamaPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .llama_atp_layers import LlamaDecoderLayerWithLogits
from ..atp_module import ATPModule


class LlamaModelWithATP(LlamaModel):
    """
    LlamaModel with ATP modules inserted at specific decoder layers.
    
    ATP modules are inserted after layers 4, 14, and 24 to perform progressive
    token pruning. The model handles both training (soft masks) and inference 
    (hard pruning) modes.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Replace standard decoder layers with ATP-enabled ones
        for layer_idx in range(len(self.layers)):
            self.layers[layer_idx] = LlamaDecoderLayerWithLogits(config, layer_idx)
        
        # ATP configuration based on paper values
        # Insert ATP after layers 4, 14, 24 for 32-layer models
        total_layers = len(self.layers)
        if total_layers == 32:
            self.atp_insertion_points = [4, 14, 24]
        elif total_layers == 24:
            self.atp_insertion_points = [3, 11, 19] 
        else:
            # Default: insert at 12.5%, 43.75%, 75% depth
            self.atp_insertion_points = [
                int(total_layers * 0.125),
                int(total_layers * 0.4375), 
                int(total_layers * 0.75)
            ]
        
        self.atp_modules = nn.ModuleDict()
        
        # Initialize ATP modules with paper values
        for layer_idx in self.atp_insertion_points:
            self.atp_modules[f"atp_after_{layer_idx}"] = ATPModule(
                hidden_dim=config.hidden_size,
                lambda_sample=3.0,  # From paper
                temperature=10.0,   # From paper
                layer_idx=layer_idx,
                total_layers=total_layers,
                sampling_rate=None,  # Use layer-adaptive rates
            )
        
        # ATP state tracking
        self.atp_masks = []
        self.atp_stats = []
        self.num_vision_tokens = 576  # Default for ViT-L/14, will be updated
        
        print(f"ATP enabled with insertion points: {self.atp_insertion_points}")

    def set_vision_token_count(self, num_tokens):
        """Set the number of vision tokens for proper vision/text separation."""
        self.num_vision_tokens = num_tokens
        
    def get_atp_config(self):
        """Return current ATP configuration for debugging."""
        return {
            'insertion_points': self.atp_insertion_points,
            'num_vision_tokens': self.num_vision_tokens,
            'total_layers': len(self.layers),
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        # Force attention output for ATP
        output_attentions = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Handle inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Initialize tracking
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # Clear previous ATP outputs
        self.atp_masks = []
        self.atp_stats = []

        # Track current vision token count (decreases after each ATP layer)
        current_vision_tokens = self.num_vision_tokens

        # Process through all decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # Forward through decoder layer (returns attention logits)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                if len(layer_outputs) > 2:
                    next_decoder_cache += (layer_outputs[2],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # Extract attention logits (last element in outputs)
            attention_logits = layer_outputs[-1]
            attention_weights = layer_outputs[1] if output_attentions else None

            # Apply ATP if this layer has an ATP module
            if idx in self.atp_insertion_points:
                atp_module = self.atp_modules[f"atp_after_{idx}"]
                
                # Separate vision and text tokens
                # Vision tokens come first, then text tokens
                batch_size = hidden_states.shape[0]
                sequence_length = hidden_states.shape[1]
                
                if current_vision_tokens > sequence_length:
                    # No vision tokens left or sequence shorter than expected
                    continue
                    
                vision_tokens = hidden_states[:, :current_vision_tokens, :]
                text_tokens = hidden_states[:, current_vision_tokens:, :]

                # Apply ATP module
                atp_output = atp_module(
                    vision_tokens=vision_tokens,
                    text_tokens=text_tokens,
                    attention_logits=attention_logits,
                    attention_weights=attention_weights,
                    training=self.training
                )

                if self.training:
                    # Training: Use soft masks, keep all tokens physically
                    self.atp_masks.append(atp_output['vision_mask'])
                    # Concatenate back (vision tokens unchanged during training)
                    hidden_states = torch.cat([atp_output['vision_tokens'], text_tokens], dim=1)
                else:
                    # Inference: Hard pruning, physically remove tokens
                    pruned_vision = atp_output['vision_tokens']
                    hidden_states = torch.cat([pruned_vision, text_tokens], dim=1)
                    
                    # Update vision token count for next ATP layer
                    current_vision_tokens = pruned_vision.shape[1]
                    
                    # Update attention mask if provided
                    if attention_mask is not None:
                        # Keep text attention mask, adjust for pruned vision tokens
                        text_mask = attention_mask[:, current_vision_tokens:]
                        vision_mask = attention_mask[:, :current_vision_tokens]
                        # This is simplified - full implementation needs proper mask handling
                        attention_mask = torch.cat([vision_mask, text_mask], dim=1)

                # Store ATP statistics
                self.atp_stats.append(atp_output['pruning_stats'])

        # Final layer normalization
        hidden_states = self.norm(hidden_states)

        # Add final hidden states
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        # Return custom dict with ATP information
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attns,
            'atp_masks': self.atp_masks,
            'atp_stats': self.atp_stats,
        }


class LlamaForCausalLMWithATP(LlamaPreTrainedModel):
    """
    LlamaForCausalLM with ATP support for token pruning.
    
    This class extends the standard LlamaForCausalLM to support ATP modules
    while maintaining compatibility with the HuggingFace ecosystem.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelWithATP(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through ATP-enabled model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs['last_hidden_state'] if return_dict else outputs[0]
        
        # Generate logits
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [nn.functional.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Standard language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.get('past_key_values'),
            hidden_states=outputs.get('hidden_states'),
            attentions=outputs.get('attentions'),
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # Handle past key values for generation
        if past_key_values is not None:
            if isinstance(past_key_values, tuple):
                past_length = past_key_values[0][0].shape[2]
            else:
                past_length = past_key_values.get_seq_length()

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_length, dtype=torch.long, device=input_ids.device
            )
        else:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past