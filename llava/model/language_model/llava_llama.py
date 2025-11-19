#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

# for custom layer
from typing import Dict, Any
from packaging import version
from transformers import __version__ as __transformer_version__
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

class ATPModule(nn.Module):
    """
    This is the custom inter layer.
    """
    def __init__(
        self,
        num_vision_tokens: int,
        lambda_sample: float = 3.0,
        temperature: float = 100.0
    ):
        super().__init__()

        # sampling scaling corfficient
        self.lambda_sample = lambda_sample

        # temperature
        self.temperature = temperature
        # ... make it a trainable hyperparameter
        self.register_buffer('T', torch.tensor(temperature))

        # AdaptiveAvgPool1d will turn any input size vector into fixed-size feature vectors
        self.feature_dim = 256

        # [B, 1, L_v] -> [B, 1, feature_dim]
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.feature_dim)


        # MLP to get theta scalars with linear transformation
        self.shared_fc = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),
            nn.ReLU()
        )

        # Two independent predition heads
        self.head_redundant = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.head_spatial = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # initialize sampling grid (should be 24x24)
        self.grid_size = int(num_vision_tokens ** 0.5)

        # Initialize instrumantation
        self.instrument = {}

    def compute_scores(
        self,
        attention_weights: torch.Tensor,
        num_vision_tokens
    ) -> torch.Tensor:
        """
        Compute scores for formula (3), (4) and (5).
        """
        batch_size = attention_weights.shape[0]
        device = attention_weights.device

        # 1. extract attention maps
        # vision token self attention
        # [B, H, L_v, L_v]
        self_attn_map = attention_weights[:, :, :num_vision_tokens, :num_vision_tokens]
        # text to vision attention
        # [B, H, L_t, L_v]
        text_vision_map = attention_weights[:, :, num_vision_tokens:, :num_vision_tokens]

        # 2. compute S_self (3)
        # for each vision token, compute the average attention it
        #   receives from the other vision tokens
        # [B, H, L_v, L_v] -> mean over heads and keys -> [B, L_v]
        S_self = self_attn_map.mean(dim=(1, 2))

        # 3. compute S_cross (4)
        # For each vision token, compute the average attention it
        #   receives from the other text tokens
        # [B, H, L_t, L_v] -> mean over heads and queries(text) -> [B, L_v]
        S_cross = text_vision_map.mean(dim=(1, 2))

        # 4. combine to compute S_redundant
        # S_redundant = (S_self + S_cross) / 2
        S_redundant = (S_self + S_cross) / 2.0

        # 5. spatial sampling
        stride = 2
        sampled_positions = self.grid_size // stride
        num_sampled = sampled_positions ** 2  # total number of tokens
        # sample rate: R^s = num_sampled / L_v
        R_s = num_sampled / num_vision_tokens

        # 6. spatial score (to mark which token got kept)
        # [batch, L_v]
        sample_mask = torch.zeros(batch_size, num_vision_tokens, device=device)

        # sampling in the grid
        for i in range(0, self.grid_size, stride):
            for j in range(0, self.grid_size, stride):
                idx = i * self.grid_size + j
                sample_mask[:, idx] = 1.0

        # 7. compute the spatial score base on the sampling (5)
        # S_spatial = 1 - R^s * λ_sample
        S_spatial = torch.where(
            sample_mask.bool(),
            1 - R_s * self.lambda_sample,
            torch.tensor(-100.0, device=device)
        )

        return S_redundant, S_spatial, S_self, S_cross

    def predict_thresholds(
        self,
        S_self: torch.Tensor,
        S_cross: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicting thresholds

        params:
            S_self: [B, L_v]
            S_cross: [B, L_v]

        return:
            theta_r: [B, 1]
            theta_s: [B, 1]
        """
        # Transform variable-size inputs to fixed size for stable training [B, L_v] → [B, feature_dim]
        S_self_pooled = self.adaptive_pool(S_self.unsqueeze(1)).squeeze(1)    # [B, 256] 
        S_cross_pooled = self.adaptive_pool(S_cross.unsqueeze(1)).squeeze(1)  # [B, 256]

        # z = Linear(concat(S_self_pooled, S_cross_pooled)) (6) - scores after adaptive pooling
        score_features = torch.cat([S_self_pooled, S_cross_pooled], dim=-1)  # [B, 512]

        # shared features
        shared_features = self.shared_fc(score_features)  # [B, 512] -> [B, 256]

        # two-head predictors (7) (8)
        theta_r = self.head_redundant(shared_features)  # [B, 256] -> [B, 1]
        theta_s = self.head_spatial(shared_features)    # [B, 256] -> [B, 1]

        return theta_r, theta_s

    def generate_masks(
        self,
        S_redundant: torch.Tensor,
        S_spatial: torch.Tensor,
        theta_r: torch.Tensor,
        theta_s: torch.Tensor,
        is_training: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate masks.

        params:
            S_redundant: [B, L_v]
            S_spatial: [B, L_v]
            theta_r: [B, 1]
            theta_s: [B, 1]

        return:
            mask_r: [B, L_v]
            mask_s: [B, L_v]
            mask: [B, L_v]
        """
        # Mask_r = σ((S_redundant - θ_r) * T) (9)
        # Mask_s = σ((S_spatial - θ_s) * T) (10)
        # where T is the temperature

        # matching the dimensions [B, 1] -> [B, L_v]
        theta_r = theta_r.expand_as(S_redundant)
        theta_s = theta_s.expand_as(S_spatial)

        if is_training:
            # during training soft masks
            mask_r = torch.sigmoid((S_redundant - theta_r) * self.T)
            mask_s = torch.sigmoid((S_spatial - theta_s) * self.T)

            # Mask = max(Mask^r, Mask^s) (11)
            # [B, L_v]
            mask = torch.max(mask_r, mask_s)
        else:
            # during inference, we use hard masks
            mask_r = (S_redundant > theta_r)
            mask_s = (S_spatial > theta_s)
            mask = torch.logical_or(mask_r, mask_s)

        # TODO: according to paper, the soft mask should later
        #       multplied with attention weights. But I don't
        #       know how yet

        return mask_r, mask_s, mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
        num_vision_tokens = 0,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.LongTensor], Dict[str, Any]]:
        """
        Propagate, calculating scores

        params:
            x: [batch, seq_len, hidden_size]
            attention_weights: [batch, num_heads, seq_len, seq_len] OR None
        return:
            [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ==== Vision and text token extraction ====
        # First num_vision_tokens are vision tokens
        vision_tokens = hidden_states[:, :num_vision_tokens, :]  # [B, L_v, hidden_size]
        text_tokens = hidden_states[:, num_vision_tokens:, :]    # [B, seq_len-L_v, hidden_size]

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        vision_position_ids = position_ids[:, :num_vision_tokens]
        text_position_ids = position_ids[:, num_vision_tokens:]


        # ==== Process attention weights ====
        # attention_weights: [B, H, seq_len, seq_len]
        if attention_weights is not None:
            # 3.3.1: compute redundant and spatial score
            (
                redundant_score,
                spatial_score,
                self_modal_score,
                cross_modal_score
            ) = self.compute_scores(
                attention_weights, num_vision_tokens
            )

            # 3.3.2 part I: token pruning
            theta_r, theta_s = self.predict_thresholds(
                self_modal_score, cross_modal_score
            )

            # 3.3.2 part II: generate masks
            mask, mask_r, mask_s = self.generate_masks(
                redundant_score, spatial_score, theta_r, theta_s
            )

            # Apply masks
            if is_training:
                # the mask are passed to next layer's attention
                pruned_vision_tokens = vision_tokens

                # get prune ratio
                ratio = mask.float().mean().item()  # For soft mask
            else:
                # hard prune
                kept_indices = mask.nonzero(as_tuple=True)[1]
                pruned_vision_tokens = vision_tokens[:, kept_indices]

                # preserve original position IDs
                if position_ids is not None:
                    pruned_position_ids = vision_position_ids[:, kept_indices]
                    position_ids = torch.cat([pruned_position_ids, text_position_ids], dim=1)

                # get prune ratio
                ratio = mask.float().mean().item()  # mask is already boolean

            # XXX: maybe a instrumentation class is better
            instrument = {
                'mask': mask,
                'mask_r': mask_r,
                'mask_s': mask_s,
                'theta_r': theta_r.mean().item(),
                'theta_s': theta_s.mean().item(),
                'num_vision_token': pruned_vision_tokens.shape[1],
                'kept_token_ratio': ratio
            }

            hidden_states = torch.cat([pruned_vision_tokens, text_tokens], dim=1)
        else:
            # empty instrument
            instrument = {}

        # save instrumentation
        self.instrument = instrument

        return hidden_states, position_ids, instrument

class PruningContext():
    """
    Shared context for all decoder layers
    """
    # initial vision tokens
    _MAX_VISION_TOKENS = 576

    def __init__(self, num_layers):
        self.num_layers = num_layers
        self.reset()

    def reset(self):
        # NOTE: extra slot to prevent boundary check
        self.num_vision_token = [self._MAX_VISION_TOKENS] * (self.num_layers + 1)
        self.mask = None

class MyDecoderLayer(LlamaDecoderLayer):
    """
    Llama layer + optional custom inter layer.
    """

    # minimal transformer version
    _MINIMAL_VERSION = "4.40.0"

    # initial vision tokens
    _MAX_VISION_TOKENS = 576

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        shared_context: PruningContext
    ):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.context: PruningContext = shared_context   # shared across layers

        # NOTE: version guard is required, since the
        #       implementation of forward() have changed.
        if version.parse(__transformer_version__) < \
            version.parse(self._MINIMAL_VERSION):
            raise ValueError(
                f"MyDecoderLayer requires transformers >= {self._MINIMAL_VERSION}, "
                f"but current version is {__transformer_version__}."
            )

        # Instantiate pruners
        self.pruner = None
        if layer_idx in [4, 14, 24]:
            self.pruner = ATPModule(self._MAX_VISION_TOKENS)

    # Apply the pruning mask before attention
    def apply_pruning_mask(self, attention_mask, hidden_states):
        # TODO: this is problematic, not sure it matches the
        #       design of the paper though
        if self.training and self.context.mask is not None:
            if attention_mask is None:
                attention_mask = torch.zeros(
                    1, 1, 1, hidden_states.shape[1],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype
                )
            bias = (1 - self.prev_atp_mask).unsqueeze(1).unsqueeze(2) * \
                torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask + bias

        self.context.mask = None
        return attention_mask

    # Exact copy of LlamaDecoderLayer.forward()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        **kwargs
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # apply pruning mask
        attention_mask = self.apply_pruning_mask(
            attention_mask, hidden_states
        )

        # continue with transformer attention
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # get number of vision tokens
        num_vision_tokens = self.context.num_vision_token[self.layer_idx]

        # insert custom module
        if self.pruner is not None:
            (
                hidden_states,
                position_ids,
                instrument
            ) = self.pruner(
                hidden_states=hidden_states,
                attention_weights=attn_weights,
                num_vision_tokens=num_vision_tokens,
                is_training=self.training
            )

            # update next layer's number of vision tokens
            self.context.num_vision_token[self.layer_idx + 1] = (
                instrument['num_vision_token']
            )

            # while training, pass the mask to next layer
            if self.training:
                self.context.mask = instrument['mask']
        else:
            # the vision token never pruned
            self.context.num_vision_token[self.layer_idx + 1] = num_vision_tokens

        return hidden_states


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

        # initializing the shared context
        self.pruning_context = PruningContext(config.num_hidden_layers)

        # replace all layers with Llama custom layers
        self.layers = nn.ModuleList(
            [MyDecoderLayer(config, i, self.pruning_context)
             for i in range(config.num_hidden_layers)]
        )

    def forward(self, *args, **kwargs):
        # reset pruning state at each forward pass
        self.pruning_context.reset()

        # invoke parent's forward
        return super().forward(*args, **kwargs)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
