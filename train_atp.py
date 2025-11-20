#!/usr/bin/env python3

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer
)
from torch.optim import AdamW
from llava.model.language_model.llava_llama import (
    PruningContext,
    MyDecoderLayer,
    LlavaLlamaModel,
    LlavaConfig
)
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.model.llava_arch import LlavaMetaModel
from llava.train.train import make_supervised_data_module

# =============================================================================
# ATP-LLaVA Model Wrapper with Fixed Weight Loading
# =============================================================================

class ATP_LlavaMetaModel(LlavaMetaModel):
    def __init__(self, config):
        super().__init__(config)

        # IMPORTANT: Create pruning context BEFORE building layers
        if not hasattr(config, 'pruning_context'):
            config.pruning_context = PruningContext(config.num_hidden_layers)
        self.pruning_context = config.pruning_context

        # Build vision components (will be loaded from checkpoint)
        self.vision_tower = build_vision_tower(config)
        self.mm_projector = build_vision_projector(config)

        # Language model layers will be created by LlamaModel.__init__
        # DO NOT replace them here - wait for from_pretrained()

class ATP_LlavaLlamaModel(ATP_LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config):
        # Initialize both parent classes to create standard LLaVA architecture
        ATP_LlavaMetaModel.__init__(self, config)
        LlamaModel.__init__(self, config)
        # Model now has standard, randomly initialized layers

    def _replace_layers_with_atp(self):
        """Replace standard layers with ATP-aware versions AFTER loading weights"""
        print(f"Replacing {len(self.layers)} standard layers with ATP layers...")
        original_layers = self.layers
        new_layers = nn.ModuleList()

        for i, original_layer in enumerate(original_layers):
            atp_layer = MyDecoderLayer(
                self.config,
                i,
                self.pruning_context
            )
            # Copy the loaded LLaVA-1.5 weights into ATP layer
            atp_layer.load_state_dict(original_layer.state_dict(), strict=True)
            new_layers.append(atp_layer)

        self.layers = new_layers
        del original_layers
        print("Layer replacement complete.")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load LLaVA-1.5 weights and extend with ATP modules
        """
        # Load config
        config = kwargs.pop('config', None)
        if config is None:
            config = LlavaConfig.from_pretrained(pretrained_model_name_or_path)

        # Add pruning context to config
        if not hasattr(config, 'pruning_context'):
            config.pruning_context = PruningContext(config.num_hidden_layers)

        # Create model instance (standard LLaVA with random init)
        model = cls(config)

        # Load state dict (LLaVA-1.5 weights)
        state_dict = kwargs.get('state_dict', None)
        if state_dict is None:
            # Load from pretrained path
            import torch
            model_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location="cpu")
            else:
                # Use safetensors if available
                from safetensors.torch import load_file
                model_file = os.path.join(pretrained_model_name_or_path, "model.safetensors")
                if os.path.exists(model_file):
                    state_dict = load_file(model_file)
                else:
                    raise FileNotFoundError(f"No model file found in {pretrained_model_name_or_path}")

        # Filter and load weights into the standard model structure
        model_state_dict = model.state_dict()
        loaded_keys = set(state_dict.keys())
        model_keys = set(model_state_dict.keys())

        # Check what keys are missing (should be only ATP modules)
        missing_keys = model_keys - loaded_keys
        unexpected_keys = loaded_keys - model_keys

        print(f"Missing keys (expected ATP modules): {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")

        # Load matching weights
        for key in list(state_dict.keys()):
            if key in model_state_dict:
                if model_state_dict[key].shape == state_dict[key].shape:
                    model_state_dict[key] = state_dict[key]
                else:
                    print(f"Shape mismatch for {key}: {state_dict[key].shape} vs {model_state_dict[key].shape}")

        model.load_state_dict(model_state_dict, strict=False)

        # CRITICAL FIX: Now replace layers with ATP modules AFTER weights are loaded
        model._replace_layers_with_atp()

        # Freeze vision components (as per paper)
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        for param in model.mm_projector.parameters():
            param.requires_grad = False

        return model


# =============================================================================
# ATP Loss Function with Correct Formulas
# =============================================================================

def compute_atp_loss(model: ATP_LlavaLlamaModel, target_tokens: int = 144):
    """
    Compute ATP losses (Eq.12-14) from pruning instruments.
    Implements exact formulas from the paper.
    """
    layers = model.language_model.model.layers
    instruments = {i: layer.pruner.instrument for i, layer in enumerate(layers)
                   if layer.pruner is not None}

    if not instruments:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device), \
               torch.tensor(0.0, device=device), {}

    # Eq. 12: Sum over ATP layers (depth-weighted token count)
    # L_atp = Σ_i_k (Sum(Mask_i_k) / 576) * i_k
    L_atp = 0.0
    total_tokens = 0

    device = next(model.parameters()).device

    for layer_idx, inst in instruments.items():
        tokens_kept = inst['mask'].sum().float()
        # Depth penalty: earlier layers have lower weight
        L_atp += (tokens_kept / 576.0) * layer_idx
        total_tokens += tokens_kept

    # Eq. 13: L1 distance to target token budget
    # L_target = || N_bar - N_target ||_1
    avg_tokens = total_tokens / len(instruments)
    # Use L1 loss (absolute difference) as per paper notation ||·||_1
    L_target = torch.abs(avg_tokens - torch.tensor(target_tokens, device=device, dtype=torch.float32))

    stats = {
        'avg_tokens': avg_tokens.item(),
        'L_atp': L_atp.item(),
        'L_target': L_target.item(),
        'prune_ratio': avg_tokens.item() / 576.0,
    }

    return L_atp, L_target, stats


# =============================================================================
# Custom Trainer (Unchanged)
# =============================================================================

class ATPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_tokens = 144

    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard forward pass
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        # Add ATP loss
        atp_loss, target_loss, stats = compute_atp_loss(model, self.target_tokens)

        if stats:
            # Eq. 14: Combined loss
            total_loss = loss + atp_loss * 0.01 + target_loss * 0.2

            # Log every 10 steps
            if self.state.global_step % 10 == 0:
                self.log(stats)
        else:
            total_loss = loss

        return (total_loss, outputs) if return_outputs else total_loss


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Path to LLaVA-1.5 checkpoint (e.g., './models/llava-v1.5-7b')")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--atp_learning_rate", type=float, default=1e-4)
    parser.add_argument("--target_tokens", type=int, default=144)
    parser.add_argument("--freeze_vision", action="store_true", default=True,
                       help="Freeze vision tower and projector (recommended)")
    args = parser.parse_args()

    print(f"Loading LLaVA-1.5 base model from: {args.model_name_or_path}")

    # Load config and create ATP model
    config = LlavaConfig.from_pretrained(args.model_name_or_path)
    config.target_tokens = args.target_tokens

    # Load model with pre-trained weights
    model = ATP_LlavaLlamaModel.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.float16,
    )

    # Verify ATP modules are initialized
    atp_params = sum(p.numel() for layer in model.language_model.model.layers
                     if hasattr(layer, 'pruner') and layer.pruner
                     for p in layer.pruner.parameters())
    print(f"ATP module parameters: {atp_params:,} (randomly initialized)")

    # Prepare data
    data_module = make_supervised_data_module(args)

    # Separate optimizer groups
    atp_param_list = [p for layer in model.language_model.model.layers
                      if hasattr(layer, 'pruner') and layer.pruner
                      for p in layer.pruner.parameters()]

    # Get base parameters (exclude ATP)
    base_param_ids = [id(p) for p in atp_param_list]
    base_params = [p for p in model.parameters() if id(p) not in base_param_ids]

    optimizer = AdamW([
        {'params': base_params, 'lr': args.learning_rate},
        {'params': atp_param_list, 'lr': args.atp_learning_rate, 'weight_decay': 0.0}
    ])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="no",
        save_total_limit=3,
        fp16=True,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_drop_last=True,
    )

    # Initialize trainer
    trainer = ATPTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module['train_dataset'],
        eval_dataset=data_module.get('eval_dataset', None),
        data_collator=data_module['data_collator'],
        optimizers=(optimizer, None)
    )

    print(f"Starting ATP fine-tuning for {args.num_epochs} epochs...")
    print(f"Target tokens: {args.target_tokens}")
    print(f"Base LR: {args.learning_rate}, ATP LR: {args.atp_learning_rate}")

    # Train
    trainer.train()

    # Save final model
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.save_pretrained(args.output_dir)

    print("ATP-LLaVA fine-tuning complete!")


# Example command:
# python train_atp.py \
#     --model_name_or_path "./models/llava-v1.5-7b" \
#     --data_path "./playground/data/llava_v1_5_mix665k.json" \
#     --output_dir "./checkpoints/atp-llava-7b" \
#     --num_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --target_tokens 144

if __name__ == "__main__":
    main()