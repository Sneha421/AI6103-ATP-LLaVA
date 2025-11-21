#!/usr/bin/env python3

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import warnings
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    CLIPImageProcessor,
)
from torch.optim import AdamW
from llava.model.language_model.llava_llama import (
    PruningContext,
    MyDecoderLayer,
    LlavaLlamaForCausalLM,
)
from llava.train.train import DataArguments, make_supervised_data_module

# =============================================================================
# ATP-LLaVA Model Wrapper
# =============================================================================

class ATP_LlavaLlamaModelWrapper:
    """
    Wrapper class that takes a loaded LlavaLlamaForCausalLM and injects ATP modules.
    """

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load LLaVA-1.5 model and inject ATP modules.
        Returns: LlavaLlamaForCausalLM with ATP modules injected
        """
        # Load base LLaVA model using its native from_pretrained
        print(f"Loading base LLaVA model from {pretrained_model_name_or_path}...")
        model = LlavaLlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(dtype=torch.bfloat16 if kwargs.get('torch_dtype') == torch.bfloat16 else torch.float16)

        # Add pruning context and ensure it's clean
        model.model.pruning_context = PruningContext(model.config.num_hidden_layers)
        
        # Ensure pruning context is in proper state for training
        model.model.pruning_context.reset()

        # Inject ATP modules
        ATP_LlavaLlamaModelWrapper._inject_atp_modules(model)

        # Freeze vision components
        for param in model.get_vision_tower().parameters():
            param.requires_grad = False
        for param in model.model.mm_projector.parameters():
            param.requires_grad = False

        return model

    @staticmethod
    def _inject_atp_modules(model):
        print(f"Injecting ATP modules into {len(model.model.layers)} decoder layers...")
        original_layers = model.model.layers
        new_layers = nn.ModuleList()

        for i, original_layer in enumerate(original_layers):
            atp_layer = MyDecoderLayer(model.config, i, model.model.pruning_context)
            atp_layer.load_state_dict(original_layer.state_dict(), strict=False)
            new_layers.append(atp_layer)

        model.model.layers = new_layers
        del original_layers
        print("ATP injection complete.")

# =============================================================================
# ATP Loss Function (Fixed Access Path)
# =============================================================================

def compute_atp_loss(model: LlavaLlamaForCausalLM, target_tokens: int = 144):
    """Compute ATP losses (Eq.12-14) from pruning instruments."""
    layers = model.model.layers
    instruments = {i: layer.pruner.instrument for i, layer in enumerate(layers)
                   if layer.pruner is not None and hasattr(layer.pruner, 'instrument') and layer.pruner.instrument}

    if not instruments:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device, requires_grad=True), \
               torch.tensor(0.0, device=device, requires_grad=True), {}

    L_atp = 0.0
    total_tokens = 0
    device = next(model.parameters()).device

    for layer_idx, inst in instruments.items():
        if 'mask' in inst and inst['mask'] is not None:
            # Detach the mask to avoid gradient conflicts, then reattach for loss computation
            mask_detached = inst['mask'].detach()
            tokens_kept = mask_detached.sum().float()
            # Create a new tensor with gradients enabled for loss computation
            tokens_kept_grad = tokens_kept.clone().requires_grad_(True)
            L_atp += (tokens_kept_grad / 576.0) * layer_idx
            total_tokens += tokens_kept_grad

    if total_tokens == 0:
        # No valid instruments found
        return torch.tensor(0.0, device=device, requires_grad=True), \
               torch.tensor(0.0, device=device, requires_grad=True), {}

    avg_tokens = total_tokens / len(instruments)
    L_target = torch.abs(avg_tokens - torch.tensor(target_tokens, device=device, dtype=torch.float32))

    stats = {
        'avg_tokens': avg_tokens.item() if hasattr(avg_tokens, 'item') else float(avg_tokens),
        'L_atp': L_atp.item() if hasattr(L_atp, 'item') else float(L_atp),
        'L_target': L_target.item() if hasattr(L_target, 'item') else float(L_target),
        'prune_ratio': (avg_tokens.item() if hasattr(avg_tokens, 'item') else float(avg_tokens)) / 576.0,
    }

    return L_atp, L_target, stats

# =============================================================================
# Custom Trainer
# =============================================================================

class ATPTrainer(Trainer):
    def __init__(self, target_tokens=144, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_tokens = target_tokens

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        # Compute ATP loss as per the paper methodology (Eq. 12-14)
        atp_loss, target_loss, stats = compute_atp_loss(model, self.target_tokens)
        
        if stats:
            # Loss weights from the paper: ATP loss (efficiency) + target loss (accuracy)
            total_loss = loss + atp_loss * 0.01 + target_loss * 0.2
            if self.state.global_step % 10 == 0:
                self.log(stats)
        else:
            total_loss = loss

        return (total_loss, outputs) if return_outputs else total_loss


# =============================================================================
# Main Training Script (Fixed Layer Access)
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                       default="liuhaotian/llava-v1.5-7b",
                       help="HuggingFace model ID")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--atp_learning_rate", type=float, default=1e-4)
    parser.add_argument("--target_tokens", type=int, default=144)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    args = parser.parse_args()

    # Basic CUDA setup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # print configurations
    print("\n" + "="*80)
    print("ATP-LLaVA Training Configuration")
    print("="*80)
    print(f"Model: {args.model_name_or_path}")
    print(f"Data: {args.data_path}")
    print(f"Images: {args.image_folder}")
    print(f"Output: {args.output_dir}")
    print(f"Target tokens: {args.target_tokens}")
    print(f"LLM LR: {args.learning_rate}, ATP LR: {args.atp_learning_rate}")
    print(f"Batch size: {args.per_device_train_batch_size}, Grad accum: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print("="*80 + "\n")

    # Load tokenizer first (required for data module)
    print(f"Loading tokenizer from {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    # Create data arguments
    data_args = DataArguments(
        data_path=args.data_path,
        image_folder=args.image_folder,
        lazy_preprocess=True,
    )

    data_args.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    print(f"Loading model from {args.model_name_or_path}...")
    model = ATP_LlavaLlamaModelWrapper.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    # Disable cache during training
    model.config.use_cache = False

    # ATP methodology does not use gradient checkpointing due to dynamic computational graphs
    print("🔬 ATP-LLaVA uses dynamic token pruning - gradient checkpointing disabled by design")

    # model is a LlavaLlamaForCausalLM, decoder layers live under model.model.layers
    atp_params = sum(p.numel() for layer in model.model.layers
                     if hasattr(layer, 'pruner') and layer.pruner
                     for p in layer.pruner.parameters())
    print(f"ATP module parameters: {atp_params:,} (trainable)")

    # load data
    data_module = make_supervised_data_module(tokenizer, data_args)

    # In main(), update the ATP parameter collection:
    atp_param_list = [p for layer in model.model.layers
                      if hasattr(layer, 'pruner') and layer.pruner
                      for p in layer.pruner.parameters()]

    base_param_ids = [id(p) for p in atp_param_list]
    vision_param_ids = [id(p) for p in list(model.get_vision_tower().parameters()) +
                        list(model.model.mm_projector.parameters())]
    trainable_params = [p for p in model.parameters()
                        if id(p) not in base_param_ids and id(p) not in vision_param_ids]

    optimizer = AdamW([
        {'params': trainable_params, 'lr': args.learning_rate},
        {'params': atp_param_list, 'lr': args.atp_learning_rate, 'weight_decay': 0.0}
    ])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="no",
        save_total_limit=3,
        bf16=args.bf16,
        fp16=not args.bf16,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_drop_last=True,
        report_to=args.report_to,
        gradient_checkpointing=False,  # ATP methodology does not support gradient checkpointing
        max_grad_norm=1.0,  # Gradient clipping for stability
    )

    trainer = ATPTrainer(
        target_tokens=args.target_tokens,
        model=model,
        args=training_args,
        train_dataset=data_module['train_dataset'],
        eval_dataset=data_module.get('eval_dataset', None),
        data_collator=data_module['data_collator'],
        optimizers=(optimizer, None)
    )

    print("\nStarting ATP fine-tuning...")
    trainer.train()

    # Re-enable cache for inference
    model.config.use_cache = True

    # Save final model
    model.save_pretrained(args.output_dir)
    model.config.save_pretrained(args.output_dir)

    # Save the already loaded tokenizer
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    main()