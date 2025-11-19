#!/usr/bin/env python3

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    TrainingArguments,
    Trainer
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
# ATP-LLaVA Model Wrapper
# =============================================================================

class ATP_LlavaMetaModel(LlavaMetaModel):
    def __init__(self, config):
        super().__init__(config)
        self.vision_tower = build_vision_tower(config)
        self.mm_projector = build_vision_projector(config)

        # Replace decoder layers with pruning-aware versions
        self.language_model.model.layers = nn.ModuleList([
            MyDecoderLayer(self.language_model.model.config, i, self.pruning_context)
            for i in range(self.language_model.model.config.num_hidden_layers)
        ])

    def forward(self, *args, **kwargs):
        # Reset pruning state
        self.pruning_context.reset()
        return super().forward(*args, **kwargs)


class ATP_LlavaLlamaModel(ATP_LlavaMetaModel, LlavaLlamaModel):
    def __init__(self, config):
        config.pruning_context = PruningContext(config.num_hidden_layers)
        super().__init__(config)
        self.pruning_context = config.pruning_context


# =============================================================================
# ATP Loss Function
# =============================================================================

def compute_atp_loss(model: ATP_LlavaLlamaModel, target_tokens: int = 144):
    """
    Compute ATP losses (Eq.12-14) from pruning instruments.
    """
    layers = model.language_model.model.layers
    instruments = {i: layer.pruner.instrument for i, layer in enumerate(layers)
                   if layer.pruner is not None}

    if not instruments:
        return torch.tensor(0.0, device=next(model.parameters()).device), \
               torch.tensor(0.0, device=next(model.parameters()).device), {}

    L_atp = 0
    total_tokens = 0

    for layer_idx, inst in instruments.items():
        tokens_kept = inst['mask'].sum()
        L_atp += (tokens_kept / 576) * layer_idx  # Depth penalty
        total_tokens += tokens_kept

    L_atp = L_atp / len(instruments)
    avg_tokens = total_tokens / len(instruments)
    L_target = F.mse_loss(avg_tokens,
                         torch.tensor(target_tokens, device=avg_tokens.device, dtype=torch.float32))

    stats = {
        'avg_tokens': avg_tokens.item(),
        'tokens_per_layer': {i: inst['num_vision_token'] for i, inst in instruments.items()},
        'prune_ratio': avg_tokens.item() / 576,
        'theta_r_mean': sum(inst['theta_r'] for inst in instruments.values()) / len(instruments),
        'theta_s_mean': sum(inst['theta_s'] for inst in instruments.values()) / len(instruments),
    }

    return L_atp, L_target, stats


# =============================================================================
# Custom Trainer with ATP Loss
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
            total_loss = loss + atp_loss * 0.01 + target_loss * 0.2

            # Log to wandb/tensorboard
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
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--atp_learning_rate", type=float, default=1e-4)
    parser.add_argument("--target_tokens", type=int, default=144)
    args = parser.parse_args()

    # Load model config and create ATP-LLaVA model
    config = LlavaConfig.from_pretrained(args.model_name_or_path)
    model = ATP_LlavaLlamaModel(config)

    # Load pretrained weights
    state_dict = torch.load(os.path.join(args.model_name_or_path, "pytorch_model.bin"),
                           map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    # Build dataset
    data_module = make_supervised_data_module(args)

    # Separate optimizer groups for ATP
    atp_params = [p for layer in model.language_model.model.layers
                  if hasattr(layer, 'pruner') and layer.pruner
                  for p in layer.pruner.parameters()]
    base_params = [p for p in model.parameters() if id(p) not in [id(ap) for ap in atp_params]]

    optimizer = AdamW([
        {'params': base_params, 'lr': args.learning_rate},
        {'params': atp_params, 'lr': args.atp_learning_rate}
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

    # Train
    trainer.train()

    # Save final model
    model.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    print("Training complete!")

'''
python train_atp.py \
    --model_name_or_path "./llava/model" \
    --data_path "./playground/data/llava_v1_5_mix665k.json" \
    --output_dir "./checkpoints/atp-llava-7b" \
    --num_epochs 1 \
    --per_device_train_batch_size 4 \
    --target_tokens 144
'''

if __name__ == "__main__":
    main()
