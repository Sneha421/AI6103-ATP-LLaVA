from atp_module import ATPModule
from atp_module import ATPLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """
    Complete example showing how to use ATP module and loss during training.
    
    NOTE: We create REALISTIC attention patterns instead of random data.
    Random attention gives uniform small values (~0.001), but real attention
    has clear peaks where tokens attend to specific other tokens.
    """
    print("=" * 80)
    print("ATP MODULE + LOSS EXAMPLE")
    print("=" * 80)
    
    # === Setup ===
    batch_size = 2
    num_vision_tokens = 576  # 24x24 grid
    num_text_tokens = 20
    hidden_dim = 4096
    num_heads = 32
    
    # Create ATP modules for multiple layers
    atp_layers = [4, 14, 24]
    atp_modules = nn.ModuleList([
        ATPModule(hidden_dim=hidden_dim) for _ in atp_layers
    ])
    
    # Create loss function
    atp_loss_fn = ATPLoss(
        lambda_atp=0.05,
        lambda_target=0.2,
        target_tokens=144,
    )
    
    print(f"\nSetup:")
    print(f"  - ATP insertion layers: {atp_layers}")
    print(f"  - Target average tokens: 144")
    print(f"  - Initial vision tokens: 576")
    
    # === Simulate Forward Pass ===
    print(f"\n{'=' * 80}")
    print("SIMULATED TRAINING STEP")
    print("=" * 80)
    
    # Dummy inputs
    vision_tokens = torch.randn(batch_size, num_vision_tokens, hidden_dim)
    text_tokens = torch.randn(batch_size, num_text_tokens, hidden_dim)
    
    seq_len = num_vision_tokens + num_text_tokens
    
    # === IMPORTANT: Create REALISTIC attention patterns ===
    # Instead of random attention, create attention with clear structure
    print("\n[Creating realistic attention patterns...]")
    
    attention_logits = torch.randn(batch_size, num_heads, seq_len, seq_len) * 0.1
    
    # Add strong self-attention for some vision tokens (simulate important objects)
    # Tokens 100-120 attend strongly to each other (e.g., main object)
    attention_logits[:, :, 100:120, 100:120] += 100.0
    
    # Tokens 200-230 attend strongly to each other (e.g., secondary object)
    attention_logits[:, :, 200:230, 200:230] += 150.0
    
    # Add strong cross-attention from text to some vision tokens
    # Text tokens strongly attend to tokens 100-120 (main object relevant to text)
    attention_logits[:, :, num_vision_tokens:, 100:120] += 125.5
    
    # Text tokens moderately attend to tokens 300-350 (background context)
    attention_logits[:, :, num_vision_tokens:, 300:350] += 100.0
    
    # Normalize to get realistic attention weights
    attention_weights = F.softmax(attention_logits, dim=-1)
    
    print("✓ Created realistic attention with important regions")
    print(f"  - High self-attention regions: [100-120], [200-230]")
    print(f"  - High cross-attention regions: [100-120], [300-350]")
    
    # Collect masks from all ATP layers
    all_masks = []
    current_vision = vision_tokens
    
    for i, (atp_module, layer_idx) in enumerate(zip(atp_modules, atp_layers)):
        print(f"\n--- ATP Module at Layer {layer_idx} ---")
        
        # Forward pass
        output = atp_module(
            current_vision,
            text_tokens,
            attention_logits,
            attention_weights,
            training=True  # Training mode
        )
        
        # Extract mask
        mask = output['vision_mask']
        all_masks.append(mask)
        
        # Update vision tokens for next layer (in training, they're unchanged)
        current_vision = output['vision_tokens']
        
        # Print stats
        stats = output['pruning_stats']
        print(f"  Redundant threshold: {stats['theta_r']:.4f}")
        print(f"  Spatial threshold: {stats['theta_s']:.4f}")
        print(f"  Avg mask value: {stats['avg_mask_value']:.4f}")
        print(f"  Est. kept tokens: {stats['estimated_kept_tokens']:.1f} / {num_vision_tokens}")
    
    # === Compute Loss ===
    print(f"\n{'=' * 80}")
    print("LOSS COMPUTATION")
    print("=" * 80)
    
    loss_dict = atp_loss_fn(all_masks, atp_layers)
    
    print(f"\nLoss Components:")
    print(f"  ATP Penalty (L_atp): {loss_dict['atp_penalty']:.4f}")
    print(f"  Target Loss (L_target): {loss_dict['target_loss']:.4f}")
    print(f"  Total ATP Loss: {loss_dict['total_atp_loss']:.4f}")
    
    print(f"\nFinal Training Loss would be:")
    print(f"  L = L_ntp + {loss_dict['lambda_atp']:.3f} * {loss_dict['atp_penalty']:.4f} + "
          f"{loss_dict['lambda_target']:.3f} * {loss_dict['target_loss']:.4f}")
    print(f"  L = L_ntp + {loss_dict['total_atp_loss']:.4f}")
    
    # === Inference Example ===
    print(f"\n{'=' * 80}")
    print("INFERENCE MODE (Hard Pruning)")
    print("=" * 80)
    
    with torch.no_grad():
        atp_modules[0].eval()
        output = atp_modules[0](
            vision_tokens,
            text_tokens,
            attention_logits,
            attention_weights,
            training=False  # Inference mode
        )
        
        stats = output['pruning_stats']
        print(f"\nPruning Results:")
        print(f"  Original tokens: {stats['original_tokens']}")
        print(f"  Kept tokens: {stats['kept_tokens']}")
        print(f"  Pruning ratio: {stats['pruning_ratio']*100:.1f}%")
        print(f"  Output shape: {output['vision_tokens'].shape}")


if __name__ == "__main__":
    example_usage()
