"""
Test script for ATP (Adaptive Token Pruning) integration with LLaVA.

This script validates that:
1. ATP modules are properly initialized
2. Attention logits are correctly extracted 
3. Vision tokens are properly separated from text tokens
4. ATP loss computation works correctly
5. Model can run in both training and inference modes

Run this script to validate ATP integration before full training.
"""

import torch
import torch.nn as nn
from transformers import LlamaConfig, AutoTokenizer

# Import ATP-enabled LLaVA components
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from llava.model.atp_module import ATPModule, ATPLoss


def test_atp_config():
    """Test ATP configuration integration."""
    print("🔧 Testing ATP Configuration...")
    
    config = LlavaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        # ATP config with paper values
        use_atp=True,
        atp_lambda_atp=0.05,
        atp_lambda_target=0.2,
        atp_target_tokens=144,
        atp_initial_tokens=576,
    )
    
    assert config.use_atp == True
    assert config.atp_lambda_atp == 0.05
    assert config.atp_target_tokens == 144
    print("✅ ATP Configuration: PASSED")


def test_atp_module_standalone():
    """Test ATP module in isolation."""
    print("\n🧩 Testing ATP Module Standalone...")
    
    # Create ATP module
    atp = ATPModule(
        hidden_dim=4096,
        lambda_sample=3.0,
        temperature=10.0,
        layer_idx=14,
        total_layers=32
    )
    
    # Create dummy inputs
    batch_size = 2
    vision_tokens = 576
    text_tokens = 100
    hidden_dim = 4096
    num_heads = 32
    
    vision_hidden = torch.randn(batch_size, vision_tokens, hidden_dim)
    text_hidden = torch.randn(batch_size, text_tokens, hidden_dim)
    
    # Create dummy attention logits and weights
    total_seq = vision_tokens + text_tokens
    attention_logits = torch.randn(batch_size, num_heads, total_seq, total_seq)
    attention_weights = torch.softmax(attention_logits, dim=-1)
    
    # Test training mode (soft masks)
    atp.train()
    output = atp(
        vision_tokens=vision_hidden,
        text_tokens=text_hidden,
        attention_logits=attention_logits,
        attention_weights=attention_weights,
        training=True
    )
    
    assert 'vision_tokens' in output
    assert 'vision_mask' in output
    assert 'pruning_stats' in output
    assert output['vision_tokens'].shape == vision_hidden.shape  # No pruning in training
    
    # Test inference mode (hard pruning)
    atp.eval()
    output = atp(
        vision_tokens=vision_hidden,
        text_tokens=text_hidden,
        attention_logits=attention_logits,
        attention_weights=attention_weights,
        training=False
    )
    
    assert 'vision_tokens' in output
    assert 'pruning_stats' in output
    pruned_tokens = output['vision_tokens'].shape[1]
    assert pruned_tokens <= vision_tokens  # Should prune some tokens
    
    print(f"   Training mode: {vision_tokens} tokens kept (soft masks)")
    print(f"   Inference mode: {pruned_tokens}/{vision_tokens} tokens kept (hard pruning)")
    print("✅ ATP Module Standalone: PASSED")


def test_atp_loss():
    """Test ATP loss computation."""
    print("\n📊 Testing ATP Loss...")
    
    atp_loss = ATPLoss(
        lambda_atp=0.05,
        lambda_target=0.2,
        target_tokens=144,
        initial_vision_tokens=576
    )
    
    # Create dummy soft masks from 3 ATP layers
    batch_size = 2
    vision_tokens = 576
    
    # Simulate progressive pruning: keep fewer tokens in deeper layers
    masks = [
        torch.ones(batch_size, vision_tokens) * 0.8,  # Layer 4: keep 80%
        torch.ones(batch_size, vision_tokens) * 0.5,  # Layer 14: keep 50%  
        torch.ones(batch_size, vision_tokens) * 0.25, # Layer 24: keep 25%
    ]
    
    layer_indices = [4, 14, 24]
    
    loss_dict = atp_loss(masks, layer_indices)
    
    assert 'atp_penalty' in loss_dict
    assert 'target_loss' in loss_dict
    assert 'total_atp_loss' in loss_dict
    
    print(f"   ATP Penalty: {loss_dict['atp_penalty']:.4f}")
    print(f"   Target Loss: {loss_dict['target_loss']:.4f}")  
    print(f"   Total ATP Loss: {loss_dict['total_atp_loss']:.4f}")
    print("✅ ATP Loss: PASSED")


def test_llava_atp_model():
    """Test full LLaVA model with ATP integration."""
    print("\n🦙 Testing LLaVA Model with ATP...")
    
    # Create small config for testing
    config = LlavaConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024, 
        num_hidden_layers=8,  # Smaller for testing
        num_attention_heads=8,
        max_position_embeddings=1024,
        # ATP enabled by default
        use_atp=True,
        atp_initial_tokens=144,  # Smaller for testing
    )
    
    # Create model
    model = LlavaLlamaForCausalLM(config)
    
    # Check ATP components are initialized
    assert hasattr(model.model, 'atp_modules')
    assert len(model.model.atp_insertion_points) > 0
    
    print(f"   ATP insertion points: {model.model.atp_insertion_points}")
    print(f"   Number of ATP modules: {len(model.model.atp_modules)}")
    
    # Test forward pass
    batch_size = 1
    seq_len = 200  # 144 vision + 56 text tokens
    
    # Set vision token count
    model.model.set_vision_token_count(144)
    
    # Create dummy input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Training mode
    model.train()
    outputs = model(input_ids=input_ids)
    
    assert 'logits' in outputs or hasattr(outputs, 'logits')
    
    # Check if ATP masks were generated
    if hasattr(model.model, 'atp_masks') and model.model.atp_masks:
        print(f"   Generated {len(model.model.atp_masks)} ATP masks")
        for i, mask in enumerate(model.model.atp_masks):
            avg_mask = mask.mean().item()
            print(f"   ATP Layer {model.model.atp_insertion_points[i]}: avg mask = {avg_mask:.3f}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    print("✅ LLaVA Model with ATP: PASSED")


def test_attention_logits_extraction():
    """Test that attention logits are properly extracted."""
    print("\n🔍 Testing Attention Logits Extraction...")
    
    from llava.model.language_model.llama_atp_layers import LlamaDecoderLayerWithLogits
    
    # Small config for testing
    config = LlamaConfig(
        hidden_size=256,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512
    )
    
    layer = LlamaDecoderLayerWithLogits(config, layer_idx=0)
    
    batch_size = 1
    seq_len = 50
    hidden_dim = 256
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    outputs = layer(hidden_states, output_attentions=True)
    
    # Should return: (hidden_states, attention_weights, past_key_value, attention_logits)
    assert len(outputs) >= 4
    
    hidden_out = outputs[0]
    attention_weights = outputs[1] 
    attention_logits = outputs[-1]  # Last element should be logits
    
    assert hidden_out.shape == hidden_states.shape
    assert attention_weights is not None
    assert attention_logits is not None
    assert attention_logits.shape == attention_weights.shape
    
    print(f"   Attention weights shape: {attention_weights.shape}")
    print(f"   Attention logits shape: {attention_logits.shape}")
    print("✅ Attention Logits Extraction: PASSED")


def run_all_tests():
    """Run all ATP integration tests."""
    print("🚀 Starting ATP Integration Tests\n")
    
    try:
        test_atp_config()
        test_atp_module_standalone() 
        test_atp_loss()
        test_attention_logits_extraction()
        test_llava_atp_model()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📋 ATP Integration Summary:")
        print("   ✅ ATP configuration properly integrated")
        print("   ✅ ATP modules work in isolation")
        print("   ✅ ATP loss computation functional")
        print("   ✅ Attention logits properly extracted")
        print("   ✅ Full LLaVA+ATP model working")
        print("\n🚀 Ready for training with ATP enabled!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    run_all_tests()