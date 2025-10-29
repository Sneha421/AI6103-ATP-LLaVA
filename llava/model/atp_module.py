import torch
import torch.nn as nn
import torch.nn.functional as F


class ATPModule(nn.Module):
    """
    Adaptive Token Pruning Module
    
    Inserts between LLM decoder layers to adaptively prune vision tokens.
    """
    
    def __init__(self, hidden_dim=4096, lambda_sample=3.0, temperature=10.0):
        """
        Args:
            hidden_dim: Hidden dimension of tokens (D)
            lambda_sample: Scaling coefficient for spatial pruning
            temperature: Temperature for soft mask sigmoid
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.lambda_sample = lambda_sample
        self.temperature = temperature
        
        # Learnable threshold prediction heads
        # Input: concatenated mean scores [S_self_mean, S_cross_mean]
        self.score_projection = nn.Linear(2, hidden_dim)
        self.redundant_threshold_head = nn.Linear(hidden_dim, 1)
        self.spatial_threshold_head = nn.Linear(hidden_dim, 1)
        
    def compute_redundant_score(self, attention_logits, attention_weights, 
                                 num_vision_tokens, num_text_tokens):
        """
        Compute redundant pruning scores from self-modality and cross-modality importance.
        
        Args:
            attention_logits: Pre-softmax attention [batch, num_heads, L, L]
            attention_weights: Post-softmax attention [batch, num_heads, L, L]
            num_vision_tokens: L_v
            num_text_tokens: L_t
            
        Returns:
            S_redundant: [batch, L_v] - redundant score for each vision token
        """
        batch_size = attention_logits.shape[0]
        
        # Average across attention heads
        attn_logits = attention_logits.mean(dim=1)  # [batch, L, L]
        attn_weights = attention_weights.mean(dim=1)  # [batch, L, L]
        
        # Extract vision-to-vision attention (self-modality)
        # Vision tokens are typically at the beginning: [0:L_v]
        vision_to_vision_logits = attn_logits[:, :num_vision_tokens, :num_vision_tokens]
        
        # Self-modality importance score (Eq. 3)
        # How much attention each vision token receives from other vision tokens
        S_self = vision_to_vision_logits.mean(dim=1)  # [batch, L_v]
        
        # Extract text-to-vision attention (cross-modality)
        # Text tokens come after vision tokens: [L_v:L_v+L_t]
        text_to_vision_weights = attn_weights[:, num_vision_tokens:num_vision_tokens+num_text_tokens, 
                                               :num_vision_tokens]
        
        # Cross-modality importance score (Eq. 4)
        # How much attention each vision token receives from text tokens
        S_cross = text_to_vision_weights.mean(dim=1)  # [batch, L_v]
        
        # Combined redundant score
        S_redundant = (S_self + S_cross) / 2.0  # [batch, L_v]
        
        return S_redundant, S_self, S_cross
    
    def compute_spatial_score(self, vision_tokens, sampling_rate=0.5):
        """
        Compute spatial pruning scores via uniform spatial sampling.
        
        Args:
            vision_tokens: [batch, L_v, D]
            sampling_rate: R_s - ratio of tokens to sample uniformly
            
        Returns:
            S_spatial: [batch, L_v] - spatial score for each token
            sampled_indices: [batch, num_sampled] - indices of sampled tokens
        """
        batch_size, L_v, _ = vision_tokens.shape
        
        # Assume vision tokens form a square grid (e.g., 24x24=576)
        grid_size = int(L_v ** 0.5)
        assert grid_size * grid_size == L_v, "Vision tokens must form square grid"
        
        # Uniform spatial sampling (every k-th token in 2D grid)
        stride = int(1.0 / sampling_rate)
        sampled_indices = []
        
        for i in range(0, grid_size, stride):
            for j in range(0, grid_size, stride):
                idx = i * grid_size + j
                if idx < L_v:
                    sampled_indices.append(idx)
        
        sampled_indices = torch.tensor(sampled_indices, device=vision_tokens.device)
        
        # Initialize spatial scores (Eq. 5)
        S_spatial = torch.zeros(batch_size, L_v, device=vision_tokens.device)
        
        # Tokens sampled at higher rates get higher scores
        # Sampled tokens: score = 1 - R_s * lambda_sample
        S_spatial[:, sampled_indices] = 1.0 - sampling_rate * self.lambda_sample
        
        return S_spatial, sampled_indices
    
    def predict_thresholds(self, S_self, S_cross):
        """
        Predict learnable pruning thresholds (Eq. 6-8).
        
        Args:
            S_self: [batch, L_v] - self-modality scores
            S_cross: [batch, L_v] - cross-modality scores
            
        Returns:
            theta_r: [batch, 1] - redundant pruning threshold
            theta_s: [batch, 1] - spatial pruning threshold
        """
        # Aggregate scores by taking mean across tokens
        S_self_mean = S_self.mean(dim=1, keepdim=True)  # [batch, 1]
        S_cross_mean = S_cross.mean(dim=1, keepdim=True)  # [batch, 1]
        
        # Concatenate and project (Eq. 6)
        score_input = torch.cat([S_self_mean, S_cross_mean], dim=1)  # [batch, 2]
        z = self.score_projection(score_input)  # [batch, hidden_dim]
        
        # Predict thresholds (Eq. 7-8)
        theta_r = torch.sigmoid(self.redundant_threshold_head(z))  # [batch, 1]
        theta_s = torch.sigmoid(self.spatial_threshold_head(z))  # [batch, 1]
        
        return theta_r, theta_s
    
    def generate_soft_mask(self, S_redundant, S_spatial, theta_r, theta_s):
        """
        Generate differentiable soft masks for training (Eq. 9-11).
        
        Args:
            S_redundant: [batch, L_v]
            S_spatial: [batch, L_v]
            theta_r: [batch, 1]
            theta_s: [batch, 1]
            
        Returns:
            Mask_final: [batch, L_v] - soft mask in [0, 1]
        """
        # Soft masks using sigmoid with temperature
        Mask_r = torch.sigmoid((S_redundant - theta_r) * self.temperature)  # [batch, L_v]
        Mask_s = torch.sigmoid((S_spatial - theta_s) * self.temperature)  # [batch, L_v]
        
        # Element-wise max: keep token if EITHER condition met
        Mask_final = torch.max(Mask_r, Mask_s)  # [batch, L_v]
        
        return Mask_final
    
    def generate_hard_mask(self, S_redundant, S_spatial, theta_r, theta_s):
        """
        Generate hard binary masks for inference.
        
        Args:
            S_redundant: [batch, L_v]
            S_spatial: [batch, L_v]
            theta_r: [batch, 1]
            theta_s: [batch, 1]
            
        Returns:
            Mask_final: [batch, L_v] - binary mask (True/False)
        """
        # Hard thresholding
        Mask_r = S_redundant > theta_r  # [batch, L_v]
        Mask_s = S_spatial > theta_s  # [batch, L_v]
        
        # Logical OR: keep if either condition met
        Mask_final = Mask_r | Mask_s  # [batch, L_v]
        
        return Mask_final
    
    def forward(self, vision_tokens, text_tokens, attention_logits, attention_weights,
                position_ids=None, training=True):
        """
        Forward pass of ATP module.
        
        Args:
            vision_tokens: [batch, L_v, D] - vision token hidden states
            text_tokens: [batch, L_t, D] - text token hidden states
            attention_logits: [batch, num_heads, L, L] - pre-softmax attention from previous layer
            attention_weights: [batch, num_heads, L, L] - post-softmax attention from previous layer
            position_ids: [batch, L_v] - original position IDs of vision tokens
            training: bool - training mode or inference mode
            
        Returns:
            dict with keys:
                - 'vision_tokens': vision tokens (all in training, pruned in inference)
                - 'text_tokens': text tokens (unchanged)
                - 'vision_mask': soft mask (training only)
                - 'position_ids': position IDs (inference only)
                - 'pruning_stats': dict with pruning statistics
        """
        batch_size, L_v, D = vision_tokens.shape
        _, L_t, _ = text_tokens.shape
        
        # Step 1: Compute redundant pruning scores
        S_redundant, S_self, S_cross = self.compute_redundant_score(
            attention_logits, attention_weights, L_v, L_t
        )
        
        # Step 2: Compute spatial pruning scores
        S_spatial, sampled_indices = self.compute_spatial_score(vision_tokens)
        
        # Step 3: Predict learnable thresholds
        theta_r, theta_s = self.predict_thresholds(S_self, S_cross)
        
        # Step 4: Generate masks
        if training:
            # Training: soft masks for gradient flow
            Mask_final = self.generate_soft_mask(S_redundant, S_spatial, theta_r, theta_s)
            
            return {
                'vision_tokens': vision_tokens,  # Keep ALL tokens
                'text_tokens': text_tokens,
                'vision_mask': Mask_final,  # Soft mask for attention
                'pruning_stats': {
                    'theta_r': theta_r.mean().item(),
                    'theta_s': theta_s.mean().item(),
                    'avg_mask_value': Mask_final.mean().item(),
                    'estimated_kept_tokens': (Mask_final > 0.5).sum(dim=1).float().mean().item()
                }
            }
        else:
            # Inference: hard masks, physically drop tokens
            Mask_final = self.generate_hard_mask(S_redundant, S_spatial, theta_r, theta_s)
            
            # Prune vision tokens
            # Handle batching by iterating (in practice, batch_size=1 for inference)
            pruned_vision_tokens = []
            pruned_position_ids = []
            
            for b in range(batch_size):
                mask_b = Mask_final[b]  # [L_v]
                kept_indices = torch.where(mask_b)[0]
                
                # Keep only selected tokens
                pruned_vision_tokens.append(vision_tokens[b, kept_indices])
                
                # Preserve original position IDs
                if position_ids is not None:
                    pruned_position_ids.append(position_ids[b, kept_indices])
                else:
                    # If not provided, assume sequential [0, 1, 2, ...]
                    original_pos = torch.arange(L_v, device=vision_tokens.device)
                    pruned_position_ids.append(original_pos[kept_indices])
            
            # Stack back (assuming same number kept per batch for simplicity)
            # In practice, you might need padding for variable lengths
            pruned_vision_tokens = torch.stack(pruned_vision_tokens, dim=0)
            pruned_position_ids = torch.stack(pruned_position_ids, dim=0)
            
            return {
                'vision_tokens': pruned_vision_tokens,  # PRUNED tokens
                'text_tokens': text_tokens,
                'position_ids': pruned_position_ids,  # Original positions preserved
                'pruning_stats': {
                    'theta_r': theta_r.mean().item(),
                    'theta_s': theta_s.mean().item(),
                    'kept_tokens': pruned_vision_tokens.shape[1],
                    'original_tokens': L_v,
                    'pruning_ratio': 1.0 - (pruned_vision_tokens.shape[1] / L_v)
                }
            }