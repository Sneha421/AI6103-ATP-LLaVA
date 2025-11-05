import torch
import torch.nn as nn
import torch.nn.functional as F

class ATPModule(nn.Module):
    """
    Adaptive Token Pruning Module from ATP-LLaVA paper.
    
    Inserts between LLM decoder layers to adaptively prune vision tokens
    based on redundancy and spatial importance scores.
    """
    
    def __init__(self, hidden_dim=4096, lambda_sample=3.0, temperature=10.0, 
                 sampling_rate=0.25, layer_idx=None, total_layers=32):
        """
        Args:
            hidden_dim: Hidden dimension of tokens (D)
            lambda_sample: Scaling coefficient for spatial pruning (default: 3.0)
            temperature: Temperature for soft mask sigmoid (default: 10.0)
            sampling_rate: Ratio of tokens to sample uniformly (default: 0.25)
                          If None and layer_idx provided, uses layer-adaptive rate
            layer_idx: Current layer index (0-31 for LLaMA-7B)
            total_layers: Total number of layers in LLM (default: 32)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.lambda_sample = lambda_sample
        self.temperature = temperature
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        
        # Determine sampling rate (layer-adaptive if layer_idx provided)
        if sampling_rate is None and layer_idx is not None:
            self.sampling_rate = self._get_layer_adaptive_sampling_rate(layer_idx, total_layers)
        else:
            self.sampling_rate = sampling_rate if sampling_rate is not None else 0.25
        
        # Learnable threshold prediction heads (Eq. 6-8)
        self.score_projection = nn.Linear(2, hidden_dim)
        self.redundant_threshold_head = nn.Linear(hidden_dim, 1)
        self.spatial_threshold_head = nn.Linear(hidden_dim, 1)
    
    @staticmethod
    def _get_layer_adaptive_sampling_rate(layer_idx, total_layers):
        """
        Compute layer-adaptive sampling rate based on layer depth.
        
        Strategy: Shallow layers need more spatial info, deep layers can be pruned more.
        
        Args:
            layer_idx: Current layer (0-indexed)
            total_layers: Total layers in model
            
        Returns:
            sampling_rate: Float in [0.0625, 0.5]
        """
        layer_ratio = layer_idx / total_layers
        
        if layer_ratio < 0.3:  # Shallow layers (0-30% depth)
            return 0.5  # Mild spatial pruning, keep more tokens
        elif layer_ratio < 0.65:  # Middle layers (30-65% depth)
            return 0.25  # Moderate spatial pruning (paper default)
        else:  # Deep layers (65-100% depth)
            return 0.0625  # Aggressive spatial pruning
        
    def compute_redundant_score(self, attention_logits, attention_weights, 
                                 num_vision_tokens, num_text_tokens):
        """
        Compute redundant pruning scores (Eq. 3-4).
        
        Combines self-modality importance (vision-to-vision attention) and
        cross-modality importance (text-to-vision attention).
        
        Args:
            attention_logits: Pre-softmax attention [batch, num_heads, L, L]
            attention_weights: Post-softmax attention [batch, num_heads, L, L]
            num_vision_tokens: L_v
            num_text_tokens: L_t
            
        Returns:
            S_redundant: [batch, L_v] - redundant score per vision token
            S_self: [batch, L_v] - self-modality score
            S_cross: [batch, L_v] - cross-modality score
        """
        # Average across attention heads
        attn_logits = attention_logits.mean(dim=1)  # [batch, L, L]
        attn_weights = attention_weights.mean(dim=1)  # [batch, L, L]
        
        # Self-modality: vision-to-vision attention logits (Eq. 3)
        vision_to_vision_logits = attn_logits[:, :num_vision_tokens, :num_vision_tokens]
        S_self = vision_to_vision_logits.mean(dim=2)  # [batch, L_v]
        
        # Cross-modality: text-to-vision attention weights (Eq. 4)
        text_to_vision_weights = attn_weights[:, num_vision_tokens:num_vision_tokens+num_text_tokens, 
                                               :num_vision_tokens]
        S_cross = text_to_vision_weights.mean(dim=1)  # [batch, L_v]
        
        # Combined redundant score
        S_redundant = (S_self + S_cross) / 2.0
        
        return S_redundant, S_self, S_cross
    
    def compute_spatial_score(self, num_vision_tokens, device):
        """
        Compute spatial pruning scores via uniform 2D sampling (Eq. 5).
        
        Sampled tokens get score: 1 - R_s * λ_sample
        Non-sampled tokens get score: 0
        
        Args:
            num_vision_tokens: L_v (e.g., 576 for 24×24 grid)
            device: torch device
            
        Returns:
            S_spatial: [L_v] - spatial score for each token
        """
        L_v = num_vision_tokens
        
        # Determine grid dimensions (handles both square and non-square)
        grid_size = int(L_v ** 0.5)
        if grid_size * grid_size != L_v:
            grid_h = int(L_v ** 0.5)
            grid_w = (L_v + grid_h - 1) // grid_h
        else:
            grid_h = grid_w = grid_size
        
        S_spatial = torch.zeros(L_v, device=device)
        
        # Compute stride for uniform sampling: stride = sqrt(1 / R_s)
        # E.g., R_s=0.25 → stride=2 → samples every 2nd token (12×12 from 24×24)
        stride = max(1, int(round((1.0 / self.sampling_rate) ** 0.5)))
        
        # Collect sampled indices
        sampled_indices = []
        for i in range(0, grid_h, stride):
            for j in range(0, grid_w, stride):
                idx = i * grid_w + j
                if idx < L_v:
                    sampled_indices.append(idx)
        
        if len(sampled_indices) > 0:
            sampled_indices = torch.tensor(sampled_indices, device=device)
            score_value = max(0.0, 1.0 - self.sampling_rate * self.lambda_sample)
            S_spatial[sampled_indices] = score_value
        
        return S_spatial
    
    def predict_thresholds(self, S_self, S_cross):
        """
        Predict learnable pruning thresholds (Eq. 6-8).
        
        Args:
            S_self: [batch, L_v] - self-modality scores
            S_cross: [batch, L_v] - cross-modality scores
            
        Returns:
            theta_r: [batch, 1] - redundant pruning threshold ∈ [0,1]
            theta_s: [batch, 1] - spatial pruning threshold ∈ [0,1]
        """
        # Aggregate scores and project to hidden dim
        S_self_mean = S_self.mean(dim=1, keepdim=True)  # [batch, 1]
        S_cross_mean = S_cross.mean(dim=1, keepdim=True)  # [batch, 1]
        score_input = torch.cat([S_self_mean, S_cross_mean], dim=1)  # [batch, 2]
        z = self.score_projection(score_input)  # [batch, hidden_dim]
        
        # Predict thresholds
        theta_r = torch.sigmoid(self.redundant_threshold_head(z))  # [batch, 1]
        theta_s = torch.sigmoid(self.spatial_threshold_head(z))  # [batch, 1]
        
        return theta_r, theta_s
    
    def generate_soft_mask(self, S_redundant, S_spatial, theta_r, theta_s):
        """
        Generate differentiable soft masks for training (Eq. 9-11).
        
        Uses sigmoid with temperature to create smooth gradients.
        """
        Mask_r = torch.sigmoid((S_redundant - theta_r) * self.temperature)
        Mask_s = torch.sigmoid((S_spatial - theta_s) * self.temperature)
        Mask_final = torch.max(Mask_r, Mask_s)  # Element-wise max
        return Mask_final
    
    def generate_hard_mask(self, S_redundant, S_spatial, theta_r, theta_s):
        """
        Generate hard binary masks for inference.
        
        Uses hard thresholding and logical OR.
        """
        Mask_r = S_redundant > theta_r
        Mask_s = S_spatial > theta_s
        Mask_final = Mask_r | Mask_s  # Logical OR
        return Mask_final
    
    def forward(self, vision_tokens, text_tokens, attention_logits, attention_weights,
                position_ids=None, training=True):
        """
        Forward pass of ATP module.
        
        Args:
            vision_tokens: [batch, L_v, D] - vision token hidden states
            text_tokens: [batch, L_t, D] - text token hidden states
            attention_logits: [batch, num_heads, L, L] - pre-softmax attention
            attention_weights: [batch, num_heads, L, L] - post-softmax attention
            position_ids: [batch, L_v] - original 2D position IDs (optional)
            training: bool - use soft masks (True) or hard pruning (False)
            
        Returns:
            dict containing:
                - vision_tokens: [batch, L_v, D] (training) or [batch, L_p_v, D] (inference)
                - text_tokens: [batch, L_t, D]
                - vision_mask: [batch, L_v] (training only)
                - position_ids: [batch, L_p_v] (inference only)
                - pruning_stats: dict with metrics
        """
        batch_size, L_v, D = vision_tokens.shape
        _, L_t, _ = text_tokens.shape
        
        # Step 1: Compute redundant scores (Eq. 3-4)
        S_redundant, S_self, S_cross = self.compute_redundant_score(
            attention_logits, attention_weights, L_v, L_t
        )
        
        # Step 2: Compute spatial scores (Eq. 5)
        S_spatial = self.compute_spatial_score(L_v, vision_tokens.device)
        S_spatial = S_spatial.unsqueeze(0).expand(batch_size, -1)  # [batch, L_v]
        
        # Step 3: Predict learnable thresholds (Eq. 6-8)
        theta_r, theta_s = self.predict_thresholds(S_self, S_cross)
        
        # Step 4: Generate masks and prune
        if training:
            # Soft masks for gradient flow
            Mask_final = self.generate_soft_mask(S_redundant, S_spatial, theta_r, theta_s)
            
            return {
                'vision_tokens': vision_tokens,  # Keep all tokens physically
                'text_tokens': text_tokens,
                'vision_mask': Mask_final,  # Soft mask for attention masking
                'pruning_stats': {
                    'theta_r': theta_r.mean().item(),
                    'theta_s': theta_s.mean().item(),
                    'avg_mask_value': Mask_final.mean().item(),
                    'estimated_kept_tokens': (Mask_final > 0.5).sum(dim=1).float().mean().item(),
                    'S_redundant_mean': S_redundant.mean().item(),
                    'S_spatial_mean': S_spatial.mean().item(),
                }
            }
        else:
            # Hard masks - physically drop tokens
            Mask_final = self.generate_hard_mask(S_redundant, S_spatial, theta_r, theta_s)
            
            # Prune vision tokens
            if batch_size == 1:
                kept_indices = torch.where(Mask_final[0])[0]
                pruned_vision_tokens = vision_tokens[0:1, kept_indices]
                
                # Preserve original position IDs
                if position_ids is not None:
                    pruned_position_ids = position_ids[0:1, kept_indices]
                else:
                    original_pos = torch.arange(L_v, device=vision_tokens.device)
                    pruned_position_ids = original_pos[kept_indices].unsqueeze(0)
                    
            else:
                # Batched inference - pad to handle variable lengths
                pruned_vision_tokens = []
                pruned_position_ids = []
                max_kept = 0
                
                for b in range(batch_size):
                    mask_b = Mask_final[b]
                    kept_indices = torch.where(mask_b)[0]
                    max_kept = max(max_kept, len(kept_indices))
                    
                    pruned_vision_tokens.append(vision_tokens[b, kept_indices])
                    
                    if position_ids is not None:
                        pruned_position_ids.append(position_ids[b, kept_indices])
                    else:
                        original_pos = torch.arange(L_v, device=vision_tokens.device)
                        pruned_position_ids.append(original_pos[kept_indices])
                
                # Pad to max length in batch
                padded_vision_tokens = []
                padded_position_ids = []
                
                for b in range(batch_size):
                    tokens = pruned_vision_tokens[b]
                    pos_ids = pruned_position_ids[b]
                    
                    if len(tokens) < max_kept:
                        pad_len = max_kept - len(tokens)
                        tokens = torch.cat([tokens, torch.zeros(pad_len, D, device=tokens.device)], dim=0)
                        pos_ids = torch.cat([pos_ids, torch.zeros(pad_len, dtype=pos_ids.dtype, device=pos_ids.device)], dim=0)
                    
                    padded_vision_tokens.append(tokens)
                    padded_position_ids.append(pos_ids)
                
                pruned_vision_tokens = torch.stack(padded_vision_tokens, dim=0)
                pruned_position_ids = torch.stack(padded_position_ids, dim=0)
            
            return {
                'vision_tokens': pruned_vision_tokens,
                'text_tokens': text_tokens,
                'position_ids': pruned_position_ids,
                'pruning_stats': {
                    'theta_r': theta_r.mean().item(),
                    'theta_s': theta_s.mean().item(),
                    'kept_tokens': pruned_vision_tokens.shape[1],
                    'original_tokens': L_v,
                    'pruning_ratio': 1.0 - (pruned_vision_tokens.shape[1] / L_v),
                    'S_redundant_mean': S_redundant.mean().item(),
                    'S_spatial_mean': S_spatial.mean().item(),
                }
            }

class ATPLoss(nn.Module):
    """
    Budget-Constrained Loss for ATP Module Training (Section 3.3.3 in paper).
    
    PURPOSE:
    Train the ATP module to balance two competing objectives:
    1. Prune as many tokens as possible (reduce computation)
    2. Keep enough tokens to maintain performance (minimize accuracy loss)
    
    The loss encourages the model to learn WHEN to prune aggressively vs. conservatively
    based on the instance and layer.
    
    Key Components:
    - L_atp: Penalty for keeping too many tokens (increases with layer depth)
    - L_target: Penalty for deviating from target token count
    - L_ntp: Next Token Prediction loss (standard LLM loss, not implemented here)
    """
    
    def __init__(
        self,
        lambda_atp=0.05,
        lambda_target=0.2,
        target_tokens=144,
        initial_vision_tokens=576,
    ):
        """
        Args:
            lambda_atp: Weight for ATP penalty term (default: 0.05)
            lambda_target: Weight for target constraint term (default: 0.2)
            target_tokens: Target average number of tokens to keep (default: 144)
            initial_vision_tokens: Initial number of vision tokens (default: 576)
        """
        super().__init__()
        self.lambda_atp = lambda_atp
        self.lambda_target = lambda_target
        self.target_tokens = target_tokens
        self.initial_vision_tokens = initial_vision_tokens
    
    def compute_atp_penalty(self, masks, layer_indices):
        """
        Compute ATP penalty term (Equation 12 in paper).
        
        L_atp = Σ (kept_tokens / 576) * layer_index
        
        WHY THIS FORMULA?
        - Penalizes keeping many tokens: (kept_tokens / 576)
        - Penalizes keeping tokens in deeper layers: * layer_index
        - Deeper layers should prune more aggressively since visual info
          has already been processed into text representations
        
        Args:
            masks: List of soft masks [batch, L_v] from each ATP layer
            layer_indices: List of layer indices where ATP is applied (e.g., [4, 14, 24])
            
        Returns:
            atp_penalty: Scalar tensor
        """
        total_penalty = 0.0
        
        for mask, layer_idx in zip(masks, layer_indices):
            # Sum mask values to get "number" of kept tokens (soft count)
            kept_tokens = mask.sum(dim=1).mean()  # Average across batch

            # Normalize by initial token count and weight by layer depth
            penalty = (kept_tokens / self.initial_vision_tokens) * layer_idx
            total_penalty += penalty
        
        return total_penalty
    
    def compute_target_constraint(self, masks):
        """
        Compute target constraint loss (Equation 13 in paper).
        
        L_target = |N - N_target|
        
        WHERE:
        - N: Average number of kept tokens across all ATP layers
        - N_target: Target token count (e.g., 144)
        
        WHY?
        Constrains the AVERAGE token count to match a specific budget.
        This prevents the model from either:
        - Keeping too many tokens (defeats the purpose)
        - Pruning too aggressively (hurts performance)
        
        Args:
            masks: List of soft masks [batch, L_v] from each ATP layer
            
        Returns:
            target_loss: Scalar tensor
        """
        # Compute average kept tokens across all ATP layers
        total_kept = 0.0
        for mask in masks:
            kept_tokens = mask.sum(dim=1).mean()  # Average across batch
            total_kept += kept_tokens
        
        avg_kept = total_kept / len(masks)
        
        # L1 distance from target
        target_loss = torch.abs(avg_kept - self.target_tokens)
        
        return target_loss
    
    def forward(self, masks, layer_indices):
        """
        Compute total ATP loss (Equation 14 in paper).
        
        L = L_ntp + λ_atp * L_atp + λ_target * L_target
        
        NOTE: L_ntp (next token prediction) is the standard LLM loss computed
        separately. This function only returns the ATP-specific losses.
        
        Args:
            masks: List of soft masks [batch, L_v] from each ATP layer
            layer_indices: List of layer indices where ATP is applied
            
        Returns:
            loss_dict: Dictionary with individual loss components
        """
        # === Equation 12: ATP Penalty ===
        atp_penalty = self.compute_atp_penalty(masks, layer_indices)
        
        # === Equation 13: Target Constraint ===
        target_loss = self.compute_target_constraint(masks)
        
        # === Equation 14: Combined ATP Loss ===
        total_atp_loss = (
            self.lambda_atp * atp_penalty + 
            self.lambda_target * target_loss
        )
        
        return {
            'atp_penalty': atp_penalty,
            'target_loss': target_loss,
            'total_atp_loss': total_atp_loss,
            'lambda_atp': self.lambda_atp,
            'lambda_target': self.lambda_target,
        }

