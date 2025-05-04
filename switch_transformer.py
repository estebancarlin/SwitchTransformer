import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


def truncated_normal_(tensor, mean=0.0, std=1.0):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

class SwitchTransformerConfig:
    """
    Configuration for the Switch Transformer model.
    """
    def __init__(self, 
                 vocab_size=30522,      # vocabulary size (e.g., 30522 for BERT's vocab)
                 d_model=512,           # hidden size / model dimensionality
                 num_heads=8,           # number of attention heads
                 num_layers=6,          # number of Transformer layers
                 d_ff=2048,             # hidden size of each expert's feed-forward network
                 num_experts=4,         # number of experts in each Switch FFN layer
                 dropout=0.1,           # dropout rate for attention outputs and residuals
                 expert_dropout=None,   # dropout rate inside experts (None -> same as dropout)
                 aux_loss_alpha=0.01,   # coefficient (alpha) for auxiliary load balancing loss
                 capacity_factor=1.0):  # capacity factor for expert routing (unused in this single-GPU demo)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.dropout = dropout
        # Use the same dropout for experts if not specified; during fine-tuning, this can be set higher for expert dropout
        self.expert_dropout = dropout if expert_dropout is None else expert_dropout
        self.aux_loss_alpha = aux_loss_alpha
        self.capacity_factor = capacity_factor

class ExpertFFN(nn.Module):
    """
    Feed-Forward Network for a single expert.
    This is a two-layer MLP with ReLU activation and dropout.
    """   
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(ExpertFFN, self).__init__()
        self.w_in = nn.Linear(d_model, d_ff)
        self.w_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        truncated_normal_(self.w_in.weight, std=math.sqrt(0.1 / d_model))
        truncated_normal_(self.w_out.weight, std=math.sqrt(0.1 / d_ff))
        nn.init.zeros_(self.w_in.bias)
        nn.init.zeros_(self.w_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.w_in(x))
        hidden = self.dropout(hidden)
        out = self.w_out(hidden)
        return out

class SwitchFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int,
                 dropout: float, expert_dropout: float, aux_loss_alpha: float, capacity_factor: float):
        super(SwitchFFN, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff, dropout=expert_dropout) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)
        self.output_dropout = nn.Dropout(dropout)
        self.aux_loss_alpha = aux_loss_alpha
        self.capacity_factor = capacity_factor

        truncated_normal_(self.router.weight, std=math.sqrt(0.1 / d_model))
        nn.init.zeros_(self.router.bias)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor, float):
        batch_size, seq_len, d_model = x.size()
        x_flat = x.view(-1, d_model)
        total_tokens = x_flat.size(0)

        # Selective precision: router input in float32
        router_input = x_flat.float()  # Cast to float32 for stable softmax
        logits = self.router(router_input)
        probs = F.softmax(logits, dim=-1)
        # print("probs:", probs.shape)
        top1_indices = torch.argmax(probs, dim=-1)
        # print("top1_indices:", top1_indices.shape)
        capacity = int((total_tokens / self.num_experts) * self.capacity_factor)

        output_flat = torch.zeros_like(x_flat)

        counts = torch.bincount(top1_indices, minlength=self.num_experts)
        # print("counts:", counts.shape)
        f_i = counts.to(x_flat.dtype) / float(total_tokens)
        prob_sums = probs.sum(dim=0) / float(total_tokens)
        P_i = prob_sums
        aux_loss = self.aux_loss_alpha * self.num_experts * torch.dot(f_i, P_i)
        
        dropped_tokens = 0
        routed_tokens = 0

        for expert_idx in range(self.num_experts):
            mask = (top1_indices == expert_idx)
            selected = torch.nonzero(mask, as_tuple=False).view(-1)
            num_selected = selected.numel()
            if num_selected == 0:
                continue
            routed_tokens += num_selected
            if num_selected > capacity:
                dropped_tokens += (num_selected - capacity)
                selected = selected[:capacity]

            expert_input = x_flat[selected]
            expert_output = self.experts[expert_idx](expert_input)
            expert_prob = probs[selected, expert_idx].unsqueeze(1)
            expert_output = expert_output * expert_prob
            output_flat[selected] = expert_output

        total_routed = routed_tokens if routed_tokens > 0 else 1
        drop_ratio = dropped_tokens / total_routed

        output = output_flat.view(batch_size, seq_len, d_model)
        output = self.output_dropout(output)
        
        return output, aux_loss, drop_ratio, counts.detach()

class SwitchTransformerBlock(nn.Module):
    """
    A Transformer block that includes multi-head attention followed by a Switch FFN layer.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_experts: int, 
                 dropout: float, expert_dropout: float, aux_loss_alpha: float, capacity_factor: float):
        super(SwitchTransformerBlock, self).__init__()
        # Multi-head self-attention layer
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        # LayerNorms (pre-normalization is used for stability)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Switch FFN layer (mixture-of-experts FFN)
        self.switch_ffn = SwitchFFN(d_model, d_ff, num_experts, dropout, expert_dropout,
                            aux_loss_alpha, capacity_factor)
        # Dropout for attention output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, float):
        """
        Forward pass for the Transformer block.
        :param x: Tensor of shape (batch_size, seq_len, d_model)
        :param attention_mask: Optional tensor of shape (batch_size, seq_len) with 1 for real tokens and 0 for padding.
                               This is used to avoid attending to padding tokens.
        :return: (y, aux_loss_block) where y is the output tensor (batch_size, seq_len, d_model),
                 aux_loss_block is the auxiliary loss from the Switch FFN layer in this block.
        """
        # Apply LayerNorm to input for attention (Pre-Norm)
        x_norm = self.norm1(x)
        # Prepare input for nn.MultiheadAttention: shape should be (seq_len, batch_size, d_model)
        x_transposed = x_norm.transpose(0, 1)  # (seq_len, batch_size, d_model)
        # Prepare key padding mask if attention_mask is provided (True for positions to mask out)
        key_padding_mask = None
        if attention_mask is not None:
            # In key_padding_mask, True indicates the position is invalid (to be masked)
            key_padding_mask = ~(attention_mask.to(torch.bool))
        # Self-attention: we use the same data for query, key, and value (self-attention)
        attn_output, _ = self.self_attn(x_transposed, x_transposed, x_transposed, 
                                        key_padding_mask=key_padding_mask, need_weights=False)
        # attn_output is shape (seq_len, batch_size, d_model); transpose back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(0, 1)
        # Apply dropout to attention output and add residual connection
        x = x + self.dropout(attn_output)
        # Apply LayerNorm before Switch FFN (Pre-Norm)
        x_norm2 = self.norm2(x)
        # Switch FFN forward pass
        ffn_output, aux_loss, drop_ratio, expert_counts = self.switch_ffn(x_norm2)
        # print("expert_counts:", expert_counts.shape)
        # Add FFN output with residual connection
        x = x + ffn_output
        # Return the output and the auxiliary loss for this block
        return x, aux_loss, drop_ratio, expert_counts.float()
    
class DenseTransformerBlock(nn.Module):
    """
    A Transformer block with a standard dense FFN instead of SwitchFFN.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super(DenseTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        truncated_normal_(self.ffn[0].weight, std=math.sqrt(0.1 / d_model))
        truncated_normal_(self.ffn[3].weight, std=math.sqrt(0.1 / d_ff))
        nn.init.zeros_(self.ffn[0].bias)
        nn.init.zeros_(self.ffn[3].bias)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        x_norm = self.norm1(x)
        x_transposed = x_norm.transpose(0, 1)
        key_padding_mask = ~(attention_mask.to(torch.bool)) if attention_mask is not None else None
        attn_output, _ = self.self_attn(x_transposed, x_transposed, x_transposed,
                                        key_padding_mask=key_padding_mask, need_weights=False)
        attn_output = attn_output.transpose(0, 1)
        x = x + self.dropout(attn_output)
        x_norm2 = self.norm2(x)
        ffn_output = self.ffn(x_norm2)
        x = x + ffn_output
        return x, torch.tensor(0.0, device=x.device)  # No aux loss for dense blocks

class SwitchTransformerLM(nn.Module):
    """
    Switch Transformer model for Masked Language Modeling (encoder-only Transformer with Switch FFN layers).
    """
    def __init__(self, config: SwitchTransformerConfig):
        super(SwitchTransformerLM, self).__init__()
        self.config = config
        # Token embedding table
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        # Positional embedding table (we use learnable positional embeddings up to a fixed max length, e.g., 512)
        self.pos_emb = nn.Embedding(512, config.d_model)
        # Transformer layers
        # self.layers = nn.ModuleList([
        #     SwitchTransformerBlock(config.d_model, config.num_heads, config.d_ff, config.num_experts,
        #                         dropout=config.dropout, expert_dropout=config.expert_dropout,
        #                         aux_loss_alpha=config.aux_loss_alpha, capacity_factor=config.capacity_factor)
        #     if i % 2 == 0 else
        #     DenseTransformerBlock(config.d_model, config.num_heads, config.d_ff, dropout=config.dropout)
        #     for i in range(config.num_layers)
        # ])
        self.layers = nn.ModuleList([
            checkpoint_wrapper(
                SwitchTransformerBlock(config.d_model, config.num_heads, config.d_ff, config.num_experts,
                                    dropout=config.dropout, expert_dropout=config.expert_dropout,
                                    aux_loss_alpha=config.aux_loss_alpha, capacity_factor=config.capacity_factor))
            if i % 2 == 0 else
            checkpoint_wrapper(
                DenseTransformerBlock(config.d_model, config.num_heads, config.d_ff, dropout=config.dropout))
            for i in range(config.num_layers)
        ])

        # Final LayerNorm after all layers (Post-norm for the output)
        self.final_ln = nn.LayerNorm(config.d_model)
        # Output projection layer to vocabulary size (tied with token embedding weights for efficiency)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_layer.weight = self.token_emb.weight  # tie output layer weight with input embedding

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor, float):
        """
        Forward pass of the SwitchTransformerLM.
        :param input_ids: Tensor of token IDs of shape (batch_size, seq_len)
        :param attention_mask: Optional tensor of shape (batch_size, seq_len) indicating which tokens are real (1) vs padding (0).
        :return: (logits, total_aux_loss) where:
                 logits is the model output for MLM (batch_size, seq_len, vocab_size),
                 total_aux_loss is the sum of auxiliary losses from all Switch layers.
        """
        
        batch_size, seq_len = input_ids.size()
        # Create position indices for the sequence length
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        # Compute token and position embeddings, then sum them to get the initial hidden states
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        # (Optionally, one could apply dropout to the embeddings here for regularization)

        total_aux_loss = 0.0  # accumulate auxiliary loss from each Switch layer
        
        # Pass through each Transformer layer
        drop_ratios = []
        expert_loads = []
        
        # if not x.requires_grad:
        #     x.requires_grad = True
            
        for layer in self.layers:
            # def custom_forward(*inputs):
            #     return layer(*inputs)
            # result = checkpoint(custom_forward, x, attention_mask, use_reentrant=False)
            
            result = layer(x, attention_mask)
            
            if isinstance(result, tuple) and len(result) == 4:
                x, layer_aux, drop_ratio, expert_counts = result
                # print("expert_counts:", expert_counts.shape)
                drop_ratios.append(drop_ratio)
                expert_loads.append(expert_counts)
                # print("expert_loads:", expert_loads[-1].shape)
                # print("expert_loads_list:", len(expert_loads))
            else:
                x, layer_aux = result
            total_aux_loss += layer_aux

        avg_drop_ratio = sum(drop_ratios) / len(drop_ratios) if drop_ratios else 0.0
        # print("avg_drop_ratio:", avg_drop_ratio)
        
        if expert_loads:
            avg_expert_load = torch.stack(expert_loads).mean(dim=0)
            # print("avg_expert_load_if:", avg_expert_load.shape)
        else:
            avg_expert_load = torch.zeros(self.config.num_experts, device=x.device)
            # print("avg_expert_load_else:", avg_expert_load.shape)

        # Final layer normalization
        x = self.final_ln(x)
        # Compute logits over vocabulary
        logits = self.output_layer(x)
        
        return logits, total_aux_loss, avg_drop_ratio, avg_expert_load
