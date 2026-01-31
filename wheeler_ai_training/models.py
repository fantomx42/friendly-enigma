"""
Wheeler AI - Text Encoder/Decoder with 2D Spatial Latent
=========================================================

Architecture:
- Encoder: Text → 2D Grid (128x128)
  Uses Perceiver-style cross-attention to map variable-length
  text into a fixed 2D spatial representation
  
- Decoder: 2D Grid → Text (autoregressive)
  Cross-attends to flattened spatial features while generating
  text token-by-token

This bridges the gap between text and Wheeler Memory frames.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat


# =============================================================================
# POSITIONAL ENCODINGS
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        
        # Create position encodings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]


class LearnedPositionalEncoding2D(nn.Module):
    """Learnable 2D positional encoding for the spatial grid."""
    
    def __init__(self, height: int, width: int, d_model: int):
        super().__init__()
        
        self.height = height
        self.width = width
        
        # Separate row and column embeddings (more efficient than full grid)
        self.row_embed = nn.Embedding(height, d_model // 2)
        self.col_embed = nn.Embedding(width, d_model // 2)
    
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate 2D positional encodings.
        
        Returns: (batch, height, width, d_model)
        """
        rows = torch.arange(self.height, device=device)
        cols = torch.arange(self.width, device=device)
        
        row_emb = self.row_embed(rows)  # (H, d/2)
        col_emb = self.col_embed(cols)  # (W, d/2)
        
        # Combine into 2D grid
        row_emb = row_emb.unsqueeze(1).expand(-1, self.width, -1)  # (H, W, d/2)
        col_emb = col_emb.unsqueeze(0).expand(self.height, -1, -1)  # (H, W, d/2)
        
        pos = torch.cat([row_emb, col_emb], dim=-1)  # (H, W, d)
        pos = pos.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, H, W, d)
        
        return pos


class FourierPositionalEncoding2D(nn.Module):
    """Fourier feature positional encoding for 2D grids."""
    
    def __init__(self, height: int, width: int, d_model: int, num_bands: int = 32):
        super().__init__()
        
        self.height = height
        self.width = width
        self.d_model = d_model
        
        # Learnable frequency bands
        self.freq_bands = nn.Parameter(torch.randn(num_bands, 2) * 0.1)
        
        # Project to d_model
        self.proj = nn.Linear(num_bands * 4, d_model)  # sin + cos for x and y
    
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate Fourier positional encodings."""
        # Normalized coordinates
        y = torch.linspace(-1, 1, self.height, device=device)
        x = torch.linspace(-1, 1, self.width, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        coords = torch.stack([yy, xx], dim=-1)  # (H, W, 2)
        
        # Apply frequency bands
        freq = self.freq_bands.to(device)  # (num_bands, 2)
        
        # Compute Fourier features
        proj_coords = coords @ freq.T  # (H, W, num_bands)
        
        features = torch.cat([
            torch.sin(proj_coords * math.pi),
            torch.cos(proj_coords * math.pi),
        ], dim=-1)  # (H, W, num_bands * 2)
        
        # Repeat for x and y dimensions
        features = torch.cat([features, features], dim=-1)  # (H, W, num_bands * 4)
        
        # Project to d_model
        pos = self.proj(features)  # (H, W, d_model)
        pos = pos.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return pos


# =============================================================================
# ATTENTION MODULES
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_v, d_model)
            mask: Optional attention mask
            is_causal: If True, apply causal masking
        
        Returns:
            (batch, seq_q, d_model)
        """
        batch_size = query.size(0)
        
        # Project
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        if is_causal:
            seq_len = query.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float('-inf'))
        
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention where queries come from one source and keys/values from another."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: What we're computing attention FOR (batch, seq_q, d_model)
            context: What we're attending TO (batch, seq_ctx, d_model)
        """
        q = self.norm_q(query)
        ctx = self.norm_kv(context)
        return self.attn(q, ctx, ctx, mask=mask)


# =============================================================================
# ENCODER: Text → 2D Grid
# =============================================================================

class TextToGridEncoder(nn.Module):
    """
    Encodes variable-length text into a fixed 2D spatial grid.
    
    Uses Perceiver-style cross-attention:
    1. Embed text tokens
    2. Process through transformer layers
    3. Learnable grid queries cross-attend to text features
    4. Output: (batch, height, width, d_model)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        grid_height: int = 64,
        grid_width: int = 64,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_grid_tokens = grid_height * grid_width
        
        # Text embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEncoding(d_model, max_seq_len)
        
        # Text processing (self-attention layers)
        self.text_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers // 2)
        ])
        
        # Learnable grid queries (the "perceiver" part)
        # These will cross-attend to the text features
        self.grid_queries = nn.Parameter(torch.randn(1, self.num_grid_tokens, d_model) * 0.02)
        
        # 2D positional encoding for grid
        self.grid_pos = LearnedPositionalEncoding2D(grid_height, grid_width, d_model)
        
        # Cross-attention layers: grid queries attend to text
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(d_model, num_heads, dropout)
            for _ in range(num_layers // 2)
        ])
        
        # Self-attention on grid (local refinement)
        self.grid_self_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers // 4)
        ])
        
        # Final projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding
        
        Returns:
            (batch, height, width, d_model) spatial representation
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Embed text
        x = self.token_embed(input_ids)
        x = self.pos_embed(x)
        
        # Process text through self-attention
        for layer in self.text_layers:
            x = layer(x)
        
        # Prepare grid queries
        grid_q = self.grid_queries.expand(batch_size, -1, -1)  # (B, H*W, d)
        
        # Add 2D positional encoding to queries
        grid_pos = self.grid_pos(batch_size, device)  # (B, H, W, d)
        grid_pos_flat = rearrange(grid_pos, 'b h w d -> b (h w) d')
        grid_q = grid_q + grid_pos_flat
        
        # Cross-attend: grid queries attend to text features
        for cross_attn in self.cross_attn_layers:
            # Residual connection
            grid_q = grid_q + cross_attn(grid_q, x)
        
        # Self-attention on grid (refine spatial relationships)
        for self_attn in self.grid_self_attn:
            grid_q = self_attn(grid_q)
        
        # Reshape to 2D grid
        grid = rearrange(grid_q, 'b (h w) d -> b h w d', h=self.grid_height, w=self.grid_width)
        
        # Final projection
        grid = self.output_proj(grid)
        grid = self.output_norm(grid)
        
        return grid


# =============================================================================
# DECODER: 2D Grid → Text (Autoregressive)
# =============================================================================

class GridToTextDecoder(nn.Module):
    """
    Decodes a 2D spatial grid back to text autoregressively.
    
    Architecture:
    1. Flatten grid to sequence
    2. For each generated token:
       - Self-attention over previously generated tokens (causal)
       - Cross-attention to grid features
       - Predict next token
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        grid_height: int = 64,
        grid_width: int = 64,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embedding for decoder input
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEncoding(d_model, max_seq_len)
        
        # Grid position encoding (for cross-attention)
        self.grid_pos = LearnedPositionalEncoding2D(grid_height, grid_width, d_model)
        
        # Grid projection
        self.grid_proj = nn.Linear(d_model, d_model)
        
        # Decoder layers: self-attention + cross-attention
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(
        self,
        grid: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Training mode: Teacher forcing with target sequence.
        
        Args:
            grid: (batch, height, width, d_model) spatial features
            target_ids: (batch, seq_len) target token IDs
            target_mask: (batch, seq_len) attention mask
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size = grid.size(0)
        device = grid.device
        
        # Flatten grid and add position encoding
        grid_pos = self.grid_pos(batch_size, device)
        grid = grid + grid_pos
        grid_flat = rearrange(grid, 'b h w d -> b (h w) d')
        grid_flat = self.grid_proj(grid_flat)
        
        # Embed target tokens
        x = self.token_embed(target_ids)
        x = self.pos_embed(x)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, grid_flat)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        grid: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            grid: (batch, height, width, d_model) spatial features
            start_token_id: ID of start/BOS token
            end_token_id: ID of end/EOS token
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
        
        Returns:
            generated_ids: (batch, seq_len) generated token IDs
        """
        batch_size = grid.size(0)
        device = grid.device
        
        # Flatten grid
        grid_pos = self.grid_pos(batch_size, device)
        grid = grid + grid_pos
        grid_flat = rearrange(grid, 'b h w d -> b (h w) d')
        grid_flat = self.grid_proj(grid_flat)
        
        # Start with BOS token
        generated = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Embed current sequence
            x = self.token_embed(generated)
            x = self.pos_embed(x)
            
            # Apply decoder layers
            for layer in self.layers:
                x = layer(x, grid_flat)
            
            # Get logits for last position
            x = self.output_norm(x)
            logits = self.output_proj(x[:, -1, :])  # (batch, vocab)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if (next_token == end_token_id).all():
                break
        
        return generated


class DecoderLayer(nn.Module):
    """Single decoder layer with causal self-attention and cross-attention."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Causal self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        # Cross-attention to grid
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model) decoder input
            context: (batch, grid_len, d_model) grid features
        """
        # Causal self-attention
        residual = x
        x = self.self_attn_norm(x)
        x = residual + self.self_attn(x, x, x, is_causal=True)
        
        # Cross-attention to grid
        residual = x
        x = self.cross_attn_norm(x)
        x = residual + self.cross_attn(x, context, context)
        
        # Feed-forward
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)
        
        return x


# =============================================================================
# COMPLETE MODEL: Wheeler Text Autoencoder
# =============================================================================

class WheelerTextAutoencoder(nn.Module):
    """
    Complete text autoencoder with 2D spatial latent space.
    
    Text → Encoder → 2D Grid → Decoder → Text
    
    The 2D grid is compatible with Wheeler Memory dynamics.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 6,
        grid_size: int = 64,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.grid_size = grid_size
        
        self.encoder = TextToGridEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=encoder_layers,
            grid_height=grid_size,
            grid_width=grid_size,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        self.decoder = GridToTextDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=decoder_layers,
            grid_height=grid_size,
            grid_width=grid_size,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.
        
        Args:
            input_ids: (batch, seq_len) input text
            target_ids: (batch, seq_len) target text (usually shifted input)
            input_mask: Optional attention mask for input
            target_mask: Optional attention mask for target
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            grid: (batch, height, width, d_model) latent representation
        """
        # Encode to grid
        grid = self.encoder(input_ids, input_mask)
        
        # Decode to text
        logits = self.decoder(grid, target_ids, target_mask)
        
        return logits, grid
    
    def encode(self, input_ids: torch.Tensor, input_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text to 2D grid."""
        return self.encoder(input_ids, input_mask)
    
    def decode(
        self,
        grid: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_length: int = 256,
        **kwargs
    ) -> torch.Tensor:
        """Decode 2D grid to text."""
        return self.decoder.generate(grid, start_token_id, end_token_id, max_length, **kwargs)
    
    def get_grid_as_image(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Convert grid to image-like tensor for Wheeler Memory.
        
        Takes the mean across d_model channels to get (batch, H, W) in [-1, 1].
        """
        # Mean across channels
        img = grid.mean(dim=-1)  # (batch, H, W)
        
        # Normalize to [-1, 1]
        img = img / (img.abs().max() + 1e-8)
        
        return img
    
    def set_grid_from_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert Wheeler Memory frame back to grid format.
        
        Expands single-channel image to d_model channels.
        """
        # img: (batch, H, W)
        # Expand to d_model channels
        grid = img.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
        return grid


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Wheeler Text Autoencoder - Architecture Test")
    print("=" * 50)
    
    # Create model
    model = WheelerTextAutoencoder(
        vocab_size=32000,
        d_model=256,
        num_heads=8,
        encoder_layers=4,
        decoder_layers=6,
        grid_size=64,
        max_seq_len=512,
        dropout=0.1
    )
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    target_ids = input_ids.clone()  # For autoencoding, target = input
    
    print(f"\nInput shape: {input_ids.shape}")
    
    logits, grid = model(input_ids, target_ids)
    
    print(f"Grid shape: {grid.shape}")
    print(f"Logits shape: {logits.shape}")
    
    # Test grid to image conversion
    img = model.get_grid_as_image(grid)
    print(f"Image shape: {img.shape}")
    print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
    
    print("\n✓ Architecture test passed!")
