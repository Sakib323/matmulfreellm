# File: dit_hgrn_bit/modeling_dit.py
import torch
import torch.nn as nn
from einops import rearrange
from diffusers.models import AutoencoderKL
from transformers import PretrainedConfig, PreTrainedModel

# Import your custom components
from hgrn_bit import HGRNBitConfig, HGRNBitBlock

class DiTHGRNConfig(PretrainedConfig):
    def __init__(
        self,
        in_channels=3,
        patch_size=(2, 16, 16),
        hidden_size=1024,
        depth=12,
        num_heads=8,
        hgrn_config_kwargs=None,
        **base_kwargs
    ):
        super().__init__(**base_kwargs)
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.hgrn_config = HGRNBitConfig(
            hidden_size=hidden_size,
            num_hidden_layers=depth,
            num_heads=num_heads,
            rotary_embeddings=False,
            moe=False,
            **(hgrn_config_kwargs or {})
        )

class VideoPatchEmbedding(nn.Module):
    """From DiT official implementation with 3D adaptation"""
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, C, T, H, W) -> (B, D, T', H', W')
        return rearrange(x, "b c t h w -> b (t h w) c")

class TimestepEmbedder(nn.Module):
    """From DiT official implementation"""
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, timesteps):
        return self.mlp(timesteps)

class DiTHGRNModel(PreTrainedModel):
    config_class = DiTHGRNConfig

    def __init__(self, config):
        super().__init__(config)
        self.patch_embed = VideoPatchEmbedding(config)
        self.timestep_embedder = TimestepEmbedder(config.hidden_size)
        
        # Initialize HGRN-Bit blocks
        self.blocks = nn.ModuleList([
            HGRNBitBlock(config.hgrn_config, layer_idx=i) 
            for i in range(config.depth)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.proj_out = nn.Linear(
            config.hidden_size, 
            config.in_channels * np.prod(config.patch_size)
        )

    def forward(self, x, timesteps):
        # Input processing
        x = self.patch_embed(x)
        temb = self.timestep_embedder(timesteps)
        
        # Add time embedding
        x = x + temb.unsqueeze(1)
        
        # Process through HGRN-Bit blocks
        for block in self.blocks:
            x, _, _ = block(x)
        
        # Final projection
        x = self.norm(x)
        x = self.proj_out(x)
        return x
