from torch import nn
import torch

def get_downsampler(downsample_method: str, downsample_factor: int, audio_dim: int, device: torch.device, dtype: torch.dtype):
    downsampler = nn.Module()
    downsampled_dim = 0

    if downsample_method == 'avg_pool':
        downsampler = (AvgPoolAdapter(downsample_factor).to(device, dtype=dtype))
        downsampled_dim = audio_dim
    elif downsample_method == 'reshape':
        downsampler = (ReshapeAdapter(downsample_factor, dtype).to(device, dtype=dtype))
        downsampled_dim = audio_dim * downsample_factor
    elif downsample_method == 'conv1d':
        downsampler = (Conv1DAdapter(audio_dim, downsample_factor).to(device, dtype=dtype))
        downsampled_dim = audio_dim
    elif downsample_method == 'linear':
        downsampler = (LinearAdapter(audio_dim, downsample_factor, dtype).to(device, dtype=dtype))
        downsampled_dim = audio_dim
    elif downsample_method == 'cif':
        downsampler = (CIFireAdapter(audio_dim, dtype).to(device, dtype=dtype))
        downsampled_dim = audio_dim

    return downsampler, downsampled_dim

class Conv1DAdapter(nn.Module):
    def __init__(self, audio_dim, downsample_K):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=audio_dim,
            out_channels=audio_dim,
            kernel_size=downsample_K,
            stride=downsample_K
        )
        self.ln = nn.LayerNorm(audio_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.ln(x)

        return x

class AvgPoolAdapter(nn.Module):
    def __init__(self, downsample_K):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=downsample_K, stride=downsample_K)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)

        return x

class ReshapeAdapter(nn.Module):
    def __init__(self, downsample_K, dtype):
        super().__init__()
        self.downsample_K = downsample_K
        self.dtype = dtype

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        if seq_len % self.downsample_K != 0:
            raise ValueError("Sequence length must be divisible by the downsample factor")

        x = x.reshape(batch_size, seq_len // self.downsample_K, hidden_size * self.downsample_K)
        x.contiguous()
        x = x.to(dtype=self.dtype)

        return x

class LinearAdapter(nn.Module):
    def __init__(self, audio_dim, downsample_K, dtype):
        super().__init__()
        self.downsample_K = downsample_K
        self.proj = nn.Linear(audio_dim * downsample_K, audio_dim)
        self.ln = nn.LayerNorm(audio_dim)
        self.dtype = dtype

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        if seq_len % self.downsample_K != 0:
            raise ValueError("Sequence length must be divisible by the downsample factor")

        x = x.reshape(batch_size, seq_len // self.downsample_K, hidden_size * self.downsample_K)
        x = self.proj(x)
        x = self.ln(x)

        return x

class CIFireAdapter(nn.Module):
    def __init__(self, audio_dim, dtype):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=audio_dim,
            out_channels=audio_dim,
            kernel_size=3,
            padding=1,
            groups=audio_dim
        )

        self.weight_proj = nn.Linear(audio_dim, 1)

    def forward(self, x, mask):
        batch_size, seq_len, hidden_size = x.shape

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        weights = self.weight_proj(x)
        alphas = torch.sigmoid(weights).squeeze(-1)

        if mask is not None:
            alphas = alphas * mask

        cumsum_alphas = torch.cumsum(alphas, dim=1)
        token_indices = torch.floor(cumsum_alphas).long()
        max_tokens = token_indices.max().item() + 1
        weighted_states = x * alphas.unsqueeze(-1)

        scatter_indices = token_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        cif_outputs = torch.zeros(batch_size, max_tokens, hidden_size, device=x.device, dtype=x.dtype)
        cif_outputs.scatter_add_(dim=1, index=scatter_indices, src=weighted_states)

        alpha_sums = torch.zeros(batch_size, max_tokens, device=x.device, dtype=x.dtype)
        alpha_sums.scatter_add_(dim=1, index=token_indices, src=alphas)

        cif_outputs = cif_outputs / (alpha_sums.unsqueeze(-1) + 1e-8)
        out_masks = (alpha_sums > 0).to(x.dtype)

        return cif_outputs, out_masks, alphas






