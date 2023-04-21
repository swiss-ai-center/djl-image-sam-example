from typing import Tuple
import mock
import torch
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom


def forward(self, size: Tuple[int, int]) -> torch.Tensor:
    """Generate positional encoding for a grid of the specified size."""
    h, w = size
    device = self.positional_encoding_gaussian_matrix.device
    grid = torch.ones((h, w), device=device, dtype=torch.float32)
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5
    y_embed = y_embed / h
    x_embed = x_embed / w

    pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
    return pe.permute(2, 0, 1)  # C x H x W


patches = (mock.patch.object(PositionEmbeddingRandom, "forward", forward),)
