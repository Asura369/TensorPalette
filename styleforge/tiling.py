import math
from typing import Callable

import torch
import torch.nn.functional as F

TILE_SIZE = 512
OVERLAP = 64
TILE_THRESHOLD = 1280


def _linear_blend_weight(h, w, device):
    y = torch.linspace(0, 1, h, device=device)
    x = torch.linspace(0, 1, w, device=device)
    wy = torch.minimum(y, 1 - y) * 2
    wx = torch.minimum(x, 1 - x) * 2
    wy = wy.clamp(min=0.01)
    wx = wx.clamp(min=0.01)
    return wy.unsqueeze(1) * wx.unsqueeze(0)


def tiled_inference(
    image: torch.Tensor,
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    tile_size: int = TILE_SIZE,
    overlap: int = OVERLAP,
) -> torch.Tensor:
    b, c, h, w = image.shape
    device = image.device

    if max(h, w) <= TILE_THRESHOLD:
        return model_fn(image)

    stride = tile_size - overlap
    output = torch.zeros_like(image)
    weight = torch.zeros(1, 1, h, w, device=device)

    tile_weight = _linear_blend_weight(tile_size, tile_size, device).unsqueeze(0).unsqueeze(0)

    n_rows = max(1, math.ceil((h - overlap) / stride))
    n_cols = max(1, math.ceil((w - overlap) / stride))

    for r in range(n_rows):
        for col in range(n_cols):
            y0 = min(r * stride, max(0, h - tile_size))
            x0 = min(col * stride, max(0, w - tile_size))
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)

            if y1 - y0 < tile_size:
                y0 = max(0, y1 - tile_size)
            if x1 - x0 < tile_size:
                x0 = max(0, x1 - tile_size)

            tile = image[:, :, y0:y1, x0:x1]

            if tile.shape[2] < tile_size or tile.shape[3] < tile_size:
                pad_h = tile_size - tile.shape[2]
                pad_w = tile_size - tile.shape[3]
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")

            result_tile = model_fn(tile)

            actual_h = y1 - y0
            actual_w = x1 - x0
            result_tile = result_tile[:, :, :actual_h, :actual_w]
            tw = tile_weight[:, :, :actual_h, :actual_w]

            output[:, :, y0:y1, x0:x1] += result_tile * tw
            weight[:, :, y0:y1, x0:x1] += tw

    weight = weight.clamp(min=1e-8)
    return output / weight
