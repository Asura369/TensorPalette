from typing import Optional

import torch
import torchvision.transforms as T
from PIL import Image

from styleforge.cin import CINTransformer
from styleforge.transformer import StyleTransformer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class InferenceEngine:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or pick_device()
        self._models: dict[str, torch.nn.Module] = {}

    def load_single_style(self, name: str, model_path: str) -> None:
        model = StyleTransformer()
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self._models[name] = model

    def load_cin(self, name: str, model_path: str, num_styles: int) -> None:
        model = CINTransformer(num_styles=num_styles)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self._models[name] = model

    def get_model(self, name: str) -> torch.nn.Module:
        if name not in self._models:
            raise KeyError(f"Model '{name}' not loaded. Available: {list(self._models.keys())}")
        return self._models[name]

    @torch.no_grad()
    def stylize(
        self,
        image: Image.Image,
        model_name: str,
        style_id: int = 0,
        style_a: Optional[int] = None,
        style_b: Optional[int] = None,
        alpha: float = 1.0,
    ) -> Image.Image:
        model = self.get_model(model_name)
        content_transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: x.mul(255))
        ])
        content = content_transform(image).unsqueeze(0).to(self.device)

        if isinstance(model, CINTransformer):
            if style_a is not None and style_b is not None:
                return self._stylize_cin_interp(model, content, style_a, style_b, alpha)
            style_ids = torch.tensor([style_id], device=self.device)
            output = model(content, style_ids)
        else:
            output = model(content)

        return self._tensor_to_image(output)

    def _stylize_cin_interp(
        self,
        model: CINTransformer,
        content: torch.Tensor,
        style_a: int,
        style_b: int,
        alpha: float,
    ) -> Image.Image:
        output = model.forward_interpolated(content, style_a, style_b, alpha)
        return self._tensor_to_image(output)

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        img = tensor[0].cpu().clone().clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        return Image.fromarray(img)
