
import torch

from styleforge.cin import CINTransformer
from styleforge.transformer import StyleTransformer


def export_single_style(model_path: str, output_path: str, image_size: int = 256) -> str:
    model = StyleTransformer()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    return output_path


def export_cin_style(
    model_path: str,
    output_path: str,
    num_styles: int,
    style_id: int = 0,
    image_size: int = 256,
) -> str:
    model = CINTransformer(num_styles=num_styles)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    class _Wrapped(torch.nn.Module):
        def __init__(self, base_model, sid):
            super().__init__()
            self.base = base_model
            self.sid = sid

        def forward(self, x):
            style_ids = torch.tensor([self.sid], device=x.device)
            return self.base(x, style_ids)

    wrapped = _Wrapped(model, style_id)
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        wrapped,
        (dummy,),
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    return output_path
