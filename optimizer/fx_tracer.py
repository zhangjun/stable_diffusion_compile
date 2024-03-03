import torch
from typing import Any, Callable, Dict, List, Optional, Union

class MyTracer(torch.fx.Tracer):
    def __init__(self, leaf_modules: Optional[List[str]] = None) -> None:
        super().__init__()
        self._leaf_modules: List[str] = leaf_modules if leaf_modules is not None else []

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        print(type(m).__name__)
        if type(m).__name__ in self._leaf_modules:
            print(f"## is_leaf_module: {m}")
            return True
        return super().is_leaf_module(m, module_qualified_name)

def symbolic_trace(
    # pyre-ignore[24]
    root: Union[torch.nn.Module, Callable],
    concrete_args: Optional[Dict[str, Any]] = None,
    leaf_modules: Optional[List[str]] = None,
) -> torch.fx.GraphModule:
    tracer = MyTracer(leaf_modules)
    graph = tracer.trace(root, concrete_args)
    return torch.fx.GraphModule(root, graph)


if __name__ == "__main__":
    model_path = "/models/counterfeitXL"

    from diffusers import UNet2DConditionModel
    unet_model = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch.float16)
    from unet.unet import UNet2DConditionModel
    model = UNet2DConditionModel(**unet_model.config).half().cuda()
    graph = symbolic_trace(model, leaf_modules=["Attention", "BasicTransformerBlock"])
    module = torch.fx.GraphModule(model, graph)
    print(module.code)

    # from transformers.utils.fx import symbolic_trace
    # https://github.com/pytorch/examples/blob/main/fx/inline_function.py
    # https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/optimization.py
    # from torch.fx.experimental.optimization import fuse
