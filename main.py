import torch
from sdxl_pipeline import SDXLPipeline, SDPipeline
import argparse
import numpy as np
from PIL import Image
import random
from diffusers import EulerAncestralDiscreteScheduler
import os
import time
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

def recursive_cuda_graphable(root):
    from sfast.cuda.graphs import (
        make_dynamic_graphed_callable,
    )
    def stat_recursive(name, module):
        if hasattr(module, "set_processor"):
            print(f"{name}.processor", " ## ", module.forward)
        else: 
            make_dynamic_graphed_callable(module.forward)
        for subname, child in module.named_children():
            # stat_recursive(f"{name}.{subname}", child)
            pass
    for name, module in root.named_children():
        stat_recursive(name, module)

def sfast_compile(pipeline):
    from sfast.compilers.diffusion_pipeline_compiler import (compile,
                                                         CompilationConfig)
    # sfast
    config = CompilationConfig.Default()
    # xformers and Triton are suggested for achieving best performance.
    try:
        import xformers
        config.enable_xformers = False
    except ImportError:
        print('xformers not installed, skip')
    try:
        import triton
        config.enable_triton = True
    except ImportError:
        print('Triton not installed, skip')
    # CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
    # But it can increase the amount of GPU memory used.
    # For StableVideoDiffusionPipeline it is not needed.
    config.enable_jit = True
    config.enable_cuda_graph = True
    pipeline = compile(pipeline, config)
    return pipeline

def compile_unet(pipeline):
    from optimizer.compile import compile
    pipeline.unet = compile(pipeline.unet)

def refresh_unet(pipeline, model_path):
    from diffusers import UNet2DConditionModel
    # unet_model = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch.float16)

    # from unet.unet import UNet2DConditionModel
    # unet = UNet2DConditionModel(**unet_model.config).half().cuda()
    # from diffusers.models.modeling_utils import load_model_dict_into_meta
    # unexpected_keys = load_model_dict_into_meta(
    #     unet, unet_model.state_dict(), device="cuda", dtype=torch.float16
    # )

    from unet.unet import UNet2DConditionModel
    unet = UNet2DConditionModel(**pipeline.unet.config).half().cuda()
    from diffusers.models.modeling_utils import load_model_dict_into_meta
    unexpected_keys = load_model_dict_into_meta(
        unet, pipeline.unet.state_dict(), device="cuda", dtype=torch.float16
    )

    pipeline.unet = unet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/models/counterfeitXL')
    parser.add_argument('--s_prompt', type=str, default='')
    args = parser.parse_args()

    model_path = args.model_path
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder='scheduler')
    model = SDXLPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")

    b_prompt = ['Haunted Houses', 'Busy City Streets', 'Bars or Cafés', 'School']
    # actions = ['standing', 'sitting', 'running', 'jumping']
    actions = ['Punching through a Brick Wall', 'Running with a Torch', 'Jumping', 'Crying Holding a Crumpled Letter']
    neg_prompt = 'monochrome, lowres, bad anatomy, worst quality, low quality'

    height, width = 768, 1024
    num_inference_steps = 30
    output1, output2, output3 = [], [], []
    seed = random.randint(0, 100000)
    seed = 48570
    print('seed:', seed)

    refresh_unet(model, model_path)
    model = sfast_compile(model)
    # compile_unet(model)

    enable_cuda_graph = False
    if enable_cuda_graph:
        if getattr(model, 'unet', None) is not None:
            recursive_cuda_graphable(model.unet)
        if getattr(model, 'vae', None) is not None:
            recursive_cuda_graphable(model.vae)
        if getattr(model, 'text_encoder', None) is not None:
            recursive_cuda_graphable(model.text_encoder)
        if getattr(model, 'text_encoder_2', None) is not None:
            recursive_cuda_graphable(model.text_encoder_2)
        if getattr(model, 'image_encoder', None) is not None:
            recursive_cuda_graphable(model.image_encoder)

    s_prompt = '1boy,black hair,curly hair,brown eyes,casual blue t-shirt,cargo green shorts,playful red cap,friendly yellow smile, solo'

    for ii in range(4):
        tgt_prompt = s_prompt + ', ' + actions[ii] + ', ' + b_prompt[ii]
        begin = time.time()
        im = model(tgt_prompt, negative_prompt=neg_prompt, generator=torch.manual_seed(seed), width=width, height=height, num_inference_steps=num_inference_steps).images[0]
        print(f"# {ii} cost: ", time.time() - begin)
        output1.append(np.asarray(im))

    output1 = np.concatenate(output1, axis=1)
    output = np.concatenate([output1], axis=0)
    os.makedirs('result', exist_ok=True)
    ix = len(os.listdir('result')) + 1
    Image.fromarray(output).save(f'result/{ix}.jpg')

if __name__ == '__main__':
    main()
