import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from typing import Dict, Any, List, Union, Tuple, Optional
from collections import defaultdict

# import oneflow.nn.functional as F
# from flash_attn import flash_attn_func
@torch.jit.script
class Counter:
    _self = None
    def __new__(cls):
        if cls._self is None:
            cls._self = super().__new__(cls)
        return cls._self

    def __init__(self, num_att_layers:int = -1) -> None:
        self.cur_step:int = 0
        self.cur_att_layer:int = 0
        self.log_mode:bool = True
        self.num_att_layers:int = num_att_layers

    def incr_step(self):
        self.cur_step += 1
        # print("incr_step: ", self.cur_step)

    def incr_att_layer(self):
        self.cur_att_layer += 1
    
    def set_step(self, num:int):
        self.cur_step = num

    def set_att_layer(self, num:int):
        self.cur_att_layer = num

    def set_num_att_layers(self, num:int):
        self.num_att_layers = num

    def reset(self, log_mode:bool =False):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.log_mode = log_mode

@torch.jit.script
class AttentionBase:
    def __init__(self):
        pass

    # @torch.jit.ignore
    def __call__(self, q, k, v, is_cross:bool, num_heads:int):
        out = self.forward(q, k, v, is_cross, num_heads)
        return out

    # @torch.jit.ignore
    def forward(self, q, k, v, is_cross:bool, num_heads:int):
        # print("attn::forward", self.cur_step, self.cur_att_layer)
        # print(q.shape, k.shape, v.shape)   # [_, seq_len, n_dim]
        n_pixel, n_dim = q.shape[1], q.shape[-1]
        q = q.reshape(-1, num_heads, q.shape[1], q.shape[-1])
        k = k.reshape(-1, num_heads, k.shape[1], k.shape[-1])
        v = v.reshape(-1, num_heads, v.shape[1], v.shape[-1])
        # q, k, v = map(lambda t: rearrange(t, '(b n) h d -> b n h d', n = num_heads), (q, k, v))

        # print(q.shape, k.shape, v.shape)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(-1, n_pixel, num_heads * n_dim)
        # out = rearrange(out, 'b h n d -> b n (h d)')

        return out

@torch.jit.script
class SimpleAttn:
    def __init__(self):
        self.attn_base = AttentionBase()
        self.counter = Counter()

    @torch.jit.ignore
    def forward(self, q, k, v, is_cross: bool, num_heads: int):
        print("SampleAttn::forward")
        return self.attn_base.forward(q, k, v, is_cross, num_heads)

@torch.jit.script
class LocalMutualSelfAttentionControl_4x:
    # @torch.jit.ignore
    def __init__(self, width:int, height:int, start_step:int = 4, start_layer:int = 10, layer_idx: Optional[List[int]] = None, step_idx: Optional[List[int]] = None, total_steps:int=50, place_in_unet:str='up'):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
        """
        self.attn_base = AttentionBase()
        self.counter = Counter()
        self.place_in_unet: str = place_in_unet
        self.k_store: dict[int, dict[int, list[torch.Tensor]]] = dict()
        self.v_store: dict[int, dict[int, list[torch.Tensor]]] = dict()
        self.cache: dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.width: int = width
        self.height: int = height
        self.total_steps: int = total_steps
        self.start_step: int = start_step
        self.start_layer: int = start_layer
        # self.layer_idx: Optional[List[int]] = None
        self.layer_idx: Optional[List[int]] = layer_idx if layer_idx is not None else list(range(start_layer, 16))
        self.step_idx: List[int] = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("step_idx: ", self.step_idx)
        print("layer_idx: ", self.layer_idx)

    def reset(self):
        self.counter.reset()

    # @torch.jit.ignore
    # def __call__(self, q, k, v, is_cross: bool, num_heads: int):
    #     return self.forward(q, k, v, is_cross, num_heads)

    def forward(self, q, k, v, is_cross: bool, num_heads: int):
        return self.forward_impl(q, k, v, is_cross, num_heads)

    @torch.jit.ignore
    def forward_impl(self, q, k, v, is_cross: bool, num_heads: int):
        """
        Attention forward function
        """
        self.counter.incr_att_layer()
        if self.counter.cur_att_layer == self.counter.num_att_layers:
            self.counter.set_att_layer(0)
            self.counter.incr_step()

        # print("curr_step: ", self.counter.cur_step, ", cur_att_layer: ", self.counter.cur_att_layer, ", num_att_layers: ", self.counter.num_att_layers, ", log_mode: ", self.counter.log_mode)
        if self.counter.log_mode:
            if is_cross or self.counter.cur_step not in self.step_idx or self.counter.cur_att_layer // 2 not in self.layer_idx:
                return self.attn_base.forward(q, k, v, is_cross, num_heads)
            
            if self.counter.cur_step not in self.k_store:
                self.k_store[self.counter.cur_step] = {-1:list(torch.tensor([1]))}
                self.v_store[self.counter.cur_step] = {-1:list(torch.tensor([0]))}
            
            self.k_store[self.counter.cur_step][self.counter.cur_att_layer] = k.chunk(2)
            self.v_store[self.counter.cur_step][self.counter.cur_att_layer] = v.chunk(2)
            return self.attn_base.forward(q, k, v, is_cross, num_heads)
            

        if is_cross or self.counter.cur_step not in self.step_idx or self.counter.cur_att_layer // 2 not in self.layer_idx:
            return self.attn_base.forward(q, k, v, is_cross, num_heads)

        # import time
        # t1 = time.time()
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)

        n_pixel = kc.shape[1]
        ratio = self.width / self.height
        n_height = int((n_pixel / ratio) ** 0.5)
        n_width = n_pixel // n_height
        # print("ddd ", n_pixel, n_height, n_width, len(self.k_store), len(self.v_store), self.cur_step, self.cur_att_layer)
        
        assert n_height * n_width == n_pixel, 'please choose a suitable resolution'

        start, end = int(n_width // 6), int(n_width // 6 * 5)
        assert start * 6 == n_width, 'please choose a suitable resolution'

        n_dim = qu.shape[-1]

        # s0 = time.time()

        # use cache ?
        log_ku, log_kc = self.k_store[self.counter.cur_step][self.counter.cur_att_layer]
        log_vu, log_vc = self.v_store[self.counter.cur_step][self.counter.cur_att_layer]
        # s1 = time.time()
        # key = ("#").join(map(str, [self.counter.cur_step, self.counter.cur_att_layer, n_height, n_width]))
        key = f"{self.counter.cur_step}#{self.counter.cur_att_layer}#{n_height}#{n_width}"
        if key not in self.cache:
            print(log_ku.shape, log_kc.shape, n_height, n_width)
            log_ku, log_kc = log_ku.reshape(log_ku.shape[0], n_height, n_width, log_ku.shape[-1]), log_kc.reshape(log_kc.shape[0], n_height, n_width, log_kc.shape[-1])
            log_vu, log_vc = log_vu.reshape(log_vu.shape[0], n_height, n_width, log_vu.shape[-1]), log_vc.reshape(log_vc.shape[0], n_height, n_width, log_vc.shape[-1])
            self.cache[key] = (log_ku, log_kc, log_vu, log_vc)
        else:
            log_ku, log_kc, log_vu, log_vc = self.cache[key]
            #print("use cache cost: ", key, time.time() - s2)
        #s2 = time.time()
        
        ku, kc = ku.reshape(ku.shape[0], n_height, n_width, ku.shape[-1]), kc.reshape(kc.shape[0], n_height, n_width, kc.shape[-1])
        vu, vc = vu.reshape(vu.shape[0], n_height, n_width, vu.shape[-1]), vc.reshape(vc.shape[0], n_height, n_width, vc.shape[-1])

        ku[:, :, start:end, :] = log_ku[:, :, start:end, :]
        vu[:, :, start:end, :] = log_vu[:, :, start:end, :]
        kc[:, :, start:end, :] = log_kc[:, :, start:end, :]
        vc[:, :, start:end, :] = log_vc[:, :, start:end, :]

        qu, qc = qu.reshape(-1, num_heads, n_height * n_width, n_dim), qc.reshape(-1, num_heads, n_height * n_width, n_dim)
        ku, kc = ku.reshape(-1, num_heads, n_height * n_width, n_dim), kc.reshape(-1, num_heads, n_height * n_width, n_dim)
        vu, vc = vu.reshape(-1, num_heads, n_height * n_width, n_dim), vc.reshape(-1, num_heads, n_height * n_width, n_dim)
        # s6 = time.time()
        # print("cost: ", s6 - s0)
        # print("stat: ", s2-s1, s3-s2, s4-s3, s5-s4, s6-s5)

        # t2 = time.time()
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        out_u = F.scaled_dot_product_attention(qu, ku, vu, attn_mask=None, dropout_p=0.0, is_causal=False)
        out_c = F.scaled_dot_product_attention(qc, kc, vc, attn_mask=None, dropout_p=0.0, is_causal=False)
        # t3 = time.time()
        out_u = out_u.transpose(1, 2).reshape(out_u.shape[0], -1, num_heads * n_dim)
        out_c = out_c.transpose(1, 2).reshape(out_c.shape[0], -1, num_heads * n_dim)
        # t4 = time.time()
        # print("## cost: ", t2 - t1, t3 - t2, t4 - t3)
        # out_u = flash_attn_func(torch.permute(qu, [0, 2, 1, 3]), torch.permute(ku, [0, 2, 1, 3]), torch.permute(vu, [0, 2, 1, 3]), dropout_p=0.0, return_attn_probs=False, causal=False)
        # out_c = flash_attn_func(torch.permute(qc, [0, 2, 1, 3]), torch.permute(kc, [0, 2, 1, 3]), torch.permute(vc, [0, 2, 1, 3]), dropout_p=0.0, return_attn_probs=False, causal=False)
        # out_u = out_u.reshape(out_u.shape[0], -1, num_heads * n_dim)
        # out_c = out_c.reshape(out_c.shape[0], -1, num_heads * n_dim)


        out = torch.cat([out_u, out_c], dim=0)

        return out

    def get_store(self):
        return self.k_store, self.v_store

class SimpleAttnProcessor(nn.Module):
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        # self.log_mode = log_mode

    # @torch.jit.ignore
    # def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
    # @torch.jit.ignore
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
        """
        The attention is similar to the original implementation of LDM CrossAttention class
        except adding some modifications on the attention
        """

        if encoder_hidden_states is not None:
            context = encoder_hidden_states
        if attention_mask is not None:
            mask = attention_mask
        # print("SampleAttn")

        to_out = attn.to_out
        if isinstance(to_out, nn.modules.container.ModuleList):
            to_out = attn.to_out[0]
        else:
            to_out = attn.to_out

        h = attn.heads
        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        q = attn.to_q(hidden_states)
        is_cross = context is not None
        context = context if is_cross else hidden_states
        
        k = attn.to_k(context)
        v = attn.to_v(context)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # inner_dim = k.shape[-1]
        # head_dim = inner_dim // attn.heads

        # q = q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2).view(-1, )
        # k = k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # v = v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the only difference
        out = self.attnstore.forward(
            q, k, v, is_cross, attn.heads)
        
        # print("attn::forward")
        return to_out(out)


def register_attention_control(pipeline, attention_store):
    attn_procs = {}
    cross_att_count = 0
    # for name, module in pipeline.unet.named_children():
    #     print(name)
    #     if hasattr(module, "set_processor"):
    #         print("attn type: ", name)
    for name in pipeline.unet.attn_processors.keys():
        print("editor: ", name)
        if "mid_block" in name or "up_blocks" in name or "down_blocks" in name:
            place_in_unet = "attn"
        else:
            continue
        # if name.startswith("mid_block"):
        #     place_in_unet = "mid"
        # elif name.startswith("up_blocks"):
        #     place_in_unet = "up"
        # elif name.startswith("down_blocks"):
        #     place_in_unet = "down"
        # else:
        #     continue

        cross_att_count += 1
        attn_procs[name] = SimpleAttnProcessor(attnstore=attention_store, place_in_unet=place_in_unet)

    print("attn num: ", len(attn_procs))
    pipeline.unet.set_attn_processor(attn_procs)
    attention_store.counter.set_num_att_layers(cross_att_count)
    # pipeline.attention_store.num_att_layers = cross_att_count

def stat(root):
    def stat_recursive(name, module):
        if hasattr(module, "set_processor"):
            print(f"{name}.processor", " ## ", module.forward)
        for subname, child in module.named_children():
            stat_recursive(f"{name}.{subname}", child)
    for name, module in root.named_children():
        stat_recursive(name, module)
