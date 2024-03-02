import torch
from torch.fx import symbolic_trace, subgraph_rewriter
import functools
import torch._dynamo as torchdynamo

def remove_dropout(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for n in gm.graph.nodes:
        is_dropout_module = n.op == "call_module" and isinstance(modules[n.target], torch.nn.Dropout)
        is_dropout_function = n.target == torch.nn.functional.dropout
        # If the target matches one of the patterns
        if is_dropout_module or is_dropout_function:
            # Set the insert point, add the new node, and replace all uses
            # of `n` with the new node
            with gm.graph.inserting_after(n):
                # new_node = gm.graph.call_function(torch.nn.Identity, n.args, n.kwargs)
                n.replace_all_uses_with(n.args[0])
            # Remove the old node from the graph
            gm.graph.erase_node(n)
    gm.recompile()

def optimize_backend(gm: torch.fx.GraphModule):
    from kernl.optimizer.normalizer import normalize_operators
    from stabletriton.optimizers import remove_dropout, fuse_attention, fuse_geglu, \
                                replace_group_norm,replace_layer_norm, \
                                replace_linear, fuse_timesteps, \
                                replace_linear_activ, replace_group_norm_activation, \
                                make_dynamic_graphed_callable, cuda_graphs_wrapper, static_inputs_pool
    normalize_operators(gm)
    remove_dropout(gm)
    fuse_attention(gm)
    fuse_geglu(gm)
    replace_linear_activ(gm, torch.nn.SiLU())

    from kernl.optimizer.group_norm import replace_group_norm, replace_group_norm_activation
    replace_group_norm_activation(gm, torch.nn.SiLU())
    replace_group_norm(gm)
    replace_layer_norm(gm)
    # Replacing Linear makes it slower??
    # Drops from 8.38 to 6.61 (Still faster than vanilla pytorch tho :) )
    # replace_linear(gm)
    fuse_timesteps(gm)

    return gm

def compile(m: torch.nn.Module):
    # gm: torch.fx.GraphModule = torch.fx.symbolic_trace(m)
    from .fx_tracer import symbolic_trace
    gm: torch.fx.GraphModule = symbolic_trace(m, leaf_modules=["Attention", "BasicTransformerBlock"])
    optimize_backend(gm)
    gm.recompile()

    gm.config = m.config
    gm.class_for_deserialization = m.__class__
    gm.device = m.device
    use_cuda_graph = False
    if use_cuda_graph:
        from stabletriton.optimizers import make_dynamic_graphed_callable
        gm.forward = make_dynamic_graphed_callable(gm.forward)
    return gm