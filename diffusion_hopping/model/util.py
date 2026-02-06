import torch
import torch.nn as nn


def scatter_mean(src, index, dim=0, dim_size=None):
    """
    Replacement for torch_scatter.scatter_mean using native PyTorch operations.
    
    Args:
        src: Source tensor
        index: Index tensor indicating which group each element belongs to
        dim: Dimension along which to scatter (default: 0)
        dim_size: Size of the output dimension (if None, inferred from max index)
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Create output tensor with the right shape
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    
    # Expand index to match src shape
    index_shape = [1] * src.ndim
    index_shape[dim] = src.shape[dim]
    index_expanded = index.view(index_shape).expand_as(src)
    
    # Use scatter_reduce with mean reduction (available in PyTorch 1.12+)
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    out = out.scatter_reduce(dim, index_expanded, src, reduce='mean', include_self=False)
    
    return out


def centered_batch(x, batch, mask=None, dim_size=None):
    if mask is None:
        mean = scatter_mean(x, batch, dim=0)
    else:
        mean = scatter_mean(
            x[mask], batch[mask], dim=0, dim_size=dim_size
        )
    return x - mean[batch]


def skip_computation_on_oom(return_value=None, error_message=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if error_message is not None:
                        print(error_message)
                    return return_value
                else:
                    raise e

        return wrapper

    return decorator
