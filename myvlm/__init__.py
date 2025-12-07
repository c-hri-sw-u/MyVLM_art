import pyrallis
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    class _TorchStub:
        class dtype:
            pass
        float16 = 'float16'
        bfloat16 = 'bfloat16'
        float32 = 'float32'
    torch = _TorchStub()

def decode_dtype(dtype: str):
    if dtype == 'float16':
        return torch.float16
    elif dtype == 'bfloat16':
        return torch.bfloat16
    else:
        return torch.float32

if _HAS_TORCH:
    pyrallis.decode.register(torch.dtype, lambda x: decode_dtype(x))
    pyrallis.encode.register(torch.dtype, lambda x: x.__str__())
