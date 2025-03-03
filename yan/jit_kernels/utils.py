import torch

_num_sms = None


def set_num_sms(num_sms: int) -> None:
    """
    Set the maximum SM count for all GEMM kernels to use.

    Arguments:
        num_sms: the desired maximum SM count for all GEMM kernels to use.
    """
    global _num_sms
    assert 0 < num_sms <= torch.cuda.get_device_properties(device='cuda').multi_processor_count
    _num_sms = num_sms


def get_num_sms() -> int:
    """
    Get the current maximum limit of SM count for all GEMM kernels to use.
    If the count is never specified, the function will return the number of device SMs.

    Returns:
        Current maximum limit of SM count for all GEMM kernels to use.
    """
    global _num_sms
    if _num_sms is None:
        _num_sms = torch.cuda.get_device_properties(device='cuda').multi_processor_count
    return _num_sms


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def get_m_alignment_for_contiguous_layout():
    """
    When we do a grouped GEMM in contiguous format, LHS are grouped into several batches along the M axis.
    Since we deal with exactly one sub-matrix of RHS for each GEMM block, batch sizes above should align well
        with GEMM block shape.
    
    Returns:
        Group-level alignment requirement for grouped contiguous layout, which is always 128.
    """
    return 128


def get_col_major_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Returns column-major format of the input tensor. `torch.transpose` will be called if necessary.
    If the input tensor is already in column-major layout, this function will do nothing.

    Arguments:
        x: input tensor to be converted to column-major format.

    Returns:
        The tensor in column-major format.
    """
    assert x.dim() in (2, 3)
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    
    b, m, n = x.shape

    # Check if already in column-major format
    if x.stride(1) == 1 and x.stride(2) == m:
        return x.squeeze(0) if remove_dim else x

    # Convert to column-major format
    # col_major_x = torch.empty((b, m, n), device=x.device, dtype=x.dtype).transpose(1, 2).contiguous().transpose(1, 2)
    # col_major_x[:] = x
    col_major_x = x.transpose(1, 2).contiguous().transpose(1, 2)

    return col_major_x.squeeze(0) if remove_dim else col_major_x