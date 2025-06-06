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

import torch

def _compare_element_wise(actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float, verbose: bool) -> tuple[bool, str]:
    """
    Performs element-wise comparison.
    Returns a tuple of (is_passed: bool, message: str).
    """
    if torch.allclose(actual, expected, rtol=rtol, atol=atol):
        message = "Verification: PASSED ✅"
        if verbose:
            message += f"\n(Tensors are element-wise close within rtol={rtol}, atol={atol})"
        return True, message

    # Failures are always verbose
    diff = torch.abs(actual - expected)
    max_diff = diff.max()
    max_diff_indices = (diff == max_diff).nonzero(as_tuple=False)
    first_max_diff_idx = max_diff_indices[0]
    
    actual_val = actual[tuple(first_max_diff_idx)]
    expected_val = expected[tuple(first_max_diff_idx)]

    message = (
        f"Verification: FAILED ❌\n"
        f"Mean diff: {diff.mean().item():.6f}, Max diff: {max_diff.item():.6f}\n"
        f"Max diff at {first_max_diff_idx.tolist()}: {actual_val.item():.6f} vs {expected_val.item():.6f}"
    )
    return False, message

def _compare_similarity(actual: torch.Tensor, expected: torch.Tensor, diff_tolerance: float, verbose: bool) -> tuple[bool, str]:
    """
    Performs similarity comparison.
    Returns a tuple of (is_passed: bool, message: str).
    """
    x, y = actual.double(), expected.double()
    numerator = 2 * (x * y).sum()
    denominator = (x.square() + y.square()).sum()

    diff = 0.0 if denominator == 0 else 1 - (numerator / denominator)
    
    if diff <= diff_tolerance:
        message = "Verification: PASSED ✅"
        if verbose:
            message += f"\n(Difference: {diff:.6f} <= Tolerance: {diff_tolerance})"
        return True, message
    
    # Failures are always verbose
    message = f"Verification: FAILED ❌\n(Difference: {diff:.6f} > Tolerance: {diff_tolerance})"
    return False, message


def compare_tensors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    mode: str = 'element-wise',
    rtol: float = 1e-5,
    atol: float = 1e-8,
    diff_tolerance: float = 1e-6,
    verbose: bool = True
) -> bool:
    """
    Compares two PyTorch tensors, prints the result, and returns a boolean status.

    Args:
        actual (torch.Tensor): The tensor produced by the model/operation.
        expected (torch.Tensor): The reference tensor.
        mode (str): 'element-wise' or 'similarity'.
        rtol (float): Relative tolerance for element-wise comparison.
        atol (float): Absolute tolerance for element-wise comparison.
        diff_tolerance (float): Maximum allowed difference for similarity comparison.
        verbose (bool): Controls output detail on success. Failures always
                        produce a detailed report. Defaults to True.

    Returns:
        bool: True if the tensors pass the comparison, False otherwise.
    """
    is_passed, message = False, ""

    if actual.shape != expected.shape:
        is_passed = False
        message = f"Verification: FAILED ❌\nShape mismatch: {actual.shape} vs {expected.shape}"
    elif mode == 'element-wise':
        is_passed, message = _compare_element_wise(actual, expected, rtol, atol, verbose)
    elif mode == 'similarity':
        is_passed, message = _compare_similarity(actual, expected, diff_tolerance, verbose)
    else:
        is_passed = False
        message = f"Verification: FAILED ❌\nInvalid mode '{mode}'. Choose 'element-wise' or 'similarity'."
    
    print(message)
    if(not is_passed) and verbose:
        print(f"Actual tensor:\n{actual.flatten()[:10]}\n")
        print(f"Expected tensor:\n{expected.flatten()[:10]}\n")
    return is_passed