import torch
import triton.language as tl
import triton

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
def batched_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        batch_size, second_dim,M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_ba0, stride_ba1, stride_am, stride_ak,  # Strides for A
        stride_bb0, stride_bb1, stride_bk, stride_bn,  # Strides for B
        stride_bc0, stride_bc1, stride_cm, stride_cn,  # Strides for C
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    # Program IDs for batch and second dimension
    bid = tl.program_id(0)  # Batch index
    sid = tl.program_id(1)  # Second dimension index
    pid = tl.program_id(2)  # Program ID within the batch and second dimension

    # Calculate program IDs for M and N dimensions
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Initialize pointers for A, B, C for the current batch and second dimension
    a_ptr_batch = a_ptr + bid * stride_ba0 + sid * stride_ba1
    b_ptr_batch = b_ptr + bid * stride_bb0 + sid * stride_bb1
    c_ptr_batch = c_ptr + bid * stride_bc0 + sid * stride_bc1


    # TODO actual computation

    # Store result back into the slice of output tensor C
    c_slice_ptr = c_ptr + batch_idx * stride_c[0] + second_dim_idx * stride_c[1]
    c_ptrs = c_slice_ptr + offs_am[:, None] * stride_c[2] + offs_bn[None, :] * stride_c[3]
    c_mask = (offs_am[:, None] < dim1) & (offs_bn[None, :] < dim4)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

def batched_matmul(a, b, activation=""):
    # Check constraints
    assert a.shape[2] == b.shape[2], "Incompatible dimensions for matrix multiplication"
    assert a.is_contiguous(), "Tensor A must be contiguous"
    assert b.is_contiguous(), "Tensor B must be contiguous"
    
    # Extract dimensions
    B1, B2, M, K = a.shape
    _, _, K, N = b.shape
    
    # Allocate output tensor
    c = torch.empty((B1, B2, M, N), device=a.device, dtype=a.dtype)
    
    # Define the grid for 3D matrix multiplication with one thread per block
    grid = lambda META: (B1 * B2, triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    # Launch the Triton kernel
    matmul_kernel[grid](
        a, b, c, 
        B1, B2, M, N, K, 
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        c.stride(0), c.stride(1), c.stride(2), c.stride(3),
        ACTIVATION=activation
    )
    return c
 
