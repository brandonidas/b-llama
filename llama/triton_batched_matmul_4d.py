import torch

import triton
import triton.language as tl


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
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
@triton.jit
def batched_4d_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        B, C, # batch and channel
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).

        # naive navigation of batch and channel. slight wastage on last group but that's fine.
        batch_stride_a, channel_stride_a, # from A.stride(0),A.stride(1)
        batch_stride_b, channel_stride_b,
        batch_stride_c, channel_stride_c,

        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,

        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    
    pid = tl.program_id(axis=0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    
    #a_inner_2d_size  = M * stride_am + K * stride_ak
    a_channel_offset = tl.program_id(axis=1) * channel_stride_a
    a_batch_offset   = tl.program_id(axis=2) * batch_stride_a 
    a_outer_dim_offset = a_channel_offset + a_batch_offset

    b_channel_offset = tl.program_id(axis=1) * channel_stride_b
    b_batch_offset   = tl.program_id(axis=2) * batch_stride_b
    b_outer_dim_offset = b_channel_offset + b_batch_offset

    c_channel_offset = tl.program_id(axis=1) * channel_stride_c
    c_batch_offset   = tl.program_id(axis=2) * batch_stride_c
    c_outer_dim_offset = c_channel_offset + c_batch_offset

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) 
    a_ptrs += a_outer_dim_offset
    #tl.device_print("a_ptr ", a_ptrs)
    #tl.device_print("a_outer_dim_offset ", a_outer_dim_offset)

    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b_ptrs += b_outer_dim_offset

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu": # TODO fuse other stuff instead, like softmax
        pass
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_ptrs += c_outer_dim_offset

    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def batched_4d_matmul(a, b, activation=""):
    # Check constraints.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    B, C, M, K = a.shape
    _, _ , K, N = b.shape
    # Allocates output.
    c = torch.empty((B,C,M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),N,B, )
    batched_4d_matmul_kernel[grid](
        a, b, c,  #
        B, C, M, N, K,  #
        
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #

        a.stride(2), a.stride(3),  #
        b.stride(2), b.stride(3),  #
        c.stride(2), c.stride(3),  #
        ACTIVATION=activation  #
    )
    return c

device = torch.device('cuda:0')

def random_matrix_test():
   # Testing the function
   B, C, M, K, N = 2, 2, 2, 2, 2
   A = torch.randn(B, C, M, K).to(device)
   B = torch.randn(B, C, K, N).to(device)

   # Using the custom function
   C_custom = batched_4d_matmul(A, B)

   # Using torch.bmm directly for comparison
   C_bmm = torch.matmul(A,B)

   # Check if results are the same
   print("shapes, right?", C_custom.shape, C_bmm.shape)
   print("row vs row:") 
   print(C_custom)
   print(C_bmm)
   print("Are the results identical? absolute tolerance = 1e-2", torch.allclose(C_custom, C_bmm,atol=1e-2))
random_matrix_test()
def identity_matrix_test():
  # Create an identity matrix for each 2x2 matrix in the 4D tensor
  identity_4d = torch.eye(2).repeat(2, 2, 1, 1).float().to(device)
  # Generate a 2 x 2 x 2 x 2 tensor with increasing numbers
  tensor_4d = torch.arange(2*2*2*2).view(2, 2, 2, 2).float().to(device)
  C_custom = batched_4d_matmul(tensor_4d,identity_4d)
  C_bmm = torch.matmul(tensor_4d, identity_4d)
  assert C_custom.is_contiguous()
  print(C_custom)
  print(C_bmm)

identity_matrix_test()

def big_matrix_test():
  torch.manual_seed(0)
  import time
  a = torch.randn((12,13,170,320), device='cuda', dtype=torch.float16)
  b = torch.randn((12,13,320,170), device='cuda', dtype=torch.float16)

  triton_start = time.time()
  triton_output = batched_4d_matmul(a, b)
  triton_end = time.time()
  triton_time = triton_end - triton_start
  
  torch_start = time.time()
  torch_output = torch.matmul(a, b)
  torch_end = time.time()
  torch_time = torch_end - torch_start

  print("triton time: ", triton_time)
  print("torch  time: ", torch_time)
  print("big matrix test: all close?", torch.allclose(triton_output, torch_output, atol=1e-2))
big_matrix_test()
