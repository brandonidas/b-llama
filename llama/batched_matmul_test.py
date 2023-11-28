import torch
import random

def test_batched_matmul(batched_matmul_fn, atol=1e-5, rtol=1e-3):
    """
    Test the custom batched matrix multiplication function.

    Args:
    - batched_matmul_fn: The custom batched matrix multiplication function.
    - atol: Absolute tolerance for comparison.
    - rtol: Relative tolerance for comparison.
    """
    # Test for different batch sizes and matrix dimensions
    for _ in range(10):  # Number of tests
        B1 = random.randint(1, 5)  # Random batch size 1
        B2 = random.randint(1, 5)  # Random batch size 2
        M = random.randint(16, 64)  # Random matrix dimension M
        N = random.randint(16, 64)  # Random matrix dimension N
        K = random.randint(16, 64)  # Random matrix dimension K

        # Generate random input tensors
        a = torch.randn(B1, B2, M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(B1, B2, K, N, device='cuda', dtype=torch.float16)

        # Compute output using custom batched matmul
        c_custom = batched_matmul_fn(a, b)

        # Compute expected output using PyTorch
        c_expected = torch.matmul(a.float(), b.float())

        # Check if outputs are close
        assert torch.allclose(c_custom, c_expected, atol=atol, rtol=rtol), f"Test failed for batch sizes ({B1}, {B2}), and dimensions ({M}, {N}, {K})"

    print("All tests passed successfully.")

# Example usage
test_batched_matmul(batched_matmul)  # Assuming `batched_matmul` is your custom function
