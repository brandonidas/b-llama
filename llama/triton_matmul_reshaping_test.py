import torch
import triton_matmul
def custom_triton_function(tensor1, tensor2, triton_func):
    # Original shapes
    original_shape1 = tensor1.shape  # torch.Size([6, 32, 18, 128])
    original_shape2 = tensor2.shape  # torch.Size([6, 32, 128, 18])

    # Reshape the first tensor to 2D (flatten the last two dimensions)
    tensor1_2d = tensor1.reshape(-1, tensor1.shape[-2] * tensor1.shape[-1])  # torch.Size([6 * 32, 18 * 128])

    # Reshape the second tensor to 2D (flatten the first two dimensions, swap last two)
    tensor2_reshaped = tensor2.transpose(2, 3)  # Swap last two dimensions
    tensor2_2d = tensor2_reshaped.reshape(-1, tensor2_reshaped.shape[-2] * tensor2_reshaped.shape[-1])  # torch.Size([6 * 32 * 128, 18])

    # Call the Triton function with the reshaped tensors
    output_2d = triton_func(tensor1_2d, tensor2_2d.T)

    # Reshape output back to original shape
    # Modify this based on what the expected output shape should be
    output = output_2d.reshape(original_shape1)

    return output

# Example usage:
tensor1 = torch.randn(6, 32, 18, 128,device='cuda', dtype=torch.float16)
tensor2 = torch.randn(6, 32, 128, 18,device='cuda', dtype=torch.float16)

# Replace 'triton_softmax_function' with your actual Triton function
output = custom_triton_function(tensor1, tensor2, triton_matmul.matmul)
print(str(torch.matmul(tensor1, tensor2)))
print(str(output))
