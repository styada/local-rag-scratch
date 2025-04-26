import mlx.core
import mlx.utils
import torch
import platform
import mlx

# Check if MPS is available
device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
print(f"Using device: {device}")

# Dummy example of loading an Ollama model using MLX
# Replace this with actual model loading code
class OllamaModel(torch.nn.Module):
    def __init__(self):
        super(OllamaModel, self).__init__()
        self.layer = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.layer(x)

# Initialize the model
model = OllamaModel().to(device)

# Example input tensor
input_tensor = torch.randn(1, 10).to(device)

# Perform inference
output = model(input_tensor)

# Print the output
print(output)

# Handle nonzero operation warning by checking macOS version
mac_version = platform.mac_ver()[0]
if mac_version and float(mac_version.split('.')[1]) < 14:
    print("Warning: macOS version is less than 14.0, nonzero operation may fall back to CPU.")
    nonzero_finite_vals = torch.masked_select(output.cpu(), output.cpu() != 0)
else:
    nonzero_finite_vals = torch.masked_select(output, output != 0)
print(nonzero_finite_vals)