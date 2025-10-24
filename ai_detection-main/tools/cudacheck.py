import torch, os
print("torch:", torch.__version__,
      "cuda:", torch.version.cuda,
      "is_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("cudnn:", torch.backends.cudnn.version(),
      "allow_tf32:", torch.backends.cuda.matmul.allow_tf32)