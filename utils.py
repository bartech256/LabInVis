# """
# Responsibility:
# - Provide helper functions used across the project.
# - Includes:
#   - Hardware info (CPU/GPU availability).
#   - Save/load config files.
#   - Save evaluation metrics.
#   - Generate unique experiment names.
# """


# import json
# import yaml
# import torch

# # Function that prints hardware information - active GPU and more
# def print_hardware_info():
#     print("Hardware Check:")
#     print(f"CUDA available: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"GPU: {torch.cuda.get_device_name()}")
#     print(f"CPU threads: {torch.get_num_threads()}")

# # Function that saves the exact settings (YAML)
# def save_config(cfg, path):
#     with open(path,"w") as f:
#         yaml.dump(cfg.__dict__, f)

# # Function that saves the results (JSON)
# def save_metrics(metrics, path):
#     with open(path,"w") as f:
#         json.dump(metrics, f, indent=4)

"""
Responsibility:
- Provide helper functions used across the project.
- Includes:
  - Hardware info (CPU/GPU availability).
  - Save/load config files.
  - Save evaluation metrics.
  - Generate unique experiment names.
"""

import json
import yaml
import torch

# Function that prints hardware information - active GPU and more
def print_hardware_info():
    print("Hardware Check:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CPU threads: {torch.get_num_threads()}")

# Function that saves the exact settings (YAML)
def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.dump(cfg.__dict__, f)

# Function that saves the results (JSON)
def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
