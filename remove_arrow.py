import os
from datasets import load_from_disk, load_dataset
from PIL import Image
import uuid

# Define the path where the .arrow files are stored and output directory
dataset_path = "/home/ruizeg2/PycharmProjects/efficient-dit/DiT/imagenet/ILSVRC___imagenet-1k/default/1.0.0/07900defe1ccf3404ea7e5e876a64ca41192f6c07406044771544ef1505831e8"  # Directory where the .arrow files are stored
output_dir = "/home/ruizeg2/PycharmProjects/efficient-dit/DiT/imagenet/ILSVRC___imagenet-1k/default/1.0.0"  # Output directory for organized dataset



train_files = [f for f in os.listdir(dataset_path) if "train" in f]
interrupt_idx = train_files.index("imagenet-1k-train-00064-of-00257.arrow")
to_remove = train_files[:interrupt_idx]
for file_path in to_remove:
    try:
        os.remove(dataset_path+'/'+file_path)
        print(f"Removed: {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except PermissionError:
        print(f"Permission denied: {file_path}")
    except Exception as e:
        print(f"Error occurred while trying to remove {file_path}: {e}")
