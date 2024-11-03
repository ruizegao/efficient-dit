import os
from datasets import load_from_disk, load_dataset
from PIL import Image
import uuid

# Define the path where the .arrow files are stored and output directory
dataset_path = "/home/ruizeg2/PycharmProjects/efficient-dit/DiT/imagenet/ILSVRC___imagenet-1k/default/1.0.0/07900defe1ccf3404ea7e5e876a64ca41192f6c07406044771544ef1505831e8"  # Directory where the .arrow files are stored
output_dir = "/home/ruizeg2/PycharmProjects/efficient-dit/DiT/imagenet/ILSVRC___imagenet-1k/default/1.0.0"  # Output directory for organized dataset

# Create output directories
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

# Function to save images from the dataset
def save_images_by_class(dataset, split_name):
    for (idx, item) in enumerate(dataset):
        image = item["image"]
        label = item["label"]
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        # Create class directory for saving images
        class_dir = os.path.join(output_dir, split_name, str(label))  # Adjust if labels are strings
        os.makedirs(class_dir, exist_ok=True)

        # Save image
        image_id = str(uuid.uuid4())  # Use a unique identifier if available
        image_path = os.path.join(class_dir, f"{image_id}.jpeg")
        image.save(image_path, format="JPEG")


# Load and save train images
train_files = [f for f in os.listdir(dataset_path) if "train" in f]
# interrupt_idx = train_files.index("imagenet-1k-train-00064-of-00257.arrow")
# print(train_files[:interrupt_idx])
for train_file in train_files:
    print(f"Loading train file: {train_file}")  # Debug statement
    train_dataset = load_dataset("arrow", data_files=os.path.join(dataset_path, train_file))["train"]
    save_images_by_class(train_dataset, "train")

# Load and save validation images
val_files = [f for f in os.listdir(dataset_path) if "val" in f]
for val_file in val_files:
    print(f"Loading val file: {val_file}")  # Debug statement
    val_dataset = load_dataset("arrow", data_files=os.path.join(dataset_path, val_file))["train"]
    save_images_by_class(val_dataset, "val")

# Load and save test images
test_files = [f for f in os.listdir(dataset_path) if "test" in f]
for test_file in test_files:
    print(f"Loading test file: {test_file}")  # Debug statement
    test_dataset = load_dataset("arrow", data_files=os.path.join(dataset_path, test_file))["train"]
    save_images_by_class(test_dataset, "test")

print("Images have been organized into train, val, and test folders.")

# from datasets import load_from_disk
#
# # Specify the path to the directory containing the Arrow files
# path_to_dataset = '/home/ruizeg2/PycharmProjects/efficient-dit/DiT/imagenet/ILSVRC___imagenet-1k/default/1.0.0/07900defe1ccf3404ea7e5e876a64ca41192f6c07406044771544ef1505831e8'
#
# # Load the dataset from disk, specifying only the training split
# dataset = load_from_disk(path_to_dataset)
#
# # Access the training split
# train_dataset = dataset['train']
#
# # Print some information about the training dataset
# print("Loaded training dataset:")
# print(train_dataset)
