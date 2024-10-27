import argparse
import torch
import numpy as np
from diffusers import DiffusionPipeline
import time
from accelerate import Accelerator



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--num_images_per_class", type=int, default=10)

    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision='fp16')
    dit_src = "facebook/DiT-XL-2-256"

    dit_pipe = DiffusionPipeline.from_pretrained(dit_src)
    dit_pipe.to(accelerator.device)
    num_images_per_class = args.num_images_per_class
    num_classes = 1000  # ImageNet has 1,000 classes
    num_steps = args.num_steps

    # Set a seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    # Dictionary to store generated images for each class
    generated_images = {class_label: [] for class_label in range(num_classes)}
    # class_labels = torch.arange(num_classes)

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self):
            # Create a list containing 50 occurrences of each number from 1 to 1000
            self.data = []
            for i in range(0, num_classes):
                self.data.extend([i] * args.num_images_per_class)  # Add the number i, 50 times to the dataset

        def __len__(self):
            # Return the total number of items in the dataset
            return len(self.data)

        def __getitem__(self, index):
            # Return the item at the given index
            return self.data[index]

    # Step 2: Create an instance of the dataset and a DataLoader
    dataset = CustomDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=1)
    dataloader, dit_pipe = accelerator.prepare([dataloader, dit_pipe])

    total_time = 0
    start_time = time.time()
    with torch.no_grad():
        for class_labels in dataloader:
            # print(f"Rank {accelerator.local_process_index}: Received class_labels {class_labels}")
            images = dit_pipe(generator=generator, class_labels=class_labels, num_inference_steps=args.num_steps).images
            torch.cuda.synchronize()
            for (img_idx, class_label) in list(enumerate(class_labels)):
                generated_images[class_label.item()].append(np.array(images[img_idx]))
    end_time = time.time()
    total_time = end_time - start_time

    print(f'Total time spent in generating {num_images_per_class*num_classes} images is {total_time}.')
    print(f'Average time spent in generating 1 image is {total_time/(num_images_per_class*num_classes)}.')
    print(f'Average time spent in 1 denoising step is {total_time/(num_images_per_class*num_classes*num_steps)}.')
    # Convert the dictionary to arrays for saving in npz format
    # For example, shape could be (num_classes, num_images_per_class, height, width, channels)
    # Make sure to adjust the dtype if necessary (e.g., using uint8 for image data)
    npz_data = {
        str(class_label): np.array(images, dtype=np.uint8)
        for class_label, images in generated_images.items()
    }

    # Save the generated images to a .npz file
    output_path = f'./DiT_results/imagenet256_samples_{num_steps}_{num_images_per_class}x{num_classes}.npz'
    np.savez_compressed(output_path, **npz_data)

    print(f"Images saved to {output_path}.")

    # dit_pipe.to('cpu')
    # del dit_pipe
    # torch.cuda.empty_cache()

if __name__ == '__main__':
    main()