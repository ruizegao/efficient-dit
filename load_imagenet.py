from datasets import load_dataset
from huggingface_hub import login

login("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")
dataset = load_dataset("ILSVRC/imagenet-1k", cache_dir="/home/ruizeg2/PycharmProjects/efficient-dit/DiT/imagenet")
