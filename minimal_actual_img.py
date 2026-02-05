import torch
import torchvision
import matplotlib.pyplot as plt

# Dataset (same index as your attack)
dataset = torchvision.datasets.CIFAR10(
    root="./data/cifar10",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

img, label = dataset[1200]  # same datapoint

# Convert CHW → HWC for matplotlib
img = img.permute(1, 2, 0)

plt.figure(figsize=(3, 3))
plt.imshow(img)
plt.title(f"Original image (label={label})")
plt.axis("off")
plt.show()
