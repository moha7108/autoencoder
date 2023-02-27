import numpy
import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms


transform = transforms.ToTensor()

mnist = datasets.MNIST(root ='./data', train = True, download= True, transform=transform)

data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=64, shuffle=True)
