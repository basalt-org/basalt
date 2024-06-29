import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import imageio.v3 as iio


class CIFAR10(Dataset):
    def __init__(self, image_folder: str, label_file: str):
        self.file_paths = glob(f"{image_folder}/*png")

        with open(label_file) as f:
            label_list = f.read().splitlines()

        self.label_dict = dict(zip(label_list, range(len(label_list))))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        image = iio.imread(self.file_paths[idx])
        image = np.transpose(image, (2, 0, 1))
        label = self.label_dict[Path(self.file_paths[idx]).stem.split("_")[1]]

        # Normalize data
        image = image / 255.0
        return image.astype(np.float32), label


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


if __name__ == "__main__":
    num_epochs = 2
    batch_size = 8
    learning_rate = 1e-3

    # Load data
    train_data = CIFAR10(image_folder="./examples/data/cifar/train/", label_file="./examples/data/cifar/labels.txt")

    # Visualize data
    #num = 0
    #plt.imshow(np.array(train_data[num][0]).squeeze())
    #plt.title("%i" % train_data[num][1])
    #plt.show()

    # Batchwise data loader
    loaders = {
        "train": DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=1
        ),
    }

    device = torch.device("cpu")
    cnn = CNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the model
    cnn.train()
    total_step = len(loaders["train"])
    start = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(loaders["train"]):
            b_x = Variable(images)
            b_y = Variable(labels)

            output = cnn(b_x)
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % 100 == 0:
                print(
                    "Epoch [{}/{}],\t Step [{}/{}],\t Loss: {:.6f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, epoch_loss/(i+1)
                    )
                )

    print(f"Training time: {time.time() - start:.2f} seconds")

    # Export to ONNX
    export_onnx = os.environ.get("export_onnx", 0)
    if export_onnx == "1":
        dummy_input = torch.randn(1, 3, 32, 32)

        # cnn.out.weight = nn.Parameter(cnn.out.weight.T) # transpose because torch saves the weight of linear layer as (output_dim, input_dim) (so they transposed and there is not a real reason for this)
        torch.onnx.export(cnn, dummy_input, "./examples/data/mnist_torch.onnx", verbose=True)