import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset


class MNIST(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.labels = torch.tensor(data.iloc[:, 0].values, dtype=torch.int64)
        self.images = torch.tensor(
            data.iloc[:, 1:].values, dtype=torch.float32
        ).reshape(-1, 1, 28, 28)

        # Normalize data
        self.images = self.images / 255.0

        self.dataset = TensorDataset(self.images, self.labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
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
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


if __name__ == "__main__":
    num_epochs = 20
    batch_size = 4
    learning_rate = 1e-3

    # Load data
    train_data = MNIST("./examples/data/mnist_train_small.csv")
    test_data = MNIST("./examples/data/mnist_test_small.csv")

    # Visualize data
    num = 0
    plt.imshow(np.array(train_data[num][0]).squeeze())
    plt.title("%i" % train_data[num][1])
    plt.show()

    # Batchwise data loader
    loaders = {
        "train": DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=1
        ),
        "test": DataLoader(
            test_data, batch_size=batch_size, shuffle=True, num_workers=1
        ),
    }

    device = torch.device("cpu")
    cnn = CNN()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the model
    cnn.train()
    total_step = len(loaders["train"])
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders["train"]):
            b_x = Variable(images)
            b_y = Variable(labels)

            output = cnn(b_x)
            loss = loss_func(output, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                "Epoch [{}/{}],\t Step [{}/{}],\t Loss: {:.6f}".format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()
                )
            )

    # Evaluate the model
    cnn.eval()
    with torch.no_grad():
        test_predictions = cnn(test_data.images)
        ce_loss = loss_func(test_predictions, test_data.labels).item()
        print(f"Cross Entropy Loss on Test Data: {ce_loss:.4f}")
