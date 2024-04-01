import torch
import torch.nn as nn
from torch import optim
import time


class SimpleNN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(in_features=n_inputs, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=1024)
        self.linear4 = nn.Linear(in_features=1024, out_features=2048)
        self.linear5 = nn.Linear(in_features=2048, out_features=1024)
        self.linear6 = nn.Linear(in_features=1024, out_features=512)
        self.linear7 = nn.Linear(in_features=512, out_features=128)
        self.relu = nn.ReLU()
        self.linear8 = nn.Linear(in_features=128, out_features=n_outputs)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        x3 = self.linear3(x2)
        x4 = self.linear4(x3)
        x5 = self.linear5(x4)
        x6 = self.linear6(x5)
        x7 = self.linear7(x6)
        x8 = self.relu(x7)
        y_pred = self.linear8(x8)
        return y_pred


if __name__ == "__main__":
    batch_size = 32
    n_inputs = 1
    n_outputs = 1
    learning_rate = 0.01

    device = torch.device("cpu")
    model = SimpleNN(n_inputs, n_outputs).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    x = torch.rand(batch_size, n_inputs).to(device) * 2 - 1
    y = torch.sin(x).to(device)

    epochs = 1000

    model.train()
    start = time.time()
    for i in range(epochs):
        x = torch.rand(batch_size, n_inputs).to(device) * 2 - 1
        y = torch.sin(x).to(device)

        outputs = model(x)
        loss = loss_func(outputs, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{i + 1}/{epochs}],\t Loss: {loss.item()}")

    print(f"Training time: {time.time() - start:.2f} seconds. Loss: {loss.item()}")
