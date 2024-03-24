import torch
import torch.nn as nn
from torch import optim
import time


class SimpleNN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(in_features=n_inputs, out_features=32)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=32, out_features=32)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=32, out_features=n_outputs)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.relu1(x1)
        x3 = self.linear2(x2)
        x4 = self.relu2(x3)
        y_pred = self.linear3(x4)
        return y_pred
    

if __name__ == '__main__':
    batch_size = 32
    n_inputs = 1
    n_outputs = 1
    learning_rate = 0.01

    device = torch.device('cpu')
    model = SimpleNN(n_inputs, n_outputs).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    x = torch.rand(batch_size, n_inputs).to(device) * 2 - 1
    y = torch.sin(x).to(device)
    
    epochs = 10000

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

        if i - 1 % 1000 == 0:
            print(f'Epoch [{i + 1}/{epochs}],\t Loss: {loss.item()}')

    print(f"Training time: {time.time() - start:.2f} seconds. Loss: {loss.item()}")