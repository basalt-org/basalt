import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time


class BostonHousing(Dataset):
    def __init__(self, data: pd.DataFrame):
        # Data: All columns except the last one / Target: Only the last column (MEDV)
        self.data = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
        self.target = torch.tensor(data.iloc[:, -1].values, dtype=torch.float32).view(
            -1, 1
        )

        # Normalize data
        self.data = (self.data - self.data.mean(dim=0)) / self.data.std(dim=0)

        # Create dataset
        self.dataset = TensorDataset(self.data, self.target)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    # Load data and split in training and testing sets
    df = pd.read_csv("./examples/data/housing.csv")

    TRAIN_PCT = 0.99
    shuffled_df = df.sample(frac=1, random_state=42)
    train_df = shuffled_df[: int(TRAIN_PCT * len(df))]
    test_df = shuffled_df[int(TRAIN_PCT * len(df)) :]

    train_data = BostonHousing(train_df)
    test_data = BostonHousing(test_df)

    # Train Parameters
    batch_size = 32
    num_epochs = 200
    learning_rate = 0.02

    # Batchwise data loader
    loaders = {
        "train": DataLoader(
            train_data, batch_size=batch_size, shuffle=False, num_workers=1
        ),
        "test": DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=1
        ),
    }

    device = torch.device("cpu")
    # model = torch.compile(LinearRegression(train_data.data.shape[1]), fullgraph=True, options={"epilogue_fusion": True, "max_autotune": True})
    model = LinearRegression(train_data.data.shape[1])
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # it seems the python for loop is what is making the program slow (so pytorch has a disadvantage thanks to python)
    model.train()

    start = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        for batch_data, batch_labels in loaders["train"]:
            start_batch = time.time()

            # Forward pass
            outputs = model(batch_data)
            loss = loss_func(outputs, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # print time in ms
            # print(f'Batch time: {1000 * (time.time() - start_batch):.2f} ms') # The speed of a batch in basalt and pytorch are similar or pytorch can be faster

        print(
            f"Epoch [{epoch + 1}/{num_epochs}],\t Avg loss per epoch:"
            f" {epoch_loss / num_batches}"
        )

    print(f"Training time: {time.time() - start:.2f} seconds")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        test_predictions = model(test_data.data)
        mse_loss = loss_func(test_predictions, test_data.target).item()
        print(f"Mean Squared Error on Test Data: {mse_loss:.4f}")
