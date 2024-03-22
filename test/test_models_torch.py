import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegression(nn.Module):
    def __init__(self, linear1_weights, linear1_bias):
        super(LinearRegression, self).__init__()
        
        self.linear1_weights = nn.Parameter(linear1_weights)
        self.linear1_bias = nn.Parameter(linear1_bias)
        
    def forward(self, x):
        output = F.linear(x, self.linear1_weights.T, self.linear1_bias)
        return output


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        loss = F.mse_loss(output, target)
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        loss = -torch.sum(target * torch.log(output)) / output.size(0)
        return loss

# Implement the class for crossentropy loss with logsoftmax
# use logsoftmax

class CrossEntropyLoss2(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2, self).__init__()

    def forward(self, output, target):
        loss = -torch.sum(target * F.log_softmax(output, dim=1)) / output.size(0)
        return loss  


class CNN(nn.Module):
    def __init__(self, conv1_weights, conv1_bias, conv2_weights, conv2_bias, linear1_weights, linear1_bias):
        super(CNN, self).__init__()

        self.conv1_weights = nn.Parameter(conv1_weights)
        self.conv1_bias = nn.Parameter(conv1_bias)
        self.conv2_weights = nn.Parameter(conv2_weights)
        self.conv2_bias = nn.Parameter(conv2_bias)
        self.linear1_weights = nn.Parameter(linear1_weights)
        self.linear1_bias = nn.Parameter(linear1_bias)

    def forward(self, x):
        x = F.conv2d(x, self.conv1_weights,
                     self.conv1_bias, stride=1, padding=2)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.conv2d(x, self.conv2_weights,
                     self.conv2_bias, stride=1, padding=2)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        output = F.linear(x, self.linear1_weights.T, self.linear1_bias)
        return output

    def print_grads(self):
        print("\nCONV1 WEIGHTS", self.conv1_weights.grad.shape)
        print(self.conv1_weights.grad)

        print("\nCONV1 BIAS", self.conv1_bias.grad.shape)
        print(self.conv1_bias.grad)

        print("\nCONV2 WEIGHTS", self.conv2_weights.grad.shape)
        print(self.conv2_weights.grad)

        print("\nCONV2 BIAS", self.conv2_bias.grad.shape)
        print(self.conv2_bias.grad)

        print("\nLINEAR1 WEIGHTS", self.linear1_weights.grad.shape)
        print(self.linear1_weights.grad)

        print("\nLINEAR1 BIAS", self.linear1_bias.grad.shape)
        print(self.linear1_bias.grad)


import numpy as np
if __name__ == "__main__":

    t_pred = np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]])
    t_true = np.array([[0, 0, 1], [1, 0, 0]])

    loss = CrossEntropyLoss2()
    t_pred = torch.tensor(t_pred, dtype=torch.float32)
    t_true = torch.tensor(t_true, dtype=torch.float32)

    print(loss(t_pred, t_true))

    print("-------")


    t_pred = np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]])
    t_true = np.array([[0, 1, 0], [0, 0, 1]])

    loss = CrossEntropyLoss2()
    t_pred = torch.tensor(t_pred, dtype=torch.float32)
    t_true = torch.tensor(t_true, dtype=torch.float32)

    print(loss(t_pred, t_true))




