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
class CrossEntropyLoss2(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2, self).__init__()

    def forward(self, output, target):
        loss = -torch.sum(target * F.log_softmax(output, dim=1)) / output.size(
            0
        )
        return loss


class CNN(nn.Module):
    def __init__(
        self,
        conv1_weights,
        conv1_bias,
        conv2_weights,
        conv2_bias,
        linear1_weights,
        linear1_bias,
    ):
        super(CNN, self).__init__()

        self.conv1_weights = nn.Parameter(conv1_weights)
        self.conv1_bias = nn.Parameter(conv1_bias)
        self.conv2_weights = nn.Parameter(conv2_weights)
        self.conv2_bias = nn.Parameter(conv2_bias)
        self.linear1_weights = nn.Parameter(linear1_weights)
        self.linear1_bias = nn.Parameter(linear1_bias)

    def forward(self, x):
        x = F.conv2d(
            x, self.conv1_weights, self.conv1_bias, stride=1, padding=2
        )
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.conv2d(
            x, self.conv2_weights, self.conv2_bias, stride=1, padding=2
        )
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


class SimpleNN(nn.Module):
    def __init__(
        self,
        linear1_weights,
        linear1_bias,
        linear2_weights,
        linear2_bias,
        linear3_weights,
        linear3_bias,
    ):
        super(SimpleNN, self).__init__()

        self.linear1_weights = nn.Parameter(linear1_weights)
        self.linear1_bias = nn.Parameter(linear1_bias)
        self.linear2_weights = nn.Parameter(linear2_weights)
        self.linear2_bias = nn.Parameter(linear2_bias)
        self.linear3_weights = nn.Parameter(linear3_weights)
        self.linear3_bias = nn.Parameter(linear3_bias)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x1 = F.linear(x, self.linear1_weights.T, self.linear1_bias)
        x2 = self.relu1(x1)
        x3 = F.linear(x2, self.linear2_weights.T, self.linear2_bias)
        x4 = self.relu2(x3)
        y_pred = F.linear(x4, self.linear3_weights.T, self.linear3_bias)
        return y_pred
