import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        output =  output + 1e-9
        loss = -torch.sum(target * torch.log(output)) / output.size(0)
        return loss

# Implement the class for crossentropy loss with logsoftmax
# use logsfomtax

class CrossEntropyLoss2(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss2, self).__init__()

    def forward(self, output, target):
        output =  output + 1e-9
        loss = -torch.sum(target * F.log_softmax(output, dim=-1)) / output.size(0)
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
        x = F.conv2d(x, self.conv2_weights,
                     self.conv2_bias, stride=1, padding=2)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        output = F.linear(x, self.linear1_weights.T, self.linear1_bias)
        output = F.softmax(output, dim=-1)
        return output
