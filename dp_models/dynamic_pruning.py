import math
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init


class Binarization(Function):

    @staticmethod
    def forward(x, input):
        x.save_for_backward(input)
        output = (input > 0.5).float()
        return output

    @staticmethod
    def backward(x, grad_output):
        input, = x.saved_variables
        grad_input = grad_output * ((input - 0.5).abs() < 0.5).float()
        return grad_input


class DynamicPruningBinarization(nn.Module):
    def __init__(self, num_ch, feature_map_size):
        super(DynamicPruningBinarization, self).__init__()

        self.gavgpool = nn.AvgPool2d(feature_map_size, stride=1)
        self.fc1 = nn.Conv2d(num_ch, num_ch // 16, kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(num_ch // 16, num_ch, kernel_size=1, stride=1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, mean=0.75)
        nn.init.constant_(self.fc2.bias, 0)

        self.bin = Binarization()

    def forward(self, x):
        x = self.gavgpool(x)

        x = self.fc1(x)
        x = torch.clamp(x, 0, 1)

        x = self.fc2(x)
        x = self.bin.apply(x)

        return x


class DynamicPruning(nn.Module):
    def __init__(self, num_ch, feature_map_size):
        super(DynamicPruning, self).__init__()

        self.gavgpool = nn.AvgPool2d(feature_map_size, stride=1)
        self.fc1 = nn.Conv2d(num_ch, num_ch // 16, kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(num_ch // 16, num_ch, kernel_size=1, stride=1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        x = self.gavgpool(x)

        x = self.fc1(x)
        x = torch.clamp(x, 0, 1)

        x = self.fc2(x)
        x = torch.clamp(x, 0, 1)

        return x
