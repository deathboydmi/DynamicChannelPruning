import math
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init

from torch.autograd import Function, Variable


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
    def __init__(self, in_channels, hidden_layer_channels):
        super(DynamicPruningBinarization, self).__init__()

        self.gavgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden_layer_channels,
                             kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(hidden_layer_channels,
                             in_channels, kernel_size=1, stride=1)

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


class DP_BN_Conv2d(nn.Module):
    def __init__(self, in_channels=0, out_channels=0,
                 kernel_size=0, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros',
                 conv2d=None, hidden_layer_channels=None):
        super(DP_BN_Conv2d, self).__init__()
        if in_channels is 0 or out_channels is 0 or kernel_size is 0:
            if conv2d is None:
                assert()
            else:
                self.__init__from_Conv2d(conv2d, hidden_layer_channels)
        else:
            if hidden_layer_channels is None:
                hidden_layer_channels = out_channels // 16
                if hidden_layer_channels < 4:
                    hidden_layer_channels = 4

            self.prun = DynamicPruningBinarization(
                in_channels, hidden_layer_channels)
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups, bias=bias,
                                  padding_mode=padding_mode)

    def __init__from_Conv2d(self, conv2d, hidden_layer_channels):
        if type(conv2d) != nn.Conv2d:
            assert()
        if hidden_layer_channels is None:
            hidden_layer_channels = conv2d.out_channels // 16
            if hidden_layer_channels < 4:
                hidden_layer_channels = 4

        self.prun = DynamicPruningBinarization(
            conv2d.in_channels, hidden_layer_channels)
        self.conv = conv2d

    def forward(self, x):
        dp_res = self.prun(x)
        x = x * dp_res
        x = self.conv(x)

        return x


class DynamicPruning(nn.Module):
    def __init__(self, in_channels, hidden_layer_channels):
        super(DynamicPruning, self).__init__()

        self.gavgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, hidden_layer_channels,
                             kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(hidden_layer_channels,
                             in_channels, kernel_size=1, stride=1)

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


class DP_Conv2d(nn.Module):
    def __init__(self, in_channels=0, out_channels=0,
                 kernel_size=0, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros',
                 conv2d=None, hidden_layer_channels=None):
        super(DP_Conv2d, self).__init__()
        if in_channels is 0 or out_channels is 0 or kernel_size is 0:
            if conv2d is None:
                assert()
            else:
                self.__init__from_Conv2d(conv2d, hidden_layer_channels)
        else:
            if hidden_layer_channels is None:
                hidden_layer_channels = out_channels // 16
                if hidden_layer_channels < 4:
                    hidden_layer_channels = 4

            self.prun = DynamicPruning(in_channels, hidden_layer_channels)
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size, stride=stride,
                                  padding=padding, dilation=dilation,
                                  groups=groups, bias=bias,
                                  padding_mode=padding_mode)

    def __init__from_Conv2d(self, conv2d, hidden_layer_channels):
        if type(conv2d) != nn.Conv2d:
            assert()
        if hidden_layer_channels is None:
            hidden_layer_channels = conv2d.out_channels // 16
            if hidden_layer_channels < 4:
                hidden_layer_channels = 4

        self.prun = DynamicPruning(
            conv2d.in_channels, hidden_layer_channels)
        self.conv = conv2d

    def forward(self, x):
        dp_res = self.prun(x)
        x = x * dp_res
        x = self.conv(x)

        return x


if __name__ == "__main__":
    x = torch.rand(1, 256, 112, 112)
    conv = nn.Conv2d(256, 512, 3, stride=1, padding=1)
    prunConv = DP_BN_Conv2d(conv2d=conv)
    print(prunConv(x))
