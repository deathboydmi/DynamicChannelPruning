import math
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
import torchvision.models as models

from dynamic_pruning import DP_Conv2d


class ResNet34_DP(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet34_DP, self).__init__()

        pretrain_model = models.resnet34(
            pretrained=True, num_classes=num_classes)

        self.conv1 = pretrain_model.conv1
        self.bn1 = pretrain_model.bn1
        self.relu = pretrain_model.relu
        self.maxpool = pretrain_model.maxpool

        self.layer1 = pretrain_model.layer1
        self.layer2 = pretrain_model.layer2
        self.layer3 = pretrain_model.layer3
        self.layer4 = pretrain_model.layer4

        self.layer3[5].conv1 = DP_Conv2d(conv2d=self.layer3[5].conv1)
        self.layer3[5].conv2 = DP_Conv2d(conv2d=self.layer3[5].conv2)
        self.layer4[1].conv1 = DP_Conv2d(conv2d=self.layer4[1].conv1)
        self.layer4[1].conv2 = DP_Conv2d(conv2d=self.layer4[1].conv2)
        self.layer4[2].conv1 = DP_Conv2d(conv2d=self.layer4[2].conv1)
        self.layer4[2].conv2 = DP_Conv2d(conv2d=self.layer4[2].conv2)

        self.avgpool = pretrain_model.avgpool
        self.fc = pretrain_model.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


i = 0


def compute_flops_hook(self, input, output):
    global i
    flops = 0
    if str(type(self).__name__) == 'Conv2d':
        output_width = output.size()[3]
        output_height = output.size()[2]
        kernel_size = self.kernel_size
        ks = self.weight.data.size()
        flops += ks[0] * ks[1] * ks[2] * ks[3] * output_width * output_height
        print("=======")
        # print(input[0].size())
        # print(output.size())
        # print(self.weight.data.size())
        print("=======")
    elif str(type(self).__name__) == 'Linear':
        flops += input[0].size()[1] * output.size()[1]

    print(i, ") ", self, " flops: ", flops)
    print()
    print()
    i += 1


if __name__ == "__main__":

    model = ResNet34_DP()
    print(model)
    exit()

    model.layer3[5].conv1.register_forward_hook(compute_flops_hook)
    model.layer3[5].conv2.register_forward_hook(compute_flops_hook)
    model.layer4[1].conv1.register_forward_hook(compute_flops_hook)
    model.layer4[1].conv2.register_forward_hook(compute_flops_hook)
    model.layer4[2].conv1.register_forward_hook(compute_flops_hook)
    model.layer4[2].conv2.register_forward_hook(compute_flops_hook)

    input = torch.rand(4, 3, 224, 224)
    output = model(input)
