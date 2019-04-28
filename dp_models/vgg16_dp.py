import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

from dynamic_pruning import DynamicPruning


class Conv2d_dp(nn.Module):
    def __init__(self, in_ch, out_ch, feature_map_size):
        super(Conv2d_dp, self).__init__()

        self.prun = DynamicPruning(in_ch, feature_map_size)
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        dp_res = self.prun(x)
        x = x * dp_res
        x = self.conv(x)

        return x


class VGG16_DP(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16_DP, self).__init__()

        pretrain_model = models.vgg16_bn(
            pretrained=True, num_classes=num_classes)

        self.features = pretrain_model.features

        temp = self.features[40]
        self.features[40] = Conv2d_dp(512, 512, 14)
        self.features[40].conv = temp
        temp = self.features[37]
        self.features[37] = Conv2d_dp(512, 512, 14)
        self.features[37].conv = temp
        temp = self.features[34]
        self.features[34] = Conv2d_dp(512, 512, 14)
        self.features[34].conv = temp
        temp = self.features[30]
        self.features[30] = Conv2d_dp(512, 512, 28)
        self.features[30].conv = temp
        temp = self.features[27]
        self.features[27] = Conv2d_dp(512, 512, 28)
        self.features[27].conv = temp
        temp = self.features[24]
        self.features[24] = Conv2d_dp(256, 512, 28)
        self.features[24].conv = temp

        self.classifier = pretrain_model.classifier

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


if __name__ == "__main__":

    vgg_dp = VGG16_DP()
    print(vgg_dp)
    exit()

    r"_________Compute Flops___________"

    def compute_flops_hook(self, input, output):
        flops = 0
        if str(type(self).__name__) == 'Conv2d':
            output_width = output.size()[3]
            output_height = output.size()[2]
            kernel_size = self.kernel_size
            ks = self.weight.data.size()
            flops += ks[0] * ks[1] * ks[2] * \
                ks[3] * output_width * output_height
            print("=======")
            print(input[0].size())
            print(output.size())
            print(self.weight.data.size())
            print("=======")
        elif str(type(self).__name__) == 'Linear':
            flops += input[0].size()[1] * output.size()[1]

        print(self, " flops: ", flops)
        print()
        print()

    def add_hook(m):
        m.register_forward_hook(compute_flops_hook)

    vgg = VGG16_DP(num_classes=1000)

    add_hook(vgg.start_conv)
    add_hook(vgg.conv0)
    add_hook(vgg.conv1)
    add_hook(vgg.conv2)
    add_hook(vgg.conv3)
    add_hook(vgg.conv4)
    add_hook(vgg.conv5)
    add_hook(vgg.conv6)
    add_hook(vgg.conv7)
    add_hook(vgg.conv8)
    add_hook(vgg.conv9)
    add_hook(vgg.conv10)
    add_hook(vgg.conv11)

    in_ = torch.rand(1, 3, 224, 224)

    out = vgg(in_)

    r"""=======
   torch.Size([1, 3, 224, 224])
   torch.Size([1, 64, 224, 224])
   torch.Size([64, 3, 3, 3])
   =======
   Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 86704128


   =======
   torch.Size([1, 64, 224, 224])
   torch.Size([1, 64, 224, 224])
   torch.Size([64, 64, 3, 3])
   =======
   Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 1849688064


   =======
   torch.Size([1, 64, 112, 112])
   torch.Size([1, 128, 112, 112])
   torch.Size([128, 64, 3, 3])
   =======
   Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 924844032


   =======
   torch.Size([1, 128, 112, 112])
   torch.Size([1, 128, 112, 112])
   torch.Size([128, 128, 3, 3])
   =======
   Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 1849688064


   =======
   torch.Size([1, 128, 56, 56])
   torch.Size([1, 256, 56, 56])
   torch.Size([256, 128, 3, 3])
   =======
   Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 924844032


   =======
   torch.Size([1, 256, 56, 56])
   torch.Size([1, 256, 56, 56])
   torch.Size([256, 256, 3, 3])
   =======
   Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 1849688064


   =======
   torch.Size([1, 256, 56, 56])
   torch.Size([1, 256, 56, 56])
   torch.Size([256, 256, 3, 3])
   =======
   Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 1849688064


   =======
   torch.Size([1, 256, 28, 28])
   torch.Size([1, 512, 28, 28])
   torch.Size([512, 256, 3, 3])
   =======
   Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 924844032


   =======
   torch.Size([1, 512, 28, 28])
   torch.Size([1, 512, 28, 28])
   torch.Size([512, 512, 3, 3])
   =======
   Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 1849688064


   =======
   torch.Size([1, 512, 28, 28])
   torch.Size([1, 512, 28, 28])
   torch.Size([512, 512, 3, 3])
   =======
   Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 1849688064


   =======
   torch.Size([1, 512, 14, 14])
   torch.Size([1, 512, 14, 14])
   torch.Size([512, 512, 3, 3])
   =======
   Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 462422016


   =======
   torch.Size([1, 512, 14, 14])
   torch.Size([1, 512, 14, 14])
   torch.Size([512, 512, 3, 3])
   =======
   Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 462422016


   =======
   torch.Size([1, 512, 14, 14])
   torch.Size([1, 512, 14, 14])
   torch.Size([512, 512, 3, 3])
   =======
   Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  flops 462422016"""
