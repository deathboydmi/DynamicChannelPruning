import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

from dynamic_pruning import DynamicPruning


class FireDP(nn.Module):
    def __init__(self, fire):
        super(FireDP, self).__init__()

        self.squeeze = fire.squeeze
        self.squeeze_activation = fire.squeeze_activation

        self.expand1x1 = fire.expand1x1
        self.expand1x1_activation = fire.expand1x1_activation

        self.prun = DynamicPruning(self.squeeze.out_channels)

        self.expand3x3 = fire.expand3x3
        self.expand3x3_activation = fire.expand3x3_activation

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        dp_res = self.prun(x)
        dp_res = dp_res * x

        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(dp_res))
        ], 1)


class SqueezeNet_DP(nn.Module):
    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet_DP, self).__init__()

        self.num_classes = num_classes

        pretrain_model = models.squeezenet1_1(
            pretrained=True, num_classes=num_classes)

        self.features = pretrain_model.features
        self.features[3] = FireDP(fire=self.features[3])
        self.features[4] = FireDP(fire=self.features[4])
        self.features[6] = FireDP(fire=self.features[6])
        self.features[7] = FireDP(fire=self.features[7])

        self.classifier = pretrain_model.classifier

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x.view(x.size(0), self.num_classes)


if __name__ == '__main__':
    model = SqueezeNet_DP()
    print(model)
    exit()
    rand_input = torch.randn(4, 3, 224, 224)

    # if number of DP-models > 1 then better use hooks with global loss
    out, dp_out = model(rand_input)
    print(dp_out)

    dp_ratiomean = torch.empty(4, 1, 1, 1)
    dp_ratiomean.fill_(0.0)

    dp_loss = torch.dist(dp_out, dp_ratiomean)

    print(dp_loss)

# end_of_file
