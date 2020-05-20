import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import BasicBlock

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


class NonFcResNet(models.ResNet):
    def __init__(self, output_lenght, block, layers, num_classes, **kwargs):
        super(NonFcResNet, self).__init__(block, layers, num_classes, **kwargs)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((output_lenght, 1))

        self.ch2cls = torch.nn.Conv2d(
            in_channels=512 * block.expansion, out_channels=num_classes, kernel_size=1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu(self.avgpool(x))
        x = self.ch2cls(x)

        # shape = [batch,channel,width]
        x = x.squeeze(-1)
        x = nn.functional.log_softmax(x, 1)
        return x


def make_model(
    arch, layers, num_classes, label_length, pretrained, progress=True, **kwargs
):
    model = NonFcResNet(label_length, BasicBlock, layers, num_classes, **kwargs)
    if pretrained:
        state_dict = models.load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    import DataSet

    model = make_model("resnet50", [3, 4, 6, 3], DataSet.num_classes, False)
