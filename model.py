import types

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from dg_methods import MixStyle


def _forward_with_mixstyle(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.mixstyle1(x)
    x = self.layer2(x)
    x = self.mixstyle2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.flatten(1)
    x = self.fc(x)
    return x


def build_resnet18(
    num_classes=2,
    pretrained=True,
    mixstyle=False,
    mixstyle_p=0.5,
    mixstyle_alpha=0.1,
):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if mixstyle:
        model.mixstyle1 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        model.mixstyle2 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        model.forward = types.MethodType(_forward_with_mixstyle, model)

    return model
