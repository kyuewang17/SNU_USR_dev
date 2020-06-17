import torch.nn as nn
import torchvision
from .__base__ import BackboneBase
from .__module__ import ExtraResModule


class RefineDetResNet34(BackboneBase):
    def build(self):
        resnet34 = torchvision.models.resnet34(pretrained=self.pretrained)
        extra = ExtraResModule(512, 128)

        net = dict()
        net['stage0'] = nn.Sequential(
            resnet34.conv1, resnet34.bn1, resnet34.relu,
            resnet34.maxpool, resnet34.layer1, resnet34.layer2)
        net['stage1'] = resnet34.layer3
        net['stage2'] = resnet34.layer4
        net['stage3'] = extra
        self.net = nn.ModuleDict(net)

    def forward(self, img):
        img = (img - self.mean) / self.std
        feature0 = self.net['stage0'](img)
        feature1 = self.net['stage1'](feature0)
        feature2 = self.net['stage2'](feature1)
        feature3 = self.net['stage3'](feature2)
        return [feature0, feature1, feature2, feature3]
