import torch.nn as nn
from torchvision.models.resnet import BasicBlock


class ExtraResModule(nn.Module):
    def __init__(self, in_channels, internal_channels):
        """
        :param in_channels: number of forward feature channles
        :param internal_channels: number of internal channels
        """
        super(ExtraResModule, self).__init__()
        stride = 2
        expansion = 1
        out_channels = expansion * internal_channels
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.resbody = BasicBlock(in_channels, internal_channels,
                                  stride=stride, downsample=downsample)

    def forward(self, x):
        return self.resbody(x)


class FirstTCB(nn.Module):
    def __init__(self, in_ch, inter_ch, is_bnorm=False):
        super(FirstTCB, self).__init__()

        layers = list()
        layers.append(nn.Conv2d(in_ch, inter_ch, kernel_size=3, padding=1))
        if is_bnorm:
            layers.append(nn.BatchNorm2d(inter_ch))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(inter_ch, inter_ch, kernel_size=3, padding=1))
        if is_bnorm:
            layers.append(nn.BatchNorm2d(inter_ch))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(inter_ch, inter_ch, kernel_size=3, padding=1))
        if is_bnorm:
            layers.append(nn.BatchNorm2d(inter_ch))
        layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, fmap):
        return self.layers.forward((fmap))


class TCB(nn.Module):
    # Transfer Connection Block Architecture
    def __init__(self, fmap_channel, pyramid_channel, inter_channel=256, is_bnorm=False):
        super(TCB, self).__init__()

        self.is_batchnorm = is_bnorm
        # Use bias if is_batchnorm is False, donot otherwise.
        use_bias = not self.is_batchnorm
        # conv + bn + relu
        self.conv1 = nn.Conv2d(fmap_channel, inter_channel, kernel_size=3, padding=1, bias=use_bias)
        # ((conv2 + bn2) element-wise add  (deconv + deconv_bn)) + relu
        # batch normalization before element-wise addition
        self.conv2 = nn.Conv2d(inter_channel, inter_channel, kernel_size=3, padding=1, bias=use_bias)
        self.deconv = nn.ConvTranspose2d(pyramid_channel, inter_channel, kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias)
        # conv + bn + relu
        self.conv3 = nn.Conv2d(inter_channel, inter_channel, kernel_size=3, padding=1, bias=use_bias)
        self.relu = nn.ReLU(inplace=True)

        if self.is_batchnorm:
            self.bn1 = nn.BatchNorm2d(inter_channel)
            self.bn2 = nn.BatchNorm2d(inter_channel)
            self.deconv_bn = nn.BatchNorm2d(inter_channel)
            self.bn3 = nn.BatchNorm2d(inter_channel)
        # attribution
        self.out_channels = inter_channel

    def forward(self, lateral, x):
        if self.is_batchnorm:
            lateral_out = self.relu(self.bn1(self.conv1(lateral)))
            out = self.relu(self.bn2(self.conv2(lateral_out)) + self.deconv_bn(self.deconv(x)))
            out = self.relu(self.bn3(self.conv3(out)))
        else:
            lateral_out = self.relu(self.conv1(lateral))
            out = self.relu(self.conv2(lateral_out) + self.deconv(x))
            out = self.relu(self.conv3(out))
        return out

