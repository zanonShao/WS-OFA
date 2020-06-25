import copy
import torch

from layers import *
from imagenet_codebase.utils import MyNetwork, make_divisible
from imagenet_codebase.networks.proxyless_nets import MobileInvertedResidualBlock


class MobileNetV3WS(MyNetwork):

    def __init__(self, first_conv, blocks, final_expand_layer, attention, bap, fc_conv):
        super(MobileNetV3WS, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.final_expand_layer = final_expand_layer
        # self.feature_mix_layer = feature_mix_layer
        # self.classifier = classifier
        self.attention = attention
        self.bap = bap
        self.fc_conv = fc_conv

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_expand_layer(x)
        attention_maps = F.relu(self.attentions(x), inplace=True)
        feature_matrix = self.bap(x, attention_maps)

        x = self.first_conv(feature_matrix.reshape((-1, max(self.final_expand_width) * self.M, 1, 1)) * 100.0)
        x = torch.squeeze(x)
        # x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        # x = self.feature_mix_layer(x)
        # x = torch.squeeze(x)
        # x = self.classifier(x)
        return x