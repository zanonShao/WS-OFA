import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from elastic_nn.modules.dynamic_layers import DynamicMBConvLayer, DynamicConvLayer, DynamicLinearLayer
from layers import ConvLayer, IdentityLayer, LinearLayer, MBInvertedConvLayer
from imagenet_codebase.networks.mobilenet_v3 import MobileNetV3, MobileInvertedResidualBlock
from imagenet_codebase.networks.mobilenet_v3_ws import MobileNetV3WS
from imagenet_codebase.utils import make_divisible, int2list
from elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3

# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP',feature_mix_layer=None):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B = features.size(0)
        M = attentions.size(1)
        for i in range(M):
            AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, 1, -1)
            if i == 0:
                feature_matrix = AiF
            else:
                feature_matrix = torch.cat([feature_matrix, AiF], dim=1)
        # feature_matrix = torch.mul(torch.sign(feature_matrix),torch.sqrt(torch.abs(feature_matrix)+1e-12))
        # feature_matrix = nn.functional.normalize(feature_matrix, 2, [1,2])
        return feature_matrix

class OFAMobileNetV3WS(OFAMobileNetV3):
    def __init__(self, n_classes=200, bn_param=(0.1, 1e-5), dropout_rate=0.1, base_stage_width=None,
                 width_mult_list=1.0, ks_list=3, expand_ratio_list=6, depth_list=4, M=32):
        self.M=M
        super(OFAMobileNetV3WS,self).__init__(n_classes, bn_param, dropout_rate, base_stage_width,
                width_mult_list, ks_list, expand_ratio_list, depth_list)

        if len(self.final_expand_width) == 1:
            self.feature_mix_layer_M = nn.ModuleList([ConvLayer(
                max(self.final_expand_width), max(self.last_channel), kernel_size=1, bias=False, use_bn=False, act_func='h_swish',
            ) for i in range(self.M)])
        else:
            self.feature_mix_layer_M = nn.ModuleList([DynamicConvLayer(
                in_channel_list=self.final_expand_width, out_channel_list=self.last_channel, kernel_size=1,
                use_bn=False, act_func='h_swish',
            )for i in range(self.M)])

        # self.feature_mix_layer=None
        # self.classifier=None

        self.attentions = nn.Conv2d(max(self.final_expand_width),M,kernel_size=1,bias=True)

        self.bap=BAP(pool='GAP')

        self.fc_conv = nn.Conv2d(self.M * max(self.last_channel),n_classes,kernel_size=1,bias=True)

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return 'OFAMobileNetV3WS'

    def forward(self, x):
        # exit(0)
        # first conv
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)

        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

        x = self.final_expand_layer(x)
        # if x.size()[0]>1:
        #     x = x.mean(3, keepdim=True).mean(2, keepdim=True)
        #     print(x)
        #     print(x.mean())
        #     exit(0)
        # print(x.size())
        attention_maps =F.relu(self.attentions(x),inplace=True)
        feature_matrix =self.bap(x,attention_maps)*0.1
        # if feature_matrix.size()[0]>1:
        #     print(feature_matrix[:,1:2,:])
        #     print(feature_matrix[:, 1:2, :].mean())
        #     print(feature_matrix[:,30:31,:])
        #     print(feature_matrix[:, 30:31, :].mean())
        #     exit(0)

        B = attention_maps.size(0)
        for i in range(self.M):
            each_part = self.feature_mix_layer_M[i](feature_matrix[:,i:i+1,:].view(B,-1,1,1)).view(B,1,-1)
            if i == 0:
                feature_matrix_temp = each_part
            else:
                feature_matrix_temp = torch.cat([feature_matrix_temp, each_part], dim=1)
        feature_matrix = feature_matrix_temp
        # print('feature_matrix2: ', feature_matrix.size())
        feature_matrix = torch.mul(torch.sign(feature_matrix),torch.sqrt(torch.abs(feature_matrix)+1e-12))
        feature_matrix = nn.functional.normalize(feature_matrix, 2, [1,2])
        # print('feature_matrix2: ', feature_matrix.size())
        # exit(0)

        # x = self.fc_conv(feature_matrix.reshape((-1,feature_matrix.size(1)*feature_matrix.size(2),1,1))*100.0)
        x = self.fc_conv(feature_matrix.reshape((-1, feature_matrix.size(1) * feature_matrix.size(2), 1, 1))*10.0)
        x = torch.squeeze(x)

        # x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        # x = self.feature_mix_layer(x)
        # x = torch.squeeze(x)
        # x = self.classifier(x)
        return x,feature_matrix

    @property
    def module_str(self):
        _str = self.first_conv.module_str + '\n'
        _str += self.blocks[0].module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'

        _str += self.final_expand_layer.module_str + '\n'
        # _str += self.feature_mix_layer.module_str + '\n'
        # _str += self.classifier.module_str + '\n'
        _str += 'Attention_Con2d'
        _str += 'BAP'
        _str += 'FC_by_Conv'

        return _str

    @property
    def config(self):
        return {
            'name': OFAMobileNetV3.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.first_conv.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'final_expand_layer': self.final_expand_layer.config,
            # 'feature_mix_layer': self.feature_mix_layer.config,
            # 'classifier': self.classifier.config,
            'Attention_Con2d' : {'name': 'Con2d','kernel_size': 1,},
            'BAP': {'name': 'BAP',},
            'FC_by_Conv': {'name': 'FC_by_Conv', 'kernel_size': 1,},
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    def load_weights_from_net(self, src_model_dict):
        model_dict = self.state_dict()
        # print(model_dict.keys())
        for key in src_model_dict:
            if key in model_dict:
                new_key = key
            elif '.bn.bn.' in key:
                new_key = key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in key:
                new_key = key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in key:
                new_key = key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in key:
                new_key = key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in key:
                new_key = key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in key:
                new_key = key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(key)
            assert new_key in model_dict, '%s' % new_key
            # if 'feature_mix_layer' or 'classifier' in new_key:
            if 'classifier' in new_key:
                model_dict.pop(new_key)
            else:
                if 'feature_mix_layer' in new_key:
                    #'feature_mix_layer_M.28.conv.weight', 'feature_mix_layer_M.29.conv.weight'
                    for i in range(self.M):
                        model_dict['feature_mix_layer_M.%d.conv.weight'%i] = src_model_dict[key]
                    model_dict[new_key] = src_model_dict[key]
                else:
                    model_dict[new_key] = src_model_dict[key]

        self.load_state_dict(model_dict,strict=False)


    """ set, sample and get active sub-networks """

    def set_active_subnet(self, wid=None, ks=None, e=None, d=None):
        width_mult_id = int2list(wid, 4 + len(self.block_group_info))
        ks = int2list(ks, len(self.blocks) - 1)
        expand_ratio = int2list(e, len(self.blocks) - 1)
        depth = int2list(d, len(self.block_group_info))

        for block, k, e in zip(self.blocks[1:], ks, expand_ratio):
            if k is not None:
                block.mobile_inverted_conv.active_kernel_size = k
            if e is not None:
                block.mobile_inverted_conv.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def set_constraint(self, include_list, constraint_type='depth'):
        if constraint_type == 'depth':
            self.__dict__['_depth_include_list'] = include_list.copy()
        elif constraint_type == 'expand_ratio':
            self.__dict__['_expand_include_list'] = include_list.copy()
        elif constraint_type == 'kernel_size':
            self.__dict__['_ks_include_list'] = include_list.copy()
        elif constraint_type == 'width_mult':
            self.__dict__['_widthMult_include_list'] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__['_depth_include_list'] = None
        self.__dict__['_expand_include_list'] = None
        self.__dict__['_ks_include_list'] = None
        self.__dict__['_widthMult_include_list'] = None

    def sample_active_subnet(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample width_mult
        width_mult_setting = None

        # sample kernel size
        ks_setting = []
        if not isinstance(ks_candidates[0], list):
            ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
        for k_set in ks_candidates:
            k = random.choice(k_set)
            ks_setting.append(k)

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.blocks) - 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)

        # sample depth
        depth_setting = []
        if not isinstance(depth_candidates[0], list):
            depth_candidates = [depth_candidates for _ in range(len(self.block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(width_mult_setting, ks_setting, expand_setting, depth_setting)

        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    def get_active_subnet(self, preserve_weight=True):
        first_conv = copy.deepcopy(self.first_conv)
        blocks = [copy.deepcopy(self.blocks[0])]

        final_expand_layer = copy.deepcopy(self.final_expand_layer)
        # feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
        # classifier = copy.deepcopy(self.classifier)
        attentions = copy.deepcopy(self.attentions)
        bap = copy.deepcopy(self.bap)
        fc_conv = copy.deepcopy(self.fc_conv)


        input_channel = blocks[0].mobile_inverted_conv.out_channels
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(MobileInvertedResidualBlock(
                    self.blocks[idx].mobile_inverted_conv.get_active_subnet(input_channel, preserve_weight),
                    copy.deepcopy(self.blocks[idx].shortcut)
                ))
                input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
            blocks += stage_blocks

        _subnet = MobileNetV3WS(first_conv, blocks, final_expand_layer, attentions, bap, fc_conv)
        _subnet.set_bn_param(**self.get_bn_param())
        return _subnet

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks[1:]:
            block.mobile_inverted_conv.re_organize_middle_weights(expand_ratio_stage)
