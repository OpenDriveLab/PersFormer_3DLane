# ==============================================================================
# Copyright (c) 2022 The PersFormer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import script
import torchvision.models as models
import geffnet

#__all__ = ['mish','Mish']

class WSConv2d(nn.Conv2d):
    def __init___(self, in_channels, out_channels, kernel_size, stride=1, 
        padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1,1,1,1) + 1e-5
        #std = torch.sqrt(torch.var(weight.view(weight.size(0),-1),dim=1)+1e-12).view(-1,1,1,1)+1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv_ws(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return WSConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

'''
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    def forward(self, x):
        return x*torch.tanh(F.softplus(x))
'''

@script
def _mish_jit_fwd(x): return x.mul(torch.tanh(F.softplus(x)))

@script
def _mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))

class MishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _mish_jit_bwd(x, grad_output)

# Cell
def mish(x): return MishJitAutoFn.apply(x)

class Mish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()

    def forward(self, x):
        return MishJitAutoFn.apply(x)
######################################################################################################################
######################################################################################################################

# pre-activation based upsampling conv block
class upConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, norm, act, num_groups):
        super(upConvLayer, self).__init__()
        conv = conv_ws
        if act == 'ELU':
            act = nn.ELU()
        elif act == 'Mish':
            act = Mish()
        else:
            act = nn.ReLU(True)
        self.conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if norm == 'GN':
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        else:
            self.norm = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = act
        self.scale_factor = scale_factor
    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)     #pre-activation
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        x = self.conv(x)
        return x

# pre-activation based conv block
class myConv(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride=1, 
                    padding=0, dilation=1, bias=True, norm='GN', act='ELU', num_groups=32):
        super(myConv, self).__init__()
        conv = conv_ws
        if act == 'ELU':
            act = nn.ELU()
        elif act == 'Mish':
            act = Mish()
        else:
            act = nn.ReLU(True)
        module = []
        if norm == 'GN': 
            module.append(nn.GroupNorm(num_groups=num_groups, num_channels=in_ch))
        else:
            module.append(nn.BatchNorm2d(in_ch, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))
        module.append(act)
        module.append(conv(in_ch, out_ch, kernel_size=kSize, stride=stride, 
                            padding=padding, dilation=dilation, groups=1, bias=bias))
        self.module = nn.Sequential(*module)
    def forward(self, x):
        out = self.module(x)
        return out

# Deep Feature Fxtractor
class deepFeatureExtractor_ResNext101(nn.Module):
    def __init__(self, lv6 = False):
        super(deepFeatureExtractor_ResNext101, self).__init__()
        # after passing ReLU   : H/2  x W/2
        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        self.encoder = models.resnext101_32x8d(pretrained=True)
        self.fixList = ['layer1.0','layer1.1','.bn']
        self.lv6 = lv6

        if lv6 is True:
            self.layerList = ['relu','layer1','layer2','layer3', 'layer4']
            self.dimList = [64, 256, 512, 1024, 2048]
        else:
            del self.encoder.layer4
            del self.encoder.fc
            self.layerList = ['relu','layer1','layer2','layer3']
            self.dimList = [64, 256, 512, 1024]

        for name, parameters in self.encoder.named_parameters():
            if name == 'conv1.weight':
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False
        
    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            #feature = v(features[-1])
            #features.append(feature)
            if any(x in k for x in self.layerList):
                out_featList.append(feature)
        return out_featList
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_VGG19(nn.Module):
    def __init__(self, lv6 = False):
        super(deepFeatureExtractor_VGG19, self).__init__()
        self.lv6 = lv6
        # after passing 6th layer   : H/2  x W/2
        # after passing 13th layer   : H/4  x W/4
        # after passing 26th layer   : H/8  x W/8
        # after passing 39th layer   : H/16 x W/16
        # after passing 52th layer   : H/32 x W/32
        self.encoder = models.vgg19_bn(pretrained=True)
        del self.encoder.avgpool
        del self.encoder.classifier
        if lv6 is True:
            self.dimList = [64, 128, 256, 512, 512]
            self.layerList = [6, 13, 26, 39, 52]
        else:
            self.dimList = [64, 128, 256, 512]
            self.layerList = [6, 13, 26, 39]
            for i in range(13):
                del self.encoder.features[-1]
        '''
        self.fixList = ['.bn']
        for name, parameters in self.encoder.named_parameters():
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False
        '''
    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            if i in self.layerList:
                out_featList.append(feature)
        return out_featList
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_DenseNet161(nn.Module):
    def __init__(self, lv6 = False):
        super(deepFeatureExtractor_DenseNet161, self).__init__()
        self.encoder = models.densenet161(pretrained=True)
        self.lv6 = lv6
        del self.encoder.classifier
        del self.encoder.features.norm5
        if lv6 is True:
            self.dimList = [96, 192, 384, 1056, 2208]
        else:
            self.dimList = [96, 192, 384, 1056]
            del self.encoder.features.denseblock4

    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder.features._modules.items():
            if ('transition' in k):
                feature = v.norm(feature)
                feature = v.relu(feature)
                feature = v.conv(feature)
                out_featList.append(feature)
                feature = v.pool(feature)
            elif k == 'conv0':
                feature = v(feature)
                out_featList.append(feature)
            elif k == 'denseblock4' and (self.lv6 is True):
                feature = v(feature)
                out_featList.append(feature)
            else:
                feature = v(feature)
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_InceptionV3(nn.Module):
    def __init__(self, lv6 = False):
        super(deepFeatureExtractor_InceptionV3, self).__init__()
        self.encoder = models.inception_v3(pretrained=True)
        self.encoder.aux_logits = False
        self.lv6 = lv6
        del self.encoder.AuxLogits
        del self.encoder.fc
        if lv6 is True:
            self.layerList = ['Conv2d_2b_3x3','Conv2d_4a_3x3','Mixed_5d','Mixed_6e','Mixed_7c']
            self.dimList = [64, 192, 288, 768, 2048]
        else:
            self.layerList = ['Conv2d_2b_3x3','Conv2d_4a_3x3','Mixed_5d','Mixed_6e']
            self.dimList = [64, 192, 288, 768]
            del self.encoder.Mixed_7a
            del self.encoder.Mixed_7b
            del self.encoder.Mixed_7c

    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():
            feature = v(feature)
            if k in ['Conv2d_2b_3x3', 'Conv2d_ta_3x3']:
                feature = F.max_pool2d(feature, kernel_size=3, stride=2)
            if any(x in k for x in self.layerList):
                out_featList.append(feature)
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_MobileNetV2(nn.Module):
    def __init__(self):
        super(deepFeatureExtractor_MobileNetV2, self).__init__()
        # after passing 1th : H/2  x W/2
        # after passing 2th : H/4  x W/4
        # after passing 3th : H/8  x W/8
        # after passing 4th : H/16 x W/16
        # after passing 5th : H/32 x W/32
        self.encoder = models.mobilenet_v2(pretrained=True)
        del self.encoder.classifier
        self.layerList = [1, 3, 6, 13, 18]
        self.dimList = [16, 24, 32, 96, 960]
    def forward(self, x):
        out_featList = []
        feature = x
        for i in range(len(self.encoder.features)):
            feature = self.encoder.features[i](feature)
            if i in self.layerList:
                out_featList.append(feature)
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_ResNet101(nn.Module):
    def __init__(self, lv6 = False):
        super(deepFeatureExtractor_ResNet101, self).__init__()
        # after passing ReLU   : H/2  x W/2
        # after passing Layer1 : H/4  x W/4
        # after passing Layer2 : H/8  x W/8
        # after passing Layer3 : H/16 x W/16
        self.encoder = models.resnet101(pretrained=True)
        self.fixList = ['layer1.0','layer1.1','.bn']

        if lv6 is True:
            self.layerList = ['relu','layer1','layer2','layer3', 'layer4']
            self.dimList = [64, 256, 512, 1024,2048]
        else:
            del self.encoder.layer4
            del self.encoder.fc
            self.layerList = ['relu','layer1','layer2','layer3']
            self.dimList = [64, 256, 512, 1024]
        
        for name, parameters in self.encoder.named_parameters():
            if name == 'conv1.weight':
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False
        
    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            #feature = v(features[-1])
            #features.append(feature)
            if any(x in k for x in self.layerList):
                out_featList.append(feature) 
        return out_featList
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

class deepFeatureExtractor_EfficientNet(nn.Module):
    def __init__(self, architecture="EfficientNet-B5", lv6=False, lv5=False, lv4=False, lv3=False):
        super(deepFeatureExtractor_EfficientNet, self).__init__()
        assert architecture in ["EfficientNet-B0", "EfficientNet-B1", "EfficientNet-B2", "EfficientNet-B3", 
                                    "EfficientNet-B4", "EfficientNet-B5", "EfficientNet-B6", "EfficientNet-B7"]
        
        if architecture == "EfficientNet-B0":
            self.encoder = geffnet.tf_efficientnet_b0_ns(pretrained=True)
            self.dimList = [16, 24, 40, 112, 1280] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 40, 112, 320] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B1":
            self.encoder = geffnet.tf_efficientnet_b1_ns(pretrained=True)
            self.dimList = [16, 24, 40, 112, 1280] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 40, 112, 320] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B2":
            self.encoder = geffnet.tf_efficientnet_b2_ns(pretrained=True)
            self.dimList = [16, 24, 48, 120, 1408] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 48, 120, 352] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B3":
            self.encoder = geffnet.tf_efficientnet_b3_ns(pretrained=True)
            self.dimList = [24, 32, 48, 136, 1536] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 32, 48, 136, 384] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B4":
            self.encoder = geffnet.tf_efficientnet_b4_ns(pretrained=True)
            self.dimList = [24, 32, 56, 160, 1792] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 32, 56, 160, 448] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B5":
            self.encoder = geffnet.tf_efficientnet_b5_ns(pretrained=True)
            self.dimList = [24, 40, 64, 176, 2048] #5th feature is extracted after conv_head or bn2
            #self.dimList = [24, 40, 64, 176, 512] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B6":
            self.encoder = geffnet.tf_efficientnet_b6_ns(pretrained=True)
            self.dimList = [32, 40, 72, 200, 2304] #5th feature is extracted after conv_head or bn2
            #self.dimList = [32, 40, 72, 200, 576] #5th feature is extracted after blocks[6]
        elif architecture == "EfficientNet-B7":
            self.encoder = geffnet.tf_efficientnet_b7_ns(pretrained=True)
            self.dimList = [32, 48, 80, 224, 2560] #5th feature is extracted after conv_head or bn2
            #self.dimList = [32, 48, 80, 224, 640] #5th feature is extracted after blocks[6]
        del self.encoder.global_pool
        del self.encoder.classifier
        #self.block_idx = [3, 4, 5, 7, 9] #5th feature is extracted after blocks[6]
        #self.block_idx = [3, 4, 5, 7, 10] #5th feature is extracted after conv_head
        self.block_idx = [3, 4, 5, 7, 11] #5th feature is extracted after bn2
        if lv6 is False:
            del self.encoder.blocks[6]
            del self.encoder.conv_head
            del self.encoder.bn2
            del self.encoder.act2
            self.block_idx = self.block_idx[:4]
            self.dimList = self.dimList[:4]
        if lv5 is False:
            del self.encoder.blocks[5]
            self.block_idx = self.block_idx[:3]
            self.dimList = self.dimList[:3]
        if lv4 is False:
            del self.encoder.blocks[4]
            self.block_idx = self.block_idx[:2]
            self.dimList = self.dimList[:2]
        if lv3 is False:
            del self.encoder.blocks[3]
            self.block_idx = self.block_idx[:1]
            self.dimList = self.dimList[:1]
        # after passing blocks[3]    : H/2  x W/2
        # after passing blocks[4]    : H/4  x W/4
        # after passing blocks[5]    : H/8  x W/8
        # after passing blocks[7]    : H/16 x W/16
        # after passing conv_stem    : H/32 x W/32
        self.fixList = ['blocks.0.0','bn']

        for name, parameters in self.encoder.named_parameters():
            if name == 'conv_stem.weight':
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        block_cnt = 0
        for k, v in self.encoder._modules.items():
            if k == 'act2':
                break
            if k == 'blocks':
                for m, n in v._modules.items():
                    feature = n(feature)
                    try:
                        if self.block_idx[block_cnt] == cnt:
                            out_featList.append(feature)
                            block_cnt += 1
                            break
                        cnt += 1
                    except:
                        continue
            else:
                feature = v(feature)
                if self.block_idx[block_cnt] == cnt:
                    out_featList.append(feature)
                    block_cnt += 1
                    break
                cnt += 1            
            
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable