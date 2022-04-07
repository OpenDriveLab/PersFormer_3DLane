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

import torch.nn as nn

def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# one block: Conv2d + (BN) + ReLu
def make_one_layer(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False, inplace=True):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    if batch_norm:
        layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace)]
    else:
        layers = [conv2d, nn.ReLU(inplace)]
    return layers