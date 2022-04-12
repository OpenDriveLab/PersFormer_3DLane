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
import torch.nn.functional as F
from utils.utils import *
import copy
from typing import Optional, List
from torch import Tensor
from .PE import build_position_encoding

from models.ops.modules import MSDeformAttn

from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_


class Deform_Attn(nn.Module):

    def __init__(self, d_model=256, dim_ff=2048, nhead=8, num_layers=1, dropout=0.1, activation='relu', npoints=4):
        super().__init__()
        # self.projection_layer = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.projection_layers = _get_clones(projection_layer, num_layers)
        self.projection_layer = MSDeformAttn(d_model=d_model, n_levels=1, n_heads=nhead, n_points=npoints)
        self.num_layers = num_layers
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    #     self.ref_pnts_gen = nn.Linear(d_model, 2)

    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     xavier_uniform_(self.ref_pnts_gen.weight.data, gain=1.0)
    #     constant_(self.ref_pnts_gen.bias.data, 0.)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    # tgt_mask & tgt_key_padding_mask is used to tgt self attention, and can be added later
    # tgt size should be bev feature size
    def forward(self, tgt, uv_feature,
                _reference_points,
                _input_spatial_shapes,
                _input_level_start_index,
                # tgt_mask: Optional[Tensor] = None,
                # uv_feature_mask: Optional[Tensor] = None,
                # tgt_key_padding_mask: Optional[Tensor] = None,
                # uv_feature_key_padding_mask: Optional[Tensor] = None,
                # pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # k = self.with_pos_embed(uv_feature, pos)
        q = self.with_pos_embed(tgt, query_pos)
        v = uv_feature
        # ref_pnts = self.ref_pnts_gen(query_pos).sigmoid().unsqueeze(-2)
        ref_pnts = _reference_points
        # tgt2 = self.projection_layer(query=q, key=k, value=v, 
        #                                 attn_mask=uv_feature_mask,
        #                                 key_padding_mask=uv_feature_key_padding_mask)[0]
        # print("query size: ", str(q.shape))
        # print("ref_pnts size: ", str(ref_pnts))
        # print("input_flatten size: ", str(v))
        # print("input_spatial_shapes size: ", str(_input_spatial_shapes))
        # print("input_level_start_index size: ", str(_input_level_start_index))
        tgt2 = self.projection_layer(query=q, reference_points=ref_pnts, input_flatten=v, 
                                        input_spatial_shapes=_input_spatial_shapes,
                                        input_level_start_index=_input_level_start_index,
                                        input_padding_mask=None)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _empty_fn():
    return None


# unit test
if __name__ == '__main__':
    global args
    parser = define_args()
    args = parser.parse_args()

    args.hidden_dim = 256
    args.position_embedding = "sine"
    pos_embed = build_position_encoding(args=args)
