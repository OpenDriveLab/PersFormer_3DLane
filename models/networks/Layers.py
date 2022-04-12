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

import torch.optim
import torch.nn as nn
from utils.utils import *
from typing import Optional, List

from models.ops.modules import MSDeformAttn, IdentityMSDeformAttn, DropoutMSDeformAttn

from .Deform_ATTN import _get_activation_fn, _get_clones

from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_


class FFN(nn.Module):

    def __init__(self,
                d_model=256,
                dim_ff=1024,
                activation='relu',
                ffn_dropout=0.,
                add_identity=True):
        super().__init__()

        self.d_model = d_model
        self.feedforward_channels = dim_ff

        self.linear1 = nn.Linear(d_model, dim_ff)
        self.activation = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(ffn_dropout)

        self.linear2 = nn.Linear(dim_ff, d_model)
        self.dropout2 = nn.Dropout(ffn_dropout)
        self.add_identity = add_identity
        self._reset_parameters()


    def _reset_parameters(self):
        xavier_uniform_(self.linear1.weight.data)
        constant_(self.linear1.bias.data, 0.)
        xavier_uniform_(self.linear2.weight.data)
        constant_(self.linear2.bias.data, 0.)


    def forward(self, x, identity=None):
        inter = self.linear2(self.dropout1(self.activation(self.linear1(x))))
        out = self.dropout2(inter)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


class EncoderLayer(nn.Module):

    '''
        one layer in Encoder,
        self-attn -> norm -> cross-attn -> norm -> ffn -> norm

        INIT:   d_model: this is C in ms uv feat map & BEV feat map
                dim_ff: num channels in feed forward net (FFN)
                activation, ffn_dropout: used in FFN
                num_levels: num layers of fpn out
                num_points, num_heads: used in deform attn
    '''
    def __init__(self,
                 d_model=None,
                 dim_ff=None,
                 activation="relu",
                 ffn_dropout=0.0,
                 num_levels=4,
                 num_points=8,
                 num_heads=8):
        super().__init__()
        self.fp16_enabled = False

        self.self_attn = IdentityMSDeformAttn(d_model=d_model, n_levels=1)  # q=v,
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = DropoutMSDeformAttn(d_model=d_model, 
                                        n_levels=num_levels, 
                                        n_points=num_points, 
                                        n_heads=num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FFN(d_model=d_model, dim_ff=dim_ff, activation=activation,
                                     ffn_dropout=ffn_dropout)
        self.norm3 = nn.LayerNorm(d_model)

    '''
        INPUT:  query: (B, bev_h*bev_w, C), this is BEV feat map
                value: (B, \sum_{l=0}^{L-1} H_l \cdot W_l, C), this is ms uv feat map from FPN, C is fixed for all scale
                bev_pos: BEV feat map pos embed (B, bev_h*bev_w, C)
                ref_2d: ref pnts used in self-attn, for query (B, bev_h*bev_w, 1, 2)
                ref_3d: ref pnts used in cross-attn, for ms uv feat map from FPN, this is IMPORTANT for uv-bev transform
                        (B, bev_h*bev_w, 4, 2)
                bev_h: height of bev feat map
                bev_w: widght of bev feat map
                spatial_shapes: input spatial shapes for cross-attn, this is used to split ms uv feat map
                level_start_index: input level start index for cross-attn, this is used to split ms uv feat map
                
            self-attn:
                input: q=v=query, ref_pnts = ref_2d (universal sampling over query space), 1-lvl
                output: query for cross-attn
            
            cross-attn:
                input: q=query, v=value=ms_uv_feat_map, ref_pnts = ref_3d (this is projection from bev loc to uv loc, 
                                                                            so that attention of each bev loc 
                                                                            can focus on relative uv loc)
                output: bev feat map
    '''
    def forward(self,
                query=None,
                value=None,
                bev_pos=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                spatial_shapes=None,
                level_start_index=None):
        
        # self attention
        identity = query

        temp_key = temp_value = query
        query = self.self_attn( query + bev_pos,
                                reference_points=ref_2d,
                                input_flatten=temp_value,
                                input_spatial_shapes=torch.tensor(
                                    [[bev_h, bev_w]], device=query.device),
                                input_level_start_index=torch.tensor(
                                    [0], device=query.device),
                                identity=identity)
        identity = query

        # norm 1
        query = self.norm1(query)

        # cross attention
        query = self.cross_attn(query,
                                reference_points=ref_3d,
                                input_flatten=value,
                                input_spatial_shapes=spatial_shapes,
                                input_level_start_index=level_start_index)
        query = query + identity

        # norm 2
        query = self.norm2(query)

        # ffn
        query = self.ffn(query)

        # norm 3
        query = self.norm3(query)

        return query


class DecoderLayer(nn.Module):

    '''
        one layer in Decoder,
        self-attn -> norm -> cross-attn -> norm -> ffn -> norm

        INIT:   d_model: this is C in ms uv feat map & BEV feat map
                dim_ff: num channels in feed forward net (FFN)
                activation, ffn_dropout: used in FFN
                num_points, num_heads: used in deform attn
    '''
    def __init__(self,
                 d_model=None,
                 dim_ff=None,
                 activation="relu",
                 ffn_dropout=0.0,
                #  num_levels=4,
                 num_points=8,
                 num_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = DropoutMSDeformAttn(d_model=d_model, n_levels=1,
                                                n_points=num_points, n_heads=num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model=d_model, dim_ff=dim_ff, activation=activation,
                                     ffn_dropout=ffn_dropout)
        self.norm3 = nn.LayerNorm(d_model)

    
    '''
        INPUT:  query: (B, num_query, C), this is shrinked BEV feat map
                value: (B, bev_h*bev_w, C), this is bev feat map from encoder
                query_pos: (B, num_query, C), shrinked BEV feat map pos embed
                ref_2d: (B, num_query, 1, 2) generated from query_pos
                bev_h: height of bev feat map
                bev_w: widght of bev feat map

            self-attn:
                input: q=k=v=query
                output: query for cross-attn
            
            cross-attn:
                input: q=query, v=value=bev_feat_map, ref_pnts = ref_2d
                output: shrinked bev feat map
    '''
    def forward(self,
            query=None,
            value=None,
            query_pos=None,
            ref_2d=None,
            bev_h=None,
            bev_w=None):
        identity = query

        temp_key = temp_value = query
        query = self.self_attn(query=query + query_pos, key=temp_key, value=temp_value)[0]
        query = query + identity

        identity = query

        query = self.norm1(query)

        query = self.cross_attn(query,
                                reference_points=ref_2d,
                                input_flatten=value,
                                input_spatial_shapes=torch.tensor(
                                    [[bev_h, bev_w]], device=query.device),
                                input_level_start_index=torch.tensor(
                                    [0], device=query.device))
        query = query + identity

        query = self.norm2(query)

        query = self.ffn(query)

        query = self.norm3(query)

        return query
        
        
