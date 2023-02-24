import torch
import numpy as np
import copy
from mmcv.cnn import ConvModule, xavier_init
from torch import nn as nn
from torch.nn import functional as F

from ..registry import FUSION_LAYERS
from .query_initialization import QueryInit, ImgQueryInit
from mmdet3d.models.model_utils.actr import build as build_actr
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.ops import knn

if IS_SPCONV2_AVAILABLE:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential
    from mmdet3d.ops import spconv as spconv

class TemporalAttn(nn.Module):
    def __init__(self, temp_attn_cfg):
        super().__init__()
        self.pts_channels = temp_attn_cfg['pts_channels']
        self.dropout = temp_attn_cfg['dropout']

        self.temp_attn = nn.MultiheadAttention(self.pts_channels, 1, dropout=dropout,
                                                kdim=self.pts_channels, vdim=self.pts_channels,
                                                batch_first=True)

    def forward(self, 
                batch_size, 
                pts_query,
                pts_query_pe, 
                knn_feat_seq, 
                knn_indices_seq, 
                knn_pe_seq):
        pts_query_feats = pts_query.features
        pts_query_indices = pts_query.indices

        pts_query_feats = self.with_pos(pts_query_feats.unsqueeze(1), pts_query_pe)
        breakpoint()
        
        return None