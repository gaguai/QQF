import torch
import numpy as np
import copy
from mmcv.cnn import ConvModule, xavier_init, build_conv_layer
from torch import nn as nn
from torch.nn import functional as F

from ..registry import FUSION_LAYERS
from .query_initialization import QueryInit, ImgQueryInit
from .stf_transformer_utils import knn_sampling, img_feats_sampling
from mmdet3d.models.model_utils.actr import build as build_actr
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.ops import knn, make_sparse_convmodule

if IS_SPCONV2_AVAILABLE:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential
    from mmdet3d.ops import spconv as spconv

class STFTRLayer(nn.Module):
    def __init__(self, stf_cfg):
        super().__init__()
        self.seq_len = stf_cfg['seq_len']
        self.pts_channels = stf_cfg['v_num_channels']
        self.dropout = stf_cfg['dropout']
        img_attn_cfg = stf_cfg['img_attn_cfg']
        self.img_channels = img_attn_cfg['img_channels']
        self.hidden_channels = img_attn_cfg['hidden_channels']
        self.dropout_i = img_attn_cfg['dropout_i']

        temp_attn = nn.MultiheadAttention(self.pts_channels, 1, dropout=self.dropout,
                                          kdim=self.pts_channels, vdim=self.pts_channels,
                                          batch_first=True)
        self.temp_attn = _get_clones(temp_attn, self.seq_len)
        self.cross_sensor_attn = nn.MultiheadAttention(self.hidden_channels, 1, 
                                                       dropout=self.dropout_i, 
                                                       kdim=self.hidden_channels, 
                                                       vdim=self.hidden_channels,
                                                       batch_first=True)
    
    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, 
                batch_size, 
                pts_query,
                pts_query_pe, 
                knn_feat_seq, 
                knn_indices_seq, 
                knn_pe_seq,
                img_feats,
                pts_info, 
                img_samples_num,
                img_metas):
        pts_query_feats = pts_query.features
        pts_query_indices = pts_query.indices
        pts_query_feats = self.with_pos_embed(pts_query_feats, pts_query_pe)
        mask_all, sampled_feat_all, xy_cams_all = img_feats_sampling(img_feats, pts_query, 
                                                img_metas, pts_info, img_samples_num)
        
        attn_out, attn_out_indices = [], []
        for b in range(batch_size):
            query = pts_query_feats[pts_query_indices[:,0]==b, :]
            for seq in range(self.seq_len):
                key = value = self.with_pos_embed(knn_feat_seq[seq][b].permute(1,0,2),
                                                  knn_pe_seq[seq][b].permute(1,0,2))
                if seq == 0:
                    out = self.temp_attn[seq](query.unsqueeze(1), key, value)[0]
                else:
                    out += self.temp_attn[seq](query.unsqueeze(1), key, value)[0]

            valid = mask_all[b][...,0].sum(dim=1) > 0
            key_i = sampled_feat_all[b][valid]
            value_i = sampled_feat_all[b][valid]
            query_i = out[valid]
            attn_mask_i = ~mask_all[b][valid].permute(0,2,1)

            breakpoint()
            cs_out = self.cross_sensor_attn(query_i, key_i, value_i, attn_mask=attn_mask_i)[0]
            cs_out = cs_out.squeeze()
            cs_out_indices = pts_query_indices[pts_query_indices[:,0]==b, :][valid]
            attn_out.append(cs_out)
            attn_out_indices.append(cs_out_indices)
        
        new_voxel_feats = spconv.SparseConvTensor(torch.cat(attn_out, 0), 
                                                  torch.cat(attn_out_indices, 0), 
                                                  pts_query.spatial_shape, 
                                                  pts_query.batch_size)
        
        return new_voxel_feats

class STFTR(nn.Module):

    def __init__(self,
                 stf_cfg):
        super(STFTR, self).__init__()

        self.knn_samples_num = stf_cfg['knn_samples_num']
        self.img_samples_num = stf_cfg['img_samples_num']
        self.num_layers = stf_cfg['num_layers']
        self.pts_channels = stf_cfg['v_num_channels']
        self.img_channels = stf_cfg['img_attn_cfg']['img_channels']
        self.hidden_channels = stf_cfg['img_attn_cfg']['hidden_channels']

        self.position_encoder = nn.Sequential(
            nn.Linear(3, int(stf_cfg['v_num_channels']/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(stf_cfg['v_num_channels']/2), stf_cfg['v_num_channels']),
            nn.ReLU(inplace=True)
        )

        self.shared_conv_img = build_conv_layer(
            dict(type='Conv2d'),
            self.img_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            bias='auto'
        )
        self.shared_conv_pts = make_sparse_convmodule(
            self.pts_channels,
            self.hidden_channels,
            kernel_size=3,
            norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
            padding=1,
            indice_key='subm',
            conv_type='SubMConv3d',
        )

        encoder_layer = STFTRLayer(stf_cfg)
        self.layer = _get_clones(encoder_layer, self.num_layers)
    
    def make_position_embedding(self, pts_feats):
        pts_indices = pts_feats.indices
        spatial_shape = torch.tensor(pts_feats.spatial_shape, device=pts_indices.device)
        norm_pts_indices = pts_indices[:,1:] / spatial_shape
        position_embedding = self.position_encoder(norm_pts_indices)

        return position_embedding
    
    def forward(self, pts_query, voxel_feats_seq, img_feats_seq, img_metas_seq, pts_info):
        batch_size = pts_query.batch_size.item()
        pts_query_pe = self.make_position_embedding(pts_query)

        knn_feat_seq, knn_indices_seq, knn_pe_seq = [], [], []
        for voxel_feats in voxel_feats_seq:
            knn_feat, knn_indices, knn_pe = knn_sampling(pts_query, voxel_feats, batch_size,
                                                         self.knn_samples_num,
                                                         self.position_encoder)

            knn_feat_seq.append(knn_feat)
            knn_indices_seq.append(knn_indices)
            knn_pe_seq.append(knn_pe)
        
        breakpoint()
        img_feats = self.shared_conv_img(img_feats_seq[0][0])
        sp_voxel_feats = self.shared_conv_pts(pts_query)
        
        # sampled_feat, mask, xy_cams_all = img_feats_sampling(img_feats, sp_voxel_feats, 
        #                                         img_metas_seq[0], pts_info, self.img_samples_num)
        for i in range(self.num_layers):
            sp_voxel_feats = self.layer[i](batch_size,
                                      sp_voxel_feats,
                                      pts_query_pe, 
                                      knn_feat_seq, 
                                      knn_indices_seq, 
                                      knn_pe_seq,
                                      img_feats,
                                      pts_info, 
                                      self.img_samples_num,
                                      img_metas_seq[0])

        return fuse_out
    
    
#     def forward(self, empty_pts_query, non_empty_pts_query, voxel_feats_seq):
#         batch_size = non_empty_pts_query.batch_size.item()
#         empty_pts_query_pe = self.make_position_embedding(empty_pts_query)
#         non_empty_pts_query_pe = self.make_position_embedding(non_empty_pts_query)
# 
#         empty_knn_feat_seq, empty_knn_indices_seq, empty_pe_seq = [], [], []
#         non_empty_knn_feat_seq, non_empty_knn_indices_seq, non_empty_pe_seq = [], [], []
#         for voxel_feats in voxel_feats_seq:
#             empty_knn_feat, empty_knn_indices, empty_pe = self.knn_sampling(
#                 empty_pts_query, voxel_feats, batch_size)
#             non_empty_knn_feat, non_empty_knn_indices, non_empty_pe = self.knn_sampling(
#                 non_empty_pts_query, voxel_feats, batch_size)
#             
#             empty_knn_feat_seq.append(empty_knn_feat)
#             empty_knn_indices_seq.append(empty_knn_indices)
#             empty_pe_seq.append(empty_pe)
# 
#             non_empty_knn_feat_seq.append(non_empty_knn_feat)
#             non_empty_knn_indices_seq.append(non_empty_knn_indices)
#             non_empty_pe_seq.append(non_empty_pe)
#         
#         breakpoint()
#         
#         return fuse_out
        
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])