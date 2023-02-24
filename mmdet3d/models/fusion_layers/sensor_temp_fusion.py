import torch
import numpy as np
from mmcv.cnn import ConvModule, xavier_init
from torch import nn as nn
from torch.nn import functional as F

from ..registry import FUSION_LAYERS
from .query_initialization import QueryInit, ImgQueryInit
from mmdet3d.models.model_utils.actr import build as build_actr
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential
    from mmdet3d.ops import spconv as spconv

from .stf_transformer import STFTR

MODEL_DIR = {'STFTR': STFTR}

@FUSION_LAYERS.register_module()
class STF_v1(nn.Module):

    def __init__(self,
                 transformer,
                 point_cloud_range,
                 voxel_size,
                 sparse_shape,
                 query_init_cfg,
                 stf_cfg):
        super(STF_v1, self).__init__()

        self.query_init = QueryInit(query_init_cfg, kernel_size=3)
        
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.sparse_shape = sparse_shape
        self.pts_info = dict(
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            sparse_shape=self.sparse_shape
        )

        self.stf = MODEL_DIR[transformer](stf_cfg)

    def split_empty(self, pts_query):
        mask_empty_voxel = (pts_query.features.sum(1) == 0)
        empty_pts_query_feats = pts_query.features[mask_empty_voxel]
        non_empty_pts_query_feats = pts_query.features[~mask_empty_voxel]
        empty_pts_query_indices = pts_query.indices[mask_empty_voxel]
        non_empty_pts_query_indices = pts_query.indices[~mask_empty_voxel]

        empty_pts_query = spconv.SparseConvTensor(empty_pts_query_feats,
                                                  empty_pts_query_indices, 
                                                  pts_query.spatial_shape, 
                                                  pts_query.batch_size)
        non_empty_pts_query = spconv.SparseConvTensor(non_empty_pts_query_feats,
                                                      non_empty_pts_query_indices, 
                                                      pts_query.spatial_shape, 
                                                      pts_query.batch_size)
        
        return empty_pts_query, non_empty_pts_query
    
    def coor2pts(self, x, pad=0.0):
        self.ratio = self.sparse_shape[1] / x.spatial_shape[1]
        pts = (x.indices.to(torch.float) + pad) * torch.tensor(
            (self.voxel_size + [1])[::-1]).cuda() * self.ratio
        pts[:, 0] = pts[:, 0] / self.ratio - pad
        pts[:, 1:] += torch.tensor(self.point_cloud_range[:3][::-1]).cuda()
        pts[:, 1:] = pts[:, [3, 2, 1]]
        pts_list = []
        for i in range(pts[-1][0].int() + 1):
            pts_list.append(pts[pts[:, 0] == i][:, 1:])
        return pts_list

    def pts2coor(self, x, batch, pad=0.0):
        pts = x[:,[2,1,0]] # x,y,z -> z,y,x
        pts[:, 0:] -= torch.tensor(self.point_cloud_range[:3][::-1]).cuda()
        coor = pts / (self.ratio * torch.tensor(self.voxel_size[::-1]).cuda()) - pad
        coor = torch.round(coor).int()
        coor = torch.cat([torch.ones([coor.shape[0],1],dtype=torch.int32).cuda()*batch,coor],dim=1)
        return coor

    def ego_motion_compensation(self, voxel_feats_seq, img_metas_seq):
        new_indices = []
        seq_len = len(voxel_feats_seq)
        for idx in range(1, seq_len):
            pts_list = self.coor2pts(voxel_feats_seq[idx])
            batch_size = len(pts_list)
            coors=[]
            for batch in range(batch_size):
                rot = torch.tensor(img_metas_seq[idx][batch]['ego_rot']).cuda().float()
                trans = torch.tensor(img_metas_seq[idx][batch]['ego_trans']).cuda().float()
                pts = torch.matmul(pts_list[batch], rot.T) + trans
                coor = self.pts2coor(pts, batch)
                coors.append(coor)
            new_indices.append(torch.cat(coors))

        new_voxel_feats_seq = []
        for seq in range(seq_len):
            if seq==0:
                new_voxel_feats_seq.append(voxel_feats_seq[seq])
            else:
                voxel_feats = voxel_feats_seq[seq].features
                sparse_shape = torch.tensor(voxel_feats_seq[seq].spatial_shape).cuda()
                new_indice = new_indices[seq-1]
                mask = torch.cat([new_indice[:,1:]>torch.tensor(sparse_shape).cuda(),
                                  new_indice[:,1:]<torch.zeros(1,3).cuda()],dim=1)
                mask = torch.max(mask,dim=1)[0]==0
                
                voxel_feats = voxel_feats[mask]
                new_indice = new_indice[mask]
                
                new_voxel_feats_seq.append(spconv.SparseConvTensor(voxel_feats,new_indice,
                                                                   sparse_shape,
                                                                   batch_size))

        return voxel_feats_seq
    
    def forward(self, img_feats_seq, voxel_feats_seq, img_metas_seq, imgs_seq, gt_bboxes_3d):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.
        Returns:
            torch.Tensor: Fused features of each point.
        """
        # breakpoint()
        voxel_feats_seq = self.ego_motion_compensation(voxel_feats_seq, img_metas_seq)
        pts_query, loss_box_of_pts = self.query_init(voxel_feats_seq[0], gt_bboxes_3d)
        empty_pts_query, non_empty_pts_query = self.split_empty(pts_query)

        fuse_out = self.stf(pts_query, voxel_feats_seq, img_feats_seq, 
                            img_metas_seq, self.pts_info)
        # fuse_out = self.stf(empty_pts_query, non_empty_pts_query, voxel_feats_seq)

        return fuse_out
        
