import torch
import numpy as np
from mmcv.cnn import ConvModule, xavier_init
from torch import nn as nn
from torch.nn import functional as F

from ..registry import FUSION_LAYERS
from .query_initialization import QueryInit, ImgQueryInit
from .img_cross_attn import ImgCrossAttn
from mmdet3d.models.model_utils.actr import build as build_actr
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential
    from mmdet3d.ops import spconv as spconv

@FUSION_LAYERS.register_module()
class STF_v1(nn.Module):

    def __init__(self,
                 transformer,
                 point_cloud_range,
                 voxel_size,
                 sparse_shape,
                 query_init_cfg,
                 img_query_init_cfg,
                 pfat_cfg,
                 init_cfg=None,
                 lt_cfg=None,
                 coord_type='LIDAR',
                 activate_out=False,
                 data_version='v1.0-trainval',
                 data_root='./data/nuscenes'):
        super(STF_v1, self).__init__()

        self.query_init = QueryInit(query_init_cfg, kernel_size=3)
        self.img_query_init = ImgQueryInit(img_query_init_cfg)

        # self.fusion_method = pfat_cfg['fusion_method']
        # self.actr = build_actr(model_cfg=pfat_cfg, model_name=transformer, lt_cfg=lt_cfg)
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.sparse_shape = sparse_shape
        self.coord_type = coord_type
        self.activate_out = activate_out

        self.data_version = data_version
        self.data_root = data_root
        print('sensor temporal fusion')
        self.nusc = None

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
        breakpoint()
        voxel_feats_seq = self.ego_motion_compensation(voxel_feats_seq, img_metas_seq)
        pts_query, loss_box_of_pts = self.query_init(voxel_feats_seq[0], gt_bboxes_3d)
        empty_pts_query, non_empty_prs_query = self.split_empty
        # img_query = self.img_query_init(img_feats_seq, pts_query, img_metas_seq, imgs_seq)
        # self.actr.max_num_ne_voxel = max(num_points_seq[0])

        return fuse_out

@FUSION_LAYERS.register_module()
class STF_v2(nn.Module):

    def __init__(self,
                 transformer,
                 point_cloud_range,
                 voxel_size,
                 sparse_shape,
                 #######################
                 img_cross_attn_cfg,
                 #######################
                 query_init_cfg,
                 img_query_init_cfg,
                 pfat_cfg,
                 init_cfg=None,
                 lt_cfg=None,
                 coord_type='LIDAR',
                 activate_out=False,
                 data_version='v1.0-trainval',
                 data_root='./data/nuscenes'):
        super(STF_v2, self).__init__()

        self.query_init = QueryInit(query_init_cfg, kernel_size=3)
        self.img_query_init = ImgQueryInit(img_query_init_cfg)
        ##########################
        self.img_cross_attn = ImgCrossAttn(img_cross_attn_cfg, point_cloud_range, 
                                           voxel_size, sparse_shape)
        ##########################
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.sparse_shape = sparse_shape

        # self.fusion_method = pfat_cfg['fusion_method']
        # self.actr = build_actr(model_cfg=pfat_cfg, model_name=transformer, lt_cfg=lt_cfg)
        self.coord_type = coord_type
        self.activate_out = activate_out

        self.data_version = data_version
        self.data_root = data_root
        print('sensor temporal fusion')
        self.nusc = None
    
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
        
        # compensation 후 범위 밖의 indices 처리
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
        voxel_feats_seq = self.ego_motion_compensation(voxel_feats_seq, img_metas_seq)
        ##########################
        voxel_feats_seq = self.img_cross_attn(voxel_feats_seq, img_feats_seq, img_metas_seq)
        ##########################
        pts_query, loss_box_of_pts = self.query_init(voxel_feats_seq[0], gt_bboxes_3d)
        # img_query = self.img_query_init(img_feats_seq, pts_query, img_metas_seq, imgs_seq)
        # self.actr.max_num_ne_voxel = max(num_points_seq[0])

        return fuse_out
        
