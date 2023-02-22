import torch
import numpy as np
import math
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import build_conv_layer
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.ops import make_sparse_convmodule
import copy

class ImgCrossAttn(nn.Module):
    def __init__(self, 
                 img_cross_attn_cfg,
                 point_cloud_range,
                 voxel_size,
                 sparse_shape): 
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.sparse_shape = sparse_shape
        self.ratio = None

        self.pts_channels = img_cross_attn_cfg['pts_channels'] # 128
        self.img_channels = img_cross_attn_cfg['img_channels'] # 256
        self.hidden_channels = img_cross_attn_cfg['hidden_channels'] # 128
        self.dropout = img_cross_attn_cfg['dropout']
        self.attn = nn.MultiheadAttention(self.hidden_channels, 1, dropout=self.dropout, 
                                          kdim=self.hidden_channels, vdim=self.hidden_channels, batch_first=True)

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
        self.bn_momentum = 0.1
        self.init_weights()

    def init_weights(self):
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
    
    def coor2pts(self, x, pad=0.0):
        pts = (x + pad) * torch.tensor(
            (self.voxel_size + [1])[::-1]).cuda() * self.ratio
        pts[..., 0] = pts[..., 0] / self.ratio - pad
        pts[..., 1:] += torch.tensor(self.point_cloud_range[:3][::-1]).cuda()
        pts[..., 1:] = pts[..., [3, 2, 1]]
        
        return pts
        
    def make_sample_grid(self, inds, batch_size):
        origin = copy.deepcopy(inds)

        batch_corners = []
        for b in range(batch_size):
            batch_inds = origin[origin[:,0]==b]
            batch_centers = batch_inds[:,1:]
            num_inds = batch_inds.shape[0]

            dims= np.ones((num_inds,3))*0.5
            ndim = int(dims.shape[1])
            corners_norm = np.stack(
                np.unravel_index(np.arange(2**ndim), [2] * ndim),
                axis=1).astype(dims.dtype)
            corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
            corners_norm = corners_norm - np.array(0.5, dtype=dims.dtype)
            corners = torch.from_numpy(dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
                [1, 2**ndim, ndim])).float().cuda()
            corners += batch_centers.view(-1,1,3)
            corners = torch.cat([torch.ones([corners.shape[0],8,1],dtype=torch.float).cuda()*b,corners],dim=2)
            batch_corners.append(corners)
        batch_corners = torch.cat(batch_corners)
        total_coors = torch.cat([inds.unsqueeze(1), batch_corners],dim=1)

        return total_coors

    def forward(self, voxel_feats_seq, img_feats_seq, img_metas_seq):
        # new_voxel_feats_seq = []
        # new_img_feats_seq = []
        # for i in range(len(img_metas_seq)):
        #     new_img_feats = self.shared_conv_img(img_feats_seq[i][0])
        #     new_voxel_feats = self.shared_conv_pts(voxel_feats_seq[i])
        #     new_img_feats_seq.append(new_img_feats)
        #     new_voxel_feats_seq.append(new_voxel_feats)
        
        # 현재 frame에서 attention
        img_feats = self.shared_conv_img(img_feats_seq[0][0])
        sp_voxel_feats = self.shared_conv_pts(voxel_feats_seq[0])
        img_metas = img_metas_seq[0]
        voxel_feats = sp_voxel_feats.features
        voxel_inds = sp_voxel_feats.indices
        if not self.ratio:
            self.ratio = self.sparse_shape[1] / sp_voxel_feats.spatial_shape[1]

        batch_size = len(img_metas)
        B, I_C, I_H, I_W = img_feats.shape
        img_feats = img_feats.view(batch_size, -1, I_C, I_H, I_W)

        sample_grid = self.make_sample_grid(voxel_inds, batch_size)
        sample_pts = self.coor2pts(sample_grid, pad=0.5) # N, 9, 4

        lidar2img = []
        batch_cnt = voxel_feats.new_zeros(batch_size).int()
        for b in range(batch_size):
            lidar2img.append(img_metas[b]['lidar2img'])
            batch_cnt[b] = (voxel_inds[:,0]==b).sum()
        lidar2img = np.asarray(lidar2img) # (B,6,4,4)
        lidar2img = voxel_feats.new_tensor(lidar2img)
        batch_bound = batch_cnt.cumsum(dim=0)
        cur_start = 0
        for b in range(batch_size):
            cur_end = batch_bound[b]
            voxel_feat = voxel_feats[cur_start:cur_end]
            voxel_ind = voxel_inds[cur_start:cur_end]
            pts = sample_pts[cur_start:cur_end]
            voxel_num_points = torch.ones(pts.shape[0]).cuda()*9
            proj_mat = lidar2img[b]
            num_cam = proj_mat.shape[0]
            num_voxels, max_points, p_dim = pts.shape # n, 9, 4
            num_pts = num_voxels * max_points 
            pts = pts.view(num_pts, p_dim)[:,1:] # n*9, 4
            voxel_pts = apply_3d_transformation(pts, 'LIDAR', img_metas[b], reverse=True).detach()
            voxel_pts = torch.cat((voxel_pts,torch.ones_like(voxel_pts[...,:1])),dim=-1).unsqueeze(0).unsqueeze(-1)
            proj_mat = proj_mat.unsqueeze(1)
            xyz_cams = torch.matmul(proj_mat, voxel_pts).squeeze(-1)
            eps = 1e-5
            mask = (xyz_cams[..., 2:3] > eps)
            xy_cams = xyz_cams[..., 0:2] / torch.maximum(
                xyz_cams[..., 2:3], torch.ones_like(xyz_cams[..., 2:3])*eps)
            img_shape = img_metas[b]['ori_shape']
            xy_cams[...,0] = xy_cams[...,0] / img_shape[1]
            xy_cams[...,1] = xy_cams[...,1] / img_shape[0]
            xy_cams = (xy_cams - 0.5) * 2
            mask = (mask & (xy_cams[..., 0:1] > -1.0) 
                 & (xy_cams[..., 0:1] < 1.0) 
                 & (xy_cams[..., 1:2] > -1.0) 
                 & (xy_cams[..., 1:2] < 1.0))
            mask = torch.nan_to_num(mask)
            # visualize
            if False: #len(pts_2d):
                import cv2
                xy_cams = (xy_cams/2) + 0.5
                xy_cams[...,0] = xy_cams[...,0] * img_shape[1]
                xy_cams[...,1] = xy_cams[...,1] * img_shape[0]
                xy_cams = xy_cams.int()
                for idx in range(num_cam):
                    image = cv2.imread(img_metas[b]['filename'][idx], cv2.COLOR_BGR2RGB)
                    for i in xy_cams[idx][mask[idx].squeeze()]:
                        x = i[0].item()
                        y = i[1].item()
                        image = cv2.circle(
                            image, (int(x), int(y)),
                            radius=1,
                            color=(0, 255, 0),
                            thickness=-1)
                    cv2.imwrite(f"./test.png", image)
                breakpoint()

            sampled_feat = F.grid_sample(img_feats[b],xy_cams.unsqueeze(-2)).squeeze(-1).permute(2,0,1)
            sampled_feat = sampled_feat.view(num_voxels, max_points, num_cam, self.hidden_channels)
            mask = mask.permute(1,0,2).view(num_voxels, max_points, num_cam, 1)

            mask_points = mask.new_zeros((mask.shape[0],mask.shape[1]+1))
            mask_points[torch.arange(mask.shape[0],device=mask_points.device).long(),voxel_num_points.long()] = 1
            mask_points = mask_points.cumsum(dim=1).bool()
            mask_points = ~mask_points
            mask = mask_points[:,:-1].unsqueeze(-1).unsqueeze(-1) & mask
            
            mask = mask.reshape(num_voxels,max_points*num_cam,1)
            sampled_feat = sampled_feat.reshape(num_voxels,max_points*num_cam,self.hidden_channels)
            K = sampled_feat
            V = sampled_feat
            Q = voxel_feat.unsqueeze(1) # B C H W
            valid = mask[...,0].sum(dim=1) > 0 
            # attn_output = voxel_feat.new_zeros(num_voxels, 1, self.pts_channels)
            voxel_feat[valid] = self.attn(Q[valid],K[valid],V[valid],attn_mask=(~mask[valid]).permute(0,2,1))[0].squeeze()
            voxel_feats[cur_start:cur_end] = voxel_feat
            cur_start = cur_end

        voxel_feats_seq[0] = voxel_feats_seq[0].replace_feature(voxel_feats)

        return voxel_feats_seq
        
