import torch
import numpy as np
import copy
import cv2
from mmcv.cnn import ConvModule, xavier_init
from torch import nn as nn
from torch.nn import functional as F

from ..registry import FUSION_LAYERS
from .query_initialization import QueryInit, ImgQueryInit
from mmdet3d.models.model_utils.actr import build as build_actr
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.ops import knn

if IS_SPCONV2_AVAILABLE:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential
    from mmdet3d.ops import spconv as spconv

def knn_sampling(pts_query, voxel_feats, batch_size, knn_samples_num, position_encoder):
    pts_query_indices = pts_query.indices
    pts_query_feats = pts_query.features
    voxel_feats_indices = voxel_feats.indices
    voxel_feats_feats = voxel_feats.features
    spatial_shape = torch.tensor(voxel_feats.spatial_shape, 
                                    device=voxel_feats_indices.device)

    sampled_voxel_feats_list = []
    sampled_voxel_indices_list = []
    sampled_voxel_pe_list = []
    for b in range(batch_size):
        pts_indices = pts_query_indices[pts_query_indices[:,0]==b, 1:]
        pts_indices = pts_indices.view(1,-1,3).float()
        voxel_indices = voxel_feats_indices[voxel_feats_indices[:,0]==b, 1:]
        voxel_indices = voxel_indices.view(1,-1,3).float()
        idx = knn(knn_samples_num, voxel_indices, pts_indices)

        sampled_voxel_feats = voxel_feats_feats[voxel_feats_indices[:,0]==b, :][idx,:]
        sampled_voxel_feats_list.append(sampled_voxel_feats.squeeze(0))

        sampled_voxel_indices = voxel_feats_indices[voxel_feats_indices[:,0]==b, :][idx,:]
        sampled_voxel_indices_list.append(sampled_voxel_indices.squeeze(0))

        norm_sampled_voxel_indices = sampled_voxel_indices[:,:,:,1:] / spatial_shape
        position_embedding = position_encoder(norm_sampled_voxel_indices)
        sampled_voxel_pe_list.append(position_embedding.squeeze(0))

    return sampled_voxel_feats_list, sampled_voxel_indices_list, sampled_voxel_pe_list

def img_feats_sampling(img_feats, sp_voxel_feats, img_metas, pts_info, sample_num):
    sparse_shape = pts_info['sparse_shape']
    voxel_size = pts_info['voxel_size']
    point_cloud_range = pts_info['point_cloud_range']
    voxel_feats = sp_voxel_feats.features
    voxel_inds = sp_voxel_feats.indices
    ratio = sparse_shape[1] / sp_voxel_feats.spatial_shape[1]

    batch_size = len(img_metas)
    B, I_C, I_H, I_W = img_feats.shape
    img_feats = img_feats.view(batch_size, -1, I_C, I_H, I_W)

    sample_grid = make_sample_grid(voxel_inds, batch_size)
    sample_pts = coor2pts(sample_grid, 0.5, ratio, voxel_size, point_cloud_range)

    lidar2img = []
    batch_cnt = voxel_feats.new_zeros(batch_size).int()
    for b in range(batch_size):
        lidar2img.append(img_metas[b]['lidar2img'])
        batch_cnt[b] = (voxel_inds[:,0]==b).sum()
    lidar2img = np.asarray(lidar2img)
    lidar2img = voxel_feats.new_tensor(lidar2img)
    batch_bound = batch_cnt.cumsum(dim=0)

    cur_start = 0
    mask_all, sampled_feat_all, xy_cams_all = [], [], []
    for b in range(batch_size):
        cur_end = batch_bound[b]
        pts = sample_pts[cur_start:cur_end]
        voxel_num_points = torch.ones(pts.shape[0]).cuda()*sample_num
        proj_mat = lidar2img[b]
        num_cam = proj_mat.shape[0]
        num_voxels, max_points, p_dim = pts.shape # n, 9, 4
        num_pts = num_voxels * max_points 
        pts = pts.view(num_pts, p_dim)[:,1:]
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
        xy_cams_all.append(xy_cams)
        # breakpoint()
        mask = (mask & (xy_cams[..., 0:1] > -1.0) 
                & (xy_cams[..., 0:1] < 1.0) 
                & (xy_cams[..., 1:2] > -1.0) 
                & (xy_cams[..., 1:2] < 1.0))
        mask = torch.nan_to_num(mask)

        sampled_feat = F.grid_sample(img_feats[b],xy_cams.unsqueeze(-2)).squeeze(-1).permute(2,0,1)
        sampled_feat = sampled_feat.view(num_voxels, max_points, num_cam, I_C)
        mask = mask.permute(1,0,2).view(num_voxels, max_points, num_cam, 1)

        mask_points = mask.new_zeros((mask.shape[0],mask.shape[1]+1))
        mask_points[torch.arange(mask.shape[0],device=mask_points.device).long(),voxel_num_points.long()] = 1
        mask_points = mask_points.cumsum(dim=1).bool()
        mask_points = ~mask_points
        mask = mask_points[:,:-1].unsqueeze(-1).unsqueeze(-1) & mask

        mask = mask.reshape(num_voxels,max_points*num_cam,1)
        mask_all.append(mask)
        sampled_feat = sampled_feat.reshape(num_voxels,max_points*num_cam,I_C)
        sampled_feat_all.append(sampled_feat)
        cur_start = cur_end
        # breakpoint()

    return mask_all, sampled_feat_all, xy_cams_all

def make_sample_grid(inds, batch_size):
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

def coor2pts(x, pad, ratio, voxel_size, point_cloud_range):
    pts = (x + pad) * torch.tensor(
        (voxel_size + [1])[::-1]).cuda() * ratio
    pts[..., 0] = pts[..., 0] / ratio - pad
    pts[..., 1:] += torch.tensor(point_cloud_range[:3][::-1]).cuda()
    pts[..., 1:] = pts[..., [3, 2, 1]]
    
    return pts

class TemporalGatedFusion():
    def __init__(self, img_channels, pts_channels, seq_len):
        self.img_channels = img_channels
        self.pts_channels = pts_channels
        self.seq_len = seq_len
    
    def forward(self, img_coor, img_feats, pts_feats):
        breakpoint()
        return None

    def pts2img(coor, pts_feat, shape, batch_dict, cam_key, _idx, img_feat):
        def visualize_img(batch_dict,cam_key):
            # pts_feat = pts_feat.detach().cpu().max(2)[0].numpy()
            # pts_feat = (pts_feat * 255.).astype(np.uint8)
            # cv2.imwrite('lidar2img.png', pts_feat)
            cv2.imwrite("aaa.jpg",batch_dict['images'][cam_key][0].detach().cpu().numpy()*255)

        def visualize_feat(img_feat):
            feat = img_feat.cpu().detach().numpy()
            min = feat.min()
            max = feat.max()
            image_features = (feat-min)/(max-min)
            image_features = (image_features*255)
            max_image_feature = np.max(np.transpose(image_features.astype("uint8"),(1,2,0)),axis=2)
            max_image_feature = cv2.applyColorMap(max_image_feature,cv2.COLORMAP_JET)
            cv2.imwrite("max_image_feature.jpg",max_image_feature)
            return max_image_feature
        def visualize_voxels(i_coor,max_img_feat):
            for coor in i_coor:
                cv2.circle(max_img_feat,(coor[1].item(),coor[0].item()), 1,(0,0,255))
            cv2.imwrite("pointed_feature.jpg",max_img_feat)
        def visualize_overlaped_voxels(i_coor, max_img_feat):
            overlaped_voxel_mask = torch.unique(i_coor,dim=0,return_counts=True)[1]>1
            voxel_after_remove = torch.unique(i_coor,dim=0,return_counts=True)[0]
            overlaped_voxel = voxel_after_remove[overlaped_voxel_mask]
            for coor in overlaped_voxel:
                cv2.circle(max_img_feat,(coor[1].item(),coor[0].item()), 1,(0,0,255))
            cv2.imwrite("overlaped_pointed_feature.jpg",max_img_feat)
        
        coor = coor[:, [1, 0]] #H,W
        i_shape = torch.cat(
            [shape + 1,
                torch.tensor([pts_feat.shape[1]]).cuda()])
        i_pts_feat = torch.zeros(tuple(i_shape), device=coor.device)
        #i_coor = (coor * shape).to(torch.long)
        i_coor = coor.to(torch.long)
        i_pts_feat[i_coor[:, 0], i_coor[:, 1]] = pts_feat
        i_pts_feat = i_pts_feat[:-1, :-1].permute(2, 0, 1)