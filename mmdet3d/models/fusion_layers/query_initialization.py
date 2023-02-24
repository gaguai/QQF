import torch
import numpy as np
from mmcv.cnn import ConvModule, xavier_init
from mmcv.cnn import build_norm_layer
from torch import nn as nn
from torch.nn import functional as F
import spconv.pytorch as spconv

from . import apply_3d_transformation

from mmdet3d.models.builder import build_loss
from mmdet3d.core.bbox.structures import get_proj_mat_by_coord_type

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

class STGatedAttn(nn.Module):
    def __init__(self, num_channels):
        super(STGatedAttn, self).__init__()
        self.num_levels = len(num_channels)
        conv_gating = []
        conv_out = []
        for nl in range(self.num_levels):
            conv_gating.append(nn.Conv2d(num_channels[nl]*2, 1, 3, 1, 1))
            conv_out.append(nn.Conv2d(num_channels[nl], num_channels[nl], 3, 1, 1))
            
        self.conv_gating = nn.ModuleList(conv_gating)
        self.conv_out = nn.ModuleList(conv_out)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
    
    def forward(self, img_feats_seq):
        seq_len = len(img_feats_seq)

        out_feat_list = []
        for nl in range(self.num_levels):
            cur_img_feat = img_feats_seq[0][nl]
            for seq in range(1,seq_len):
                prev_img_feat = img_feats_seq[seq][nl]
                cur_weight = self.conv_gating[nl](torch.cat([cur_img_feat, prev_img_feat], 1))
                cur_weight = self.sigmoid(cur_weight)
                prev_weight = 1 - cur_weight
                gated_feat = cur_img_feat * cur_weight + prev_img_feat * prev_weight
                if seq == 1:
                    gated_feat_all = gated_feat
                else:
                    gated_feat_all += gated_feat
        
            out_feat = self.conv_out[nl](gated_feat_all)
            out_feat_list.append(out_feat)
        
        return out_feat_list

class ImgQueryInit(nn.Module):
    def __init__(self,
                 img_query_init_cfg,
                 data_version='v1.0-trainval',
                 data_root='./data/nuscenes'):
        super(ImgQueryInit, self).__init__()
        self.num_channels = img_query_init_cfg['num_channels']
        self.num_backbone_outs = len(self.num_channels)
        self.sparse_shape = img_query_init_cfg['sparse_shape']
        self.voxel_size = img_query_init_cfg['voxel_size']
        self.point_cloud_range = img_query_init_cfg['point_cloud_range']
        self.max_num_ne_voxel = img_query_init_cfg['max_num_ne_voxel']
        self.coord_type = img_query_init_cfg['coord_type']

        self.gated_attn = STGatedAttn(self.num_channels)
        
        self.data_version = data_version
        self.data_root = data_root
        self.nusc = None

    def split_param(self, pts_feats, coor_2d, coor_2d_o, img_feats, pts, num_points, img_meta):
        """nuscene dataset have 6 imgae in each sample
        1. convert img_feats [B, 6, C, H, W] -> [B*6, C, H, W]
        2. convert else [B, P, C] -> [B*6, P, C] base coor_2d[:, 0]
        """
        N = 6  # number of images
        if False:
            for b in range(len(img_feats)):
                B, _, C, H, W = img_feats[b].shape
                img_feats[b] = img_feats[b].view(-1, C, H, W)

        IC = img_feats[0].shape[1]
        B, P, C = pts_feats.shape
        max_points = 0
        for b in range(B):
            b_mod = coor_2d[b][:, 0][:num_points[b]].to(torch.long)
            for n in range(N):
                mask_n = (b_mod == n).sum()
                max_points = max(max_points, mask_n)
        pts_feats_n = torch.zeros((B * N, max_points, C), device=pts.device)
        img_feats_n = torch.zeros((B * N, max_points, IC), device=pts.device)
        coor_2d_n = torch.zeros((B * N, max_points, 2), device=pts.device)
        coor_2d_n_o = torch.zeros((B * N, max_points, 2), device=pts.device)
        pts_mask_n = torch.zeros((B * N, max_points), device=pts.device, dtype=torch.long)
        pts_n = torch.zeros((B * N, max_points, 3), device=pts.device)
        num_points_n = []
        for b in range(B):
            b_mod = coor_2d[b][:, 0][:num_points[b]].to(torch.long)
            for n in range(N):
                mask = (b_mod == n)
                mask_n = mask.sum()
                pts_feats_n[b*N+n, :mask_n] = pts_feats[b, :num_points[b]][mask]
                coor_2d_n[b*N+n, :mask_n] = coor_2d[b, :num_points[b], 1:3][mask]
                coor_2d_n_o[b*N+n, :mask_n] = coor_2d_o[b, :num_points[b], 1:3][mask] // 4
                pts_n[b*N+n, :mask_n] = pts[b, :num_points[b]][mask]
                img_coor = coor_2d_o[b, :num_points[b]][mask][:, 1:].to(torch.long) // 4
                img_feats_n[b*N+n, :mask_n] = img_feats[0][b*6+n, :, img_coor[:, 1], img_coor[:, 0]].permute(1, 0)
                pts_mask_n[b*N+n][:mask_n] = mask.nonzero().squeeze()
                num_points_n.append(mask_n)

        return pts_feats_n, img_feats_n, coor_2d_n, coor_2d_n_o, pts_n, num_points_n, pts_mask_n
    
    def get_2d_coor_multi(self, img_meta, points, proj_mat, coord_type, img_scale_factor,
                      img_crop_offset, img_flip, img_pad_shape, img_shape,
                      nusc):
        # apply transformation based on info in img_meta
        points = apply_3d_transformation(
            points, coord_type, img_meta, reverse=True)

        coor_2d = torch.zeros(points.shape[0], 3).to(device=points.device)
        coor_2d_o = torch.zeros(points.shape[0], 3).to(device=points.device)
        for idx in range(len(img_meta['filename'])):
            # project points to camera coordinate
            pts_2d, point_idx = projection(points, nusc, img_meta, idx)

            # img transformation: scale -> crop -> flip
            # the image is resized by img_scale_factor
            img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
            img_coors -= img_crop_offset

            # grid sample, the valid grid range should be in [-1,1]
            coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

            if img_flip:
                # by default we take it as horizontal flip
                # use img_shape before padding for flip
                orig_h, orig_w = img_shape
                coor_x = orig_w - coor_x

            h, w = img_pad_shape
            grid_o = torch.cat([coor_x, coor_y],
                            dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2
            coor_y = coor_y / h
            coor_x = coor_x / w
            grid = torch.cat([coor_x, coor_y],
                            dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

            coor_2d[point_idx, 0] = idx
            coor_2d[point_idx, 1:3] = grid
            coor_2d_o[point_idx, 0] = idx
            coor_2d_o[point_idx, 1:3] = grid_o

        return coor_2d, coor_2d_o
    
    def coor2pts(self, x, pad=0.0):
        ratio = self.sparse_shape[1] / x.spatial_shape[1]
        pts = (x.indices.to(torch.float) + pad) * torch.tensor(
            (self.voxel_size + [1])[::-1]).cuda() * ratio
        pts[:, 0] = pts[:, 0] / ratio - pad
        pts[:, 1:] += torch.tensor(self.point_cloud_range[:3][::-1]).cuda()
        pts[:, 1:] = pts[:, [3, 2, 1]]
        pts_list = []
        for i in range(pts[-1][0].int() + 1):
            pts_list.append(pts[pts[:, 0] == i][:, 1:])
        return pts_list
    
    def forward(self, img_feats_seq, voxel_feats, img_metas_seq, imgs_seq):
        if self.nusc is None:
            self.nusc = NuScenes(version=self.data_version, dataroot=self.data_root, verbose=True)
        batch_size = voxel_feats.batch_size.item()
        img_feats_seq = [img_feats[:self.num_backbone_outs] for img_feats in img_feats_seq]
        pts = self.coor2pts(voxel_feats, 0.5)
        img_feats = self.gated_attn(img_feats_seq)

        num_points = [pt.shape[0] for pt in pts]
        self.max_num_ne_voxel = max(num_points)

        pts_feats = voxel_feats.features
        pts_feats_b = torch.zeros(
            (batch_size, self.max_num_ne_voxel, pts_feats.shape[1]),
            device=pts_feats.device)
        coor_2d_b = torch.zeros((batch_size, self.max_num_ne_voxel, 3),
                                device=pts_feats.device)
        coor_2d_b_o = torch.zeros((batch_size, self.max_num_ne_voxel, 3),
                                   device=pts_feats.device)
        pts_b = torch.zeros((batch_size, self.max_num_ne_voxel, 3),
                            device=pts_feats.device)
        
        st = 0
        for b in range(batch_size):
            img_meta = img_metas_seq[0][b]
            img_scale_factor = (
                pts[b].new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                pts[b].new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
            coor_2d, coor_2d_o = self.get_2d_coor_multi(
                img_meta=img_meta,
                points=pts[b],
                proj_mat=pts[b].new_tensor(proj_mat),
                coord_type=self.coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_meta['input_shape'][:2],
                img_shape=img_meta['img_shape'][:2],
                nusc=self.nusc)
            pts_b[b, :pts[b].shape[0]] = pts[b][:, :3]
            coor_2d_b[b, :pts[b].shape[0]] = coor_2d
            coor_2d_b_o[b, :pts[b].shape[0]] = coor_2d_o
            pts_feats_b[b, :pts[b].shape[0]] = pts_feats[st:st+num_points[b]]
            st += num_points[b]
            # breakpoint()
        
        pts_feats_n, img_feats_n, coor_2d_n, coor_2d_n_o, pts_n, num_points_n, pts_mask_n = self.split_param(
            pts_feats_b, coor_2d_b, coor_2d_b_o, img_feats, pts_b, num_points, img_meta)
        
        breakpoint()
        return pts_feats_n, img_feats_n, coor_2d_n, coor_2d_n_o, pts_n, num_points_n, pts_mask_n


class QueryInit(nn.Module):
    def __init__(self,
                 query_init_cfg,
                 kernel_size=3):
        super(QueryInit, self).__init__()

        in_channels = query_init_cfg['in_channels']
        self.voxel_stride = query_init_cfg['voxel_stride']
        self.topk = query_init_cfg['top_k']
        self.point_cloud_range = torch.Tensor(query_init_cfg['point_cloud_range']).cuda()
        self.voxel_size = torch.Tensor(query_init_cfg['voxel_size']).cuda()
        self.inv_idx =  torch.Tensor([2, 1, 0]).long().cuda()
        self.threshold = query_init_cfg['threshold']
        self.loss_focal = query_init_cfg['loss_focal']
        self.norm_cfg = query_init_cfg['norm_cfg']
        self.update_query = query_init_cfg['update_query']

        _step = int(kernel_size//2)
        kernel_offsets = [[i, j, k] for i in range(-_step, _step+1) for j in range(-_step, _step+1) for k in range(-_step, _step+1)]
        kernel_offsets.remove([0, 0, 0])
        self.kernel_offsets = torch.Tensor(kernel_offsets).cuda()

        self.conv_imp = spconv.SubMConv3d(in_channels, kernel_size**3, 
                                          kernel_size=3, stride=1,padding=1, bias=False,
                                          indice_key='focal_imp')
        self.focal_loss = build_loss(self.loss_focal)

        if query_init_cfg['update_query']:
            self.conv = spconv.SubMConv3d(in_channels, in_channels, 
                                        kernel_size=3, stride=1,padding=1, bias=False,
                                        indice_key='query_conv')
            self.bn = build_norm_layer(self.norm_cfg, in_channels)[1]
            self.relu = nn.ReLU(True)

    def sort_by_indices(self, features, indices):
        """
            To sort the sparse features with its indices in a convenient manner.
            Args:
                features: [N, C], sparse features
                indices: [N, 4], indices of sparse features
        """
        idx = indices[:, 1:]
        idx_sum = idx.select(1, 0) * idx[:, 1].max() * idx[:, 2].max() + idx.select(1, 1) * idx[:, 2].max() + idx.select(1, 2)
        _, ind = idx_sum.sort()
        features = features[ind]
        indices = indices[ind]
        return features, indices
    
    def check_repeat(self, features, indices, sort_first=False):
        """
            Check that whether there are replicate indices in the sparse features, 
            remove the replicate features if any.
        """
        if sort_first:
            features, indices = self.sort_by_indices(features, indices)
            features, indices = features.flip([0]), indices.flip([0])
            idx = indices[:, 1:].int()
            idx_sum = torch.add(torch.add(idx.select(1, 0) * idx[:, 1].max() * idx[:, 2].max(), idx.select(1, 1) * idx[:, 2].max()), idx.select(1, 2))
            _unique, inverse = torch.unique_consecutive(idx_sum, return_inverse=True, dim=0)
        else:
            _unique, inverse = torch.unique(indices, return_inverse=True, dim=0)
        if _unique.shape[0] < indices.shape[0]:
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            if sort_first:
                features_new = torch.zeros((_unique.shape[0], features.shape[-1]), device=features.device)
                features_new.index_add_(0, inverse.long(), features)
                features = features_new
                perm_ = inverse.new_empty(_unique.size(0)).scatter_(0, inverse, perm)
                indices = indices[perm_].int()
            else:
                inverse, perm = inverse.flip([0]), perm.flip([0])
                perm_ = inverse.new_empty(_unique.size(0)).scatter_(0, inverse, perm)
                features = features[perm_]
                indices = _unique.int()
        return features, indices
    
    def split_voxels(self, x, b, imps_3d, voxels_3d, kernel_offsets, mask_multi=True, 
                     topk=True, threshold=0.5):
        """
        Generate and split the voxels into foreground and background sparse features, based on the predicted importance values.
        Args:
            x: [N, C], input sparse features
            b: int, batch size id
            imps_3d: [N, kernelsize**3], the prediced importance values
            voxels_3d: [N, 3], the 3d positions of voxel centers 
            kernel_offsets: [kernelsize**3, 3], the offset coords in an kernel
            mask_multi: bool, whether to multiply the predicted mask to features
            topk: bool, whether to use topk or threshold for selection
            threshold: float, threshold value
        """

        index = x.indices[:, 0]
        batch_index = index==b
        indices_ori = x.indices[batch_index]
        features_ori = x.features[batch_index]
        mask_voxel = imps_3d[batch_index, -1].sigmoid()
        mask_kernel = imps_3d[batch_index, :-1].sigmoid()

        if mask_multi:
            features_ori *= mask_voxel.unsqueeze(-1)

        if topk:
            _, indices = mask_voxel.sort(descending=True)
            indices_fore = indices[:int(mask_voxel.shape[0]*threshold)]
            indices_back = indices[int(mask_voxel.shape[0]*threshold):]
        else:
            indices_fore = mask_voxel > threshold
            indices_back = mask_voxel <= threshold

        features_fore = features_ori[indices_fore]
        coords_fore = indices_ori[indices_fore]

        # breakpoint()
        mask_kernel_fore = mask_kernel[indices_fore]
        mask_kernel_bool = mask_kernel_fore>=threshold
        voxel_kerels_imp = kernel_offsets.unsqueeze(0).repeat(mask_kernel_bool.shape[0],1, 1)

        indices_fore_kernels = coords_fore[:, 1:].unsqueeze(1).repeat(1, kernel_offsets.shape[0], 1)
        indices_with_imp = indices_fore_kernels + voxel_kerels_imp
        selected_indices = indices_with_imp[mask_kernel_bool]
        spatial_indices = (selected_indices[:, 0] >0) * (selected_indices[:, 1] >0) * (selected_indices[:, 2] >0)  * \
                            (selected_indices[:, 0] < x.spatial_shape[0]) * (selected_indices[:, 1] < x.spatial_shape[1]) * (selected_indices[:, 2] < x.spatial_shape[2])
        selected_indices = selected_indices[spatial_indices]
        selected_indices = torch.cat([torch.ones((selected_indices.shape[0], 1), device=features_fore.device)*b, selected_indices], dim=1)

        selected_features = torch.zeros((selected_indices.shape[0], features_ori.shape[1]), device=features_fore.device)
        selected_features, selected_indices = self.check_repeat(selected_features, selected_indices)

        features_fore_cat = torch.cat([features_fore, selected_features], dim=0)
        coords_fore = torch.cat([coords_fore, selected_indices], dim=0)

        features_fore, coords_fore = self.check_repeat(features_fore_cat, coords_fore, sort_first=True)
        features_back = features_ori[indices_back]
        coords_back = indices_ori[indices_back]

        return features_fore, coords_fore, features_back, coords_back
    
    def _gen_sparse_features(self, x, imp3_3d, voxels_3d, gt_boxes=None):
        """
            Generate the output sparse features from the focal sparse conv.
            Args:
                x: [N, C], lidar sparse features
                imps_3d: [N, kernelsize**3], the predicted importance values
                voxels_3d: [N, 3], the 3d positions of voxel centers
                gt_boxes: for focal loss calculation
        """
        index = x.indices[:, 0]
        batch_size = x.batch_size
        voxel_features_fore = []
        voxel_indices_fore = []
        voxel_features_back = []
        voxel_indices_back = []

        box_of_pts_cls_targets = []
        mask_voxels = []

        loss_box_of_pts = 0
        for b in range(batch_size):

            if self.training:
                gt_boxes_batch = gt_boxes[b].tensor.cuda()
                gt_boxes_batch_idx = (gt_boxes_batch**2).sum(-1)>0
                gt_boxes_centers_batch = gt_boxes_batch[gt_boxes_batch_idx, :3]
                gt_boxes_sizes_batch = gt_boxes_batch[gt_boxes_batch_idx, 3:6]

                index = x.indices[:, 0]
                batch_index = index==b
                mask_voxel = imp3_3d[batch_index, -1].sigmoid()
                mask_voxels.append(mask_voxel)
                voxels_3d_batch = voxels_3d[batch_index]
                dist_voxels_to_gtboxes = (voxels_3d_batch[:, self.inv_idx].unsqueeze(1).repeat(1, gt_boxes_centers_batch.shape[0], 1) - gt_boxes_centers_batch.unsqueeze(0)).abs()
                offsets_dist_boundry = dist_voxels_to_gtboxes - gt_boxes_sizes_batch.unsqueeze(0)
                inboxes_voxels = ~torch.all(~torch.all(offsets_dist_boundry<=0, dim=-1), dim=-1)
                box_of_pts_cls_targets.append(inboxes_voxels)

            features_fore, indices_fore, features_back, indices_back = self.split_voxels(x, b, imp3_3d, voxels_3d, self.kernel_offsets, mask_multi=True, topk=self.topk, threshold=self.threshold)

            voxel_features_fore.append(features_fore)
            voxel_indices_fore.append(indices_fore)
            voxel_features_back.append(features_back)
            voxel_indices_back.append(indices_back)

        voxel_features_fore = torch.cat(voxel_features_fore+voxel_features_back, dim=0)
        voxel_indices_fore = torch.cat(voxel_indices_fore+voxel_indices_back, dim=0)
        out = spconv.SparseConvTensor(voxel_features_fore, voxel_indices_fore, x.spatial_shape, x.batch_size)
        
        if self.training:
            mask_voxels = torch.cat(mask_voxels)
            box_of_pts_cls_targets = torch.cat(box_of_pts_cls_targets)
            mask_voxels_two_classes = torch.cat([1-mask_voxels.unsqueeze(-1), mask_voxels.unsqueeze(-1)], dim=1)
            loss_box_of_pts += self.focal_loss(mask_voxels_two_classes, box_of_pts_cls_targets.long())
        return out, loss_box_of_pts
    
    def forward(self, x, gt_bboxes_3d):
        spatial_indices = x.indices[:, 1:] * self.voxel_stride
        voxels_3d = (spatial_indices * self.voxel_size[self.inv_idx]) + self.point_cloud_range[:3][self.inv_idx]

        imp3_3d = self.conv_imp(x).features
        out, loss_box_of_pts = self._gen_sparse_features(x, imp3_3d, voxels_3d, gt_bboxes_3d[0] if self.training else None)

        if self.update_query:
            updated_query = self.conv(out)
            updated_query = updated_query.replace_feature(self.bn(updated_query.features))
            updated_query = updated_query.replace_feature(self.relu(updated_query.features))
            return updated_query, loss_box_of_pts
        else:
            return out, loss_box_of_pts
        
def projection(points, nusc, img_meta, idx, img_features=None, data_root=None):

    def translate(points, x):
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            points[i, :] = points[i, :] + x[i]
        return points

    def rotate(points, rot_matrix: np.ndarray):
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        points[:3, :] = np.dot(rot_matrix, points[:3, :])
        return points

    ##################################################################
    # projection
    # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py
    # def map_pointcloud_to_image
    ##################################################################

    sample_token = nusc.get('sample', img_meta['sample_idx'])
    cam_token = sample_token['data'][img_meta['filename'][idx].split('__')[-2]]
    cam = nusc.get('sample_data', cam_token)
    pointsensor_token = sample_token['data']['LIDAR_TOP']
    min_dist = 1.0

    # pc = LidarPointCloud.from_file(img_meta['pts_filename'])
    img_shape = img_meta['ori_shape'][:2]
    pointsensor = nusc.get('sample_data', pointsensor_token)
    points_ = torch.transpose(points, 1, 0).cpu().numpy()

    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor',
                         pointsensor['calibrated_sensor_token'])
    points_ = rotate(points_,
                     Quaternion(cs_record['rotation']).rotation_matrix)
    points_ = translate(points_, np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    points_ = rotate(points_,
                     Quaternion(poserecord['rotation']).rotation_matrix)
    points_ = translate(points_, np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    points_ = translate(points_, -np.array(poserecord['translation']))
    points_ = rotate(points_,
                     Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    points_ = translate(points_, -np.array(cs_record['translation']))
    points_ = rotate(points_,
                     Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    depths = points_[2, :]
    pts_2d = view_points(
        points_[:3, :],
        np.array(cs_record['camera_intrinsic']),
        normalize=True)
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, pts_2d[0, :] > 1)
    mask = np.logical_and(mask, pts_2d[0, :] < img_shape[1] - 1)
    mask = np.logical_and(mask, pts_2d[1, :] > 1)
    mask = np.logical_and(mask, pts_2d[1, :] < img_shape[0] - 1)
    pts_2d = pts_2d[:, mask]
    pts_2d = points.new_tensor(np.transpose(pts_2d[:2, :], (1, 0)))
    point_idx = (mask != 0).nonzero()[0]

    return pts_2d, point_idx
