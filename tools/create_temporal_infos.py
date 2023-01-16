import pickle as pkl
import os
from nuscenes.nuscenes import NuScenes
import glob
from tqdm import tqdm 
import copy
data_root = './data/nuscenes'
# nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True) 
pkl_path = os.path.join(data_root,'nuscenes_infos_train.pkl')
seq_len = 3

with open(pkl_path,'rb') as f:
    infos = pkl.load(f)

temporal_infos = {}
temporal_infos['infos'] = []

for i in tqdm(range(len(infos['infos']))):
    if seq_len==2:
        prev_info, cur_info = infos['infos'][i-1], infos['infos'][i]
        
        if prev_info['scene_token']==cur_info['scene_token']:
            temporal_infos['infos'].append([cur_info, prev_info])
        else:
            temporal_infos['infos'].append([cur_info, cur_info])
    
    if seq_len==3:
        t_2_info, t_1_info, cur_info = infos['infos'][i-2], infos['infos'][i-1], infos['infos'][i]
        if t_1_info['scene_token']==cur_info['scene_token']:
            if t_2_info['scene_token']==t_1_info['scene_token']:
                temporal_infos['infos'].append([cur_info,t_1_info,t_2_info])
            else:
                temporal_infos['infos'].append([cur_info,t_1_info,t_1_info])
        else:
            temporal_infos['infos'].append([cur_info,cur_info,cur_info])

    if seq_len==4:
        t_3_info, t_2_info = infos['infos'][i-3], infos['infos'][i-2]
        t_1_info, cur_info = infos['infos'][i-1], infos['infos'][i]
        if t_1_info['scene_token']==cur_info['scene_token']:
            if t_2_info['scene_token']==t_1_info['scene_token']:
                if t_3_info['scene_token']==t_2_info['scene_token']:
                    temporal_infos['infos'].append([cur_info,t_1_info,t_2_info,t_3_info])
                else:
                    temporal_infos['infos'].append([cur_info,t_1_info,t_2_info,t_2_info])
            else:
                temporal_infos['infos'].append([cur_info,t_1_info,t_1_info,t_1_info])
        else:
            temporal_infos['infos'].append([cur_info,cur_info,cur_info,cur_info])

temporal_infos['metadata'] = infos['metadata']

result_pkl_path = os.path.join(data_root,f'nuscenes_seq_{seq_len}_infos_train.pkl')

with open(result_pkl_path,'wb') as f:
    pkl.dump(temporal_infos, f)