import pickle as pkl
import os
from nuscenes.nuscenes import NuScenes
import glob
from tqdm import tqdm 
import copy
data_root = './data/nuscenes'
# nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True) 
pkl_path = os.path.join(data_root,'nuscenes_infos_val.pkl')
seq_len = 2

with open(pkl_path,'rb') as f:
    infos = pkl.load(f)

prev_infos = {}
prev_infos['infos'] = []

for i in tqdm(range(len(infos['infos']))):
    prev_info, cur_info = infos['infos'][i-1], infos['infos'][i]
    
    if prev_info['scene_token']==cur_info['scene_token']:
        prev_infos['infos'].append([cur_info,prev_info])
    else:
        prev_infos['infos'].append([cur_info,cur_info])
   
prev_infos['metadata'] = infos['metadata']

result_pkl_path = os.path.join(data_root,'nuscenes_seq_2_infos_val.pkl')

with open(result_pkl_path,'wb') as f:
    pkl.dump(prev_infos, f)