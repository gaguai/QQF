import pickle as pkl
import os
from nuscenes.nuscenes import NuScenes
import glob
from tqdm import tqdm 

data_root = './data/nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=data_root, verbose=True) 
pkl_path = os.path.join(data_root,'nuscenes_dbinfos_train.pkl')

with open(pkl_path,'rb') as f:
    dbinfos = pkl.load(f)

db_list = os.listdir('data/nuscenes/nuscenes_gt_database')


prev_dbinfos = {}

for obj in tqdm(dbinfos.keys()):
    prev_dbinfos[obj] = []
    db_cls = dbinfos[obj]

    for i in tqdm(db_cls):
        sample_token = i['image_idx']
        sample_data = nusc.get('sample',sample_token)
        
        prev_token = sample_data['prev']

        db_fname = i['path'].split('/')[-1]

        if prev_token=='':
            prev_dbinfos[obj].append(i)
            continue

        prev_fname = db_fname.split('_')
        prev_fname[0] = prev_token
        prev_fname = '_'.join(prev_fname)

        if prev_fname in db_list:
            for j in db_cls:
                if j['path'].split('/')[-1] == prev_fname:
                    prev_dbinfos[obj].append(j)
                    break
        else:
            prev_dbinfos[obj].append(i)

result_pkl_path = os.path.join(data_root,'nuscenes_t-1_dbinfos_train.pkl')

with open(result_pkl_path,'wb') as f:
    pkl.dump(prev_dbinfos, f)