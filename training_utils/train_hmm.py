import sys
sys.path.insert(0, '/storage/jalverio/')
import original_hmmlearn as hmmlearn
import numpy as np
import pickle
import os
import re

root = '/storage/jalverio/hmmlearn/training_utils/pickup_dataset_downsampled'
regex = r'\d+\.pkl'
idxs = sorted([int(path.replace('.pkl', '')) for path in os.listdir(root) if re.match(regex, path)])

for idx in idxs:
    idx_path = os.path.join(root, '%s.pkl' % idx)
    with open(idx_path, 'rb') as f:
        obs_dicts = pickle.load(f)
    for obs_dict in obs_dicts:
        import pdb; pdb.set_trace()
        x_fractional_distance, y_fractional_distance, z_fractional_distance = obs_dict['fractional_distances']
        finger_width = obs_dict['finger_width']
        object_height = obs_dict['object_pos'][2]
        block_vertical_velocity = obs_dict['object_velocity'][2]
        relative_x_velocity, relative_y_velocity, relative_z_velocity = obs_dict['object_relative_velocity']
        import pdb; pdb.set_trace()
        
        
    
    


