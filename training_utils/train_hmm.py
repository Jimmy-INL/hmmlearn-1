import sys
from hmmlearn.hmm import GaussianHMM
import numpy as np
import pickle
import os
import re

root = '/storage/jalverio/hmmlearn/training_utils/pickup_dataset_downsampled'
regex = r'\d+\.pkl'
idxs = sorted([int(path.replace('.pkl', '')) for path in os.listdir(root) if re.match(regex, path)])

all_videos = list()
for idx in idxs:
    idx_path = os.path.join(root, '%s.pkl' % idx)
    with open(idx_path, 'rb') as f:
        obs_dicts = pickle.load(f)
    video = list()
    for obs_dict in obs_dicts:
        x_fractional_distance, y_fractional_distance, z_fractional_distance = obs_dict['fractional_distances']
        finger_width = obs_dict['finger_width']
        object_height = obs_dict['object_position'][2] - 0.4247690273656512
        block_vertical_velocity = obs_dict['object_velocity'][2]
        relative_x_velocity, relative_y_velocity, relative_z_velocity = obs_dict['object_relative_velocity']
        feature_vec = np.array([x_fractional_distance, y_fractional_distance, z_fractional_distance, finger_width, object_height, block_vertical_velocity, relative_x_velocity, relative_y_velocity, relative_z_velocity])
        video.append(feature_vec)
    video = np.array(video)  # 10 x feature_vec_length == 10x9
    assert video.shape == (10, 9)

    all_videos.append(video)  # target shape: length of all_videos together

num_videos = len(all_videos)
all_videos = np.concatenate(all_videos)
lengths = [10 for _ in range(num_videos)]
assert all_videos.shape == (num_videos * 10, 9)
startprob = np.zeros(3)
startprob[0] = 1.
transmat = [[0.99, 0.01, 0.],
            [0., 0.99, 0.01],
            [0., 0., 1.]]
transmat = np.array(transmat)


NUM_ATTEMPTS = 5

for attempt in range(NUM_ATTEMPTS):
    model = GaussianHMM(n_components=3, covariance_type='diag', params='tmc', init_params='mc', verbose=True)
    model.startprob_ = startprob
    model.transmat_ = transmat

    model.fit(all_videos)

