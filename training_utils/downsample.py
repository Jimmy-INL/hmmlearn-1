import os
import re
import numpy as np
import pickle
from numpy.random import choice
import cv2

NUM_RESAMPLES = 3
root = '/storage/jalverio/hmmlearn/training_utils/pickup_dataset'
new_root = '/storage/jalverio/hmmlearn/training_utils/pickup_dataset_downsampled'

if not os.path.isdir(root):
    root = '/Users/julianalverio/code/hmmlearn/training_utils/pickup_dataset'
    new_root = '/Users/julianalverio/code/hmmlearn/training_utils/pickup_dataset_downsampled'    


def write_mp4(frames, path=None):
    if path is None:
        path = '/storage/jalverio/test.mp4'
    shape = (500, 500)
    writer = cv2.VideoWriter(path, 0x7634706d, 5, shape)
    for frame in frames:
        writer.write(frame)
    writer.release()

def downsample_idxs(length):
    chunk_size = length // 10
    low = 0
    high = 0
    sampled_idxs = list()
    for count in range(10):
        low = high
        high += chunk_size
        if count == 10:
            high = length
        sampled_idx = choice(np.arange(low, high))
        sampled_idxs.append(sampled_idx)
    return sampled_idxs

def resample_videos():
    print('Resampling %s times' % NUM_RESAMPLES)
    regex = r'\d+\.pkl'
    idxs = sorted([int(path.replace('.pkl', '')) for path in os.listdir(root) if re.match(regex, path)])
    for idx in idxs:
        print('resampling %s out of %s' % (idx, max(idxs)))
        for resample in range(NUM_RESAMPLES):
            observations_path = os.path.join(root, '%s.pkl' % idx)
            frames_path = os.path.join(root, '%s.npy' % idx)
            with open(observations_path, 'rb') as f:
                observations = pickle.load(f)
            frames = np.load(frames_path)
            length = len(observations)
            downsampled_idxs = downsample_idxs(length)
            new_observations = [observations[down_idx] for down_idx in downsampled_idxs]
            new_frames = [frames[down_idx] for down_idx in downsampled_idxs]

            new_frames_path = os.path.join(new_root, '%s.npy' % idx)
            new_observations_path = os.path.join(new_root, '%s.pkl' % idx)
            new_mp4_path = os.path.join(new_root, '%s.mp4' % idx)
            np.save(new_frames_path, new_frames)
            with open(new_observations_path, 'wb') as f:
                pickle.dump(new_observations, f)
            write_mp4(new_frames, path=new_mp4_path)


if __name__ == '__main__':
    resample_videos()
