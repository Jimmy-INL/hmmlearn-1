import numpy as np
import cv2
import os



def mp4_to_pkl(root, path):
    vidcap = cv2.VideoCapture(path)
    success, frame = vidcap.read()
    all_frames = list()
    while success:
        all_frames.append(frame)
        success, frame = vidcap.read()
    all_frames = np.array(all_frames)
    np_save_path = path.replace('.mp4', '.npy')
    np.save(np_save_path, all_frames)


if __name__ == '__main__':
    root = '/storage/jalverio/hmmlearn/training_utils/pickup_dataset'
    mp4_files = [path for path in os.listdir(root) if path.endswith('.mp4')]
    for count, mp4_file in enumerate(mp4_files):
        print('%s out of %s' % (count, len(mp4_files)))
        mp4_path = os.path.join(root, mp4_file)
        mp4_to_pkl(root, mp4_path)
    print('done')

