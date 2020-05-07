import numpy as np
import os

feature_idx_to_name = dict({
            0: 'horiz_dist',
            1: 'vert_dist',
            2: 'depth_dist',
            3: 'vert_displ',
            4: 'finger_width'
        })

prefix = '/Users/julianalverio/code/sentence_tracker/models_and_data/hmms'
if not os.path.isdir(prefix):
    prefix = '/storage/jalverio/sentence_tracker/models_and_data/hmms'

startprob = np.array([1., 0.])
transition = np.array([[1., 1.],
                       [0., 1.]])
transition_path = os.path.join(prefix, 'transmat.npy')
startprob_path = os.path.join(prefix, 'startprob.npy')
np.save(transition_path, transition)
np.save(startprob_path, startprob)

# HORIZONTAL
first_means = np.array([0.18, 0.0])[:, np.newaxis]
first_covars = np.array([0.06, 0.04])[:, np.newaxis, np.newaxis]
first_means_path = os.path.join(prefix, '0_means.npy')
first_covars_path = os.path.join(prefix, '0_covars.npy')
np.save(first_means_path, first_means)
np.save(first_covars_path, first_covars)

# VERTICAL
second_means = np.array([0.18, 0.0])[:, np.newaxis]
second_covars = np.array([0.06, 0.04])[:, np.newaxis, np.newaxis]
second_means_path = os.path.join(prefix, '1_means.npy')
second_covars_path = os.path.join(prefix, '1_covars.npy')
np.save(second_means_path, second_means)
np.save(second_covars_path, second_covars)

# DEPTH
third_means = np.array([0.18, 0.0])[:, np.newaxis]
third_covars = np.array([0.06, 0.04])[:, np.newaxis, np.newaxis]
third_means_path = os.path.join(prefix, '2_means.npy')
third_covars_path = os.path.join(prefix, '2_covars.npy')
np.save(third_means_path, third_means)
np.save(third_covars_path, third_covars)

# VERT DISPLACEMENT
fourth_means = np.array([0.06, 0.2])[:, np.newaxis]
fourth_covars = np.array([0.2, 0.07])[:, np.newaxis, np.newaxis]
fourth_means_path = os.path.join(prefix, '3_means.npy')
fourth_covars_path = os.path.join(prefix, '3_covars.npy')
np.save(fourth_means_path, fourth_means)
np.save(fourth_covars_path, fourth_covars)

# FINGER WIDTH
fifth_means = np.array([0.0, 0.042])[:, np.newaxis]
fifth_covars = np.array([0.02, 0.0113])[:, np.newaxis, np.newaxis]
fifth_means_path = os.path.join(prefix, '4_means.npy')
fifth_covars_path = os.path.join(prefix, '4_covars.npy')
np.save(fifth_means_path, fifth_means)
np.save(fifth_covars_path, fifth_covars)

print('hand-tuned model saved!')


for model_idx in range(5):
    print('Feature:', feature_idx_to_name[model_idx])
    covars_path = os.path.join(prefix, '%s_covars.npy' % model_idx)
    covars = np.load(covars_path)
    means_path = os.path.join(prefix, '%s_means.npy' % model_idx)
    means = np.load(means_path)
    print('means', means)
    print('covars', covars)
