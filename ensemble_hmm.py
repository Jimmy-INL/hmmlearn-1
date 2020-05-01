from hmm import TimedGaussianHMM
import numpy as np
import pickle
import copy
import os
import re
from scipy.stats import norm


class EnsembleHMM(object):
    def __init__(self, num_states=None, num_frames=10, prefix=None, train=False):
        np.seterr(all='raise')
        self.num_states = num_states
        self.num_frames = num_frames
        if not prefix:
            if os.path.exists('/storage/jalverio/hmmlearn/tuned_params'):
                prefix = '/storage/jalverio/hmmlearn/tuned_params'
            else:
                assert False, 'I could not find a good path to initialize the hmms in the init of EnsembleMultivariateMultinomial'
        self.prefix = prefix
        if train:
            self.train_ensemble()
        models = dict()
        models_dir = os.path.join(self.prefix, 'hmms')
        file_regex = '(\d+)_.*\.npy'
        matrix_files = os.listdir(models_dir)
        feature_idxs = list()
        for filename in matrix_files:
            if re.match(file_regex, filename):
                feature_idxs.append(int(re.match(file_regex, filename).groups()[0]))
        feature_idxs = sorted(list(set(feature_idxs)))

        for feature_idx in feature_idxs:
            transmat_path = os.path.join(models_dir, 'transmat.npy')
            startprob_path = os.path.join(models_dir, 'startprob.npy')

            means_path = os.path.join(models_dir, '%s_means.npy' % feature_idx)
            covars_path = os.path.join(models_dir, '%s_covars.npy' % feature_idx)
            model_path = os.path.join(models_dir, '%s.pkl' % feature_idx)
            # warning! hard-coded number of samples (number of frames in a video)
            model = TimedGaussianHMM(n_components=self.num_states, num_frames=self.num_frames, n_iter=100, verbose=True, params='mc', covariance_type='diag', init_params='mc')
            # model.startprob_ = np.load(startprob_path)
            model.startprob_ = np.zeros(30)
            model.startprob_[0] = 1.
            model.covars_ = np.load(covars_path)[:, :, 0]
            model._covars_ = model.covars_
            model.means_ = np.load(means_path)
            model.ndim = 1
            model.n_features = 1
            models[feature_idx] = model
        self.models = models
        print('loaded %s hmm models' % len(self.models))

    def load_model(self, transmat_path, startprob_path, means_path, covars_path):
        transmat = np.load(transmat_path)
        # startprob = np.load(startprob_path)
        # warning! hard-coded number
        startprob = np.zeros(30, 0)
        startprob[0] = 1.
        means = np.load(means_path)
        covars = np.load(covars_path)[:, :, 0]
        covars = np.clip(covars, a_min=5e-3, a_max=None)

        num_states = startprob.shape[0]
        # this probably needs to be updated?
        import pdb; pdb.set_trace()
        model = GaussianHMM(n_components=num_states, n_iter=100, verbose=True, params='mc', covariance_type='diag', init_params='mc', min_covar=5e-3)
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = covars
        model.n_features = 1
        return model
    
    def _unpack_video(self, video):
        num_features = video[0].shape[0]
        feature_dict = dict()
        for feature_idx in range(num_features):
            feature_dict[feature_idx] = list()
        for frame in video:
            for idx, feature in enumerate(frame):
                feature_dict[idx].append(feature)
        length = len(video)
        return feature_dict, length, num_features

    # get the forward matrices for all the frames in a video
    def predict_video(self, video):
        feature_dict, length, num_features = self._unpack_video(video)
        preds_dict = dict()
        for feature_idx in range(num_features):
            preds_dict[feature_idx] = list()  # maps from feature_idx to list of preds for feature
        for feature_idx in range(num_features):
            feature_video = np.array(feature_dict[feature_idx])
            model = self.models[feature_idx]
            fwd_lattice = model.get_fwd_lattice(video)
            preds_dict[feature_idx].append(fwd_lattice)
        all_preds = np.array([value for value in preds_dict.values()])
        all_preds = np.squeeze(all_preds)
        return all_preds
                
    def predict_one_frame(self, feature_vec):
        preds = list()
        for model_idx, value in enumerate(feature_vec):
            model = self.models[model_idx]
            model_preds = list()
            for mean, covar in zip(np.squeeze(model.means_), np.squeeze(model.covars_)):
                model_preds.append(norm(loc=mean, scale=covar).pdf(value))
            preds.append(np.array(model_preds))
        model_preds = np.stack(preds, axis=0)
        ensemble_preds = np.mean(model_preds, axis=0)
        
        return model_preds, ensemble_preds

    # this only accepts a single video!
    # main function for computing the reward from a video
    def distribution_distance_single_video(self, video, episode):
        feature_dict, length, num_features = self._unpack_video(video)
        probs_dict = dict()  # map from feature_number to a list of probs, one matrix per frame
        for feature_idx in range(num_features):
            feature_video = np.array(feature_dict[feature_idx])[:, np.newaxis]
            model = self.models[feature_idx]
            probs = model.predict_proba_simple(feature_video)  # (10, 3)  ==  (num_frames, num_states)
            probs_dict[feature_idx] = probs
        all_probs = np.array(list(zip(probs_dict.values())))  # (1, 5, 10, 3)
        all_probs = np.squeeze(all_probs)  # (5, 10, 3) == (num_features, num_frames/video, num_states)
        all_probs_original = all_probs.copy()  # for debugging
        all_probs = all_probs[:, -1, :]  # grab last frame

        all_probs /= all_probs.sum(axis=1)[:, np.newaxis]  # normalize to probability space
        all_distances = list()
        for feature_idx in range(num_features):
            feature_dist = all_probs[feature_idx]
            distance = self._distance_from_target(feature_dist)
            all_distances.append(distance)
        mean_distance = np.mean(all_distances)
        mean_probs = np.mean(all_probs, axis=0)
        mean_probs /= np.sum(mean_probs)  # normalize probabilities
        most_likely_state = np.argmax(mean_probs)
        return mean_distance, mean_probs, most_likely_state

    # if you don't have a target, the target is [0, 0, 1]. Find the custom earth-mover distance
    # if you have a target, ignore the first probability in the distribution and find the weighted mean of
    # the rest of the distribution. Find the difference between these weighted means and normalize
    def _distance_from_target(self, actual):
        distance = (actual * np.arange(actual.shape[0] - 1, -1, -1)).sum()
        max_distance = actual.shape[0] - 1
        return 1. - distance / max_distance

    #### TRAINING STUFF
    '''
    videos is a length-1000 list (the dataset)
    videos[0] is a length-9 list (one video)
    videos[0][0] is a (5,) array (one frame)
    '''
    def train_model(self, model, data, lengths, attempts=3):
        models = list()
        for attempt in range(attempts):
            current_model = copy.deepcopy(model)
            current_model.fit(data, lengths)
            score = current_model.monitor_.history[-1]
            models.append((current_model, score))
        sorted_models = sorted(models, key=lambda x: x[1], reverse=True)
        return sorted_models[0][0]

    
    def make_transition_matrix(self, p=0.99):
        transition_matrix = np.eye(self.num_states) * p
        for row_idx in range(transition_matrix.shape[0]):
            for col_idx in range(transition_matrix.shape[1]):
                if (row_idx + 1) == col_idx:
                    transition_matrix[row_idx, col_idx] = 1 - p
        transition_matrix[-1, -1] = 1.
        return transition_matrix

    def train_ensemble(self):
        with open(os.path.join(self.prefix, 'pickup_features.pkl'), 'rb') as f:
            videos = pickle.load(f)

        feature_dict, lengths, num_features, num_videos = self._unpack_videos(videos)

        transition_matrix = self.make_transition_matrix()
        startprob = np.zeros(self.num_states)
        startprob[0] = 1.

        # for feature_idx in range(0, num_features):
        for feature_idx in [4]:
            print('now starting %s' % feature_idx)
            # model = GaussianHMM(n_components=self.num_states, n_iter=100, verbose=True, params='mc', covariance_type='diag', init_params='mc')
            # model.transmat_ = transition_matrix
            # model.startprob_ = startprob
            model = MultinomialHMM(n_components=self.num_states, n_iter=25, verbose=True, params='e', init_params='e',transmat_prior=transition_matrix, startprob_prior=startprob)
            data = np.array(feature_dict[feature_idx]).astype(np.uint8)
            data = np.expand_dims(data, axis=1)
            lengths = np.full(num_videos, 9)
            model = self.train_model(model, data, lengths, attempts=5)
            transmat_save_path = os.path.join(self.prefix, 'hmms/%s_transmat.npy' % feature_idx)
            np.save(transmat_save_path, model.transmat_)
            startprob_save_path = os.path.join(self.prefix, 'hmms/%s_startprob.npy' % feature_idx)
            np.save(startprob_save_path, model.startprob_)
            # means_save_path = os.path.join(self.prefix, 'hmms/%s_means.npy' % feature_idx)
            # np.save(means_save_path, model.means_)
            # covars_save_path = os.path.join(self.prefix, 'hmms/%s_covars.npy' % feature_idx)
            # np.save(covars_save_path, model.covars_)
            emission_save_path = os.path.join(self.prefix, 'hmms/%s_emissions.npy' % feature_idx)
            np.save(emission_save_path, model.emissionprob_)
            print('pickled!')
            import pdb; pdb.set_trace()

if __name__ == '__main__':
    # view_models()
    ensemble = EnsembleMutivariateMultinomial(train=True, num_states=3)
