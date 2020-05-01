# Hidden Markov Models
#
# Author: Ron Weiss <ronweiss@gmail.com>
#         Shiqiao Du <lucidfrontier.45@gmail.com>
# API changes: Jaques Grobler <jaquesgrobler@gmail.com>
# Modifications to create of the HMMLearn module: Gael Varoquaux
# More API changes: Sergei Lebedev <superbobry@gmail.com>

"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import logging

import numpy as np
from scipy.special import logsumexp
from sklearn import cluster
from sklearn.utils import check_random_state

import _utils
from stats import log_multivariate_normal_density
from base import _BaseHMM, _TimeBaseHMM
from utils import (
    fill_covars, iter_from_X_lengths, log_mask_zero, log_normalize, normalize)
import sys
sys.path.insert(0, '/storage/jalverio/hmmlearn')
from time_hmm import make_transition_matrix
sys.path.pop(0)
from base import ConvergenceMonitor

import logging
import string
import sys
from collections import deque

from sklearn.base import BaseEstimator
from sklearn.utils import check_array

from utils import normalize, log_normalize, iter_from_X_lengths, log_mask_zero
from scipy.stats import norm

__all__ = ["GMMHMM", "GaussianHMM", "MultinomialHMM", "TimedGaussianHMM"]


_log = logging.getLogger(__name__)
COVARIANCE_TYPES = frozenset(("spherical", "diag", "full", "tied"))


def _check_and_set_gaussian_n_features(model, X):
    _, n_features = X.shape
    if hasattr(model, "n_features") and model.n_features != n_features:
        raise ValueError("Unexpected number of dimensions, got {} but "
                         "expected {}".format(n_features, model.n_features))
    model.n_features = n_features

class TimedGaussianHMM(_TimeBaseHMM):
    r"""Hidden Markov Model with Gaussian emissions.

    Parameters
    ----------
    n_components : int
        Number of states.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of

        * "spherical" --- each state uses a single variance value that
          applies to all features.
        * "diag" --- each state uses a diagonal covariance matrix.
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix.
        * "tied" --- all states use **the same** full covariance matrix.

        Defaults to "diag".

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    means_prior, means_weight : array, shape (n_components, ), optional
        Mean and precision of the Normal prior distribtion for
        :attr:`means_`.

    covars_prior, covars_weight : array, shape (n_components, ), optional
        Parameters of the prior distribution for the covariance matrix
        :attr:`covars_`.

        If :attr:`covariance_type` is "spherical" or "diag" the prior is
        the inverse gamma distribution, otherwise --- the inverse Wishart
        distribution.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or`"map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    means\_ : array, shape (n_components, n_features)
        Mean parameters for each state.

    covars\_ : array
        Covariance parameters for each state.

        The shape depends on :attr:`covariance_type`::

            (n_components, )                        if "spherical",
            (n_components, n_features)              if "diag",
            (n_components, n_features, n_features)  if "full"
            (n_features, n_features)                if "tied",

    Examples
    --------
    >>> from hmmlearn.hmm import GaussianHMM
    >>> GaussianHMM(n_components=2)  #doctest: +ELLIPSIS
    GaussianHMM(algorithm='viterbi',...
    """
    def __init__(self, n_components=1, num_frames=10, covariance_type='diag',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc"):
        self.params = params
        self.init_params = init_params
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)

        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.transmat_ = make_transition_matrix(num_frames=10, num_states=3)
        self.n_samples = num_frames
        self.n_components = n_components
        self.n_possibilities = num_frames * n_components
        self.checked = False

    # @property
    # def covars_(self):
    #     """Return covars as a full matrix."""
    #     return fill_covars(self._covars_, self.covariance_type,
    #                        self.n_components, self.n_features)

    #         # _utils.check_is_fitted(self, "startprob_")  # this is useless
         # self._check()

    def score(self, X, lengths=None):
        import pdb; pdb.set_trace()
        if not self.checked:
            self.check_all()
        logprob = 0
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij, _fwdlattice = self._do_foreward_pass(framelogprob)
            logprob += logprobij
        return logprob

    # compute posterior log probability of a sample under a model
    def score_samples(self, X, lengths=None):
        logprob = 0
        posteriors = np.zeros((self.n_samples, self.n_possibilities))
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij, fwdlattice = self._do_forward_pass(framelogprob)
            logprob += logprobij

            bwdlattice = self._do_backward_pass(framelogprob)
        posteriors[i:j] = self._compute_posteriors(fwdlattice, bwdlattice)
        return logprob, posteriors

    # compute the log probability of a video under a model, ONLY using the fwd lattice
    def score_samples_fwdlattice(self, X, lengths=None):
        framelogprob = self._compute_log_likelihood(X)
        logprob, fwdlattice = self._do_forward_pass(framelogprob)
        with np.errstate(under="ignore"):
            # convert to probability space and normalize
            fwdlattice = np.exp(fwdlattice)
            fwdlattice /= fwdlattice.sum(axis=1)[:, np.newaxis]
        
        return logprob, fwdlattice

    # working
    # predict only what the hmm thinks of that frame
    def predict_proba_simple(self, X, lengths=None):
        all_model_preds = list()
        for value in X:
            model_preds = list()
            for mean, covar in zip(self.means_, self.covars_):
                dist = norm(loc=mean, scale=covar)
                try:
                    prediction = dist.pdf(value)
                except:
                    import pdb; pdb.set_trace()
                    pass
                model_preds.append(prediction)
            all_model_preds.append(model_preds)
        all_model_preds = np.array(all_model_preds)
        all_model_preds = np.squeeze(all_model_preds)  # (10, 3)
        all_model_preds /= all_model_preds.sum(axis=1)[:, np.newaxis]
        return all_model_preds
                
                                   

    def predict_proba(self, X, lengths=None):
        # these are probabilities, not logs
        _, posteriors = self.score_samples_fwdlattice(X, lengths)
        # now let's change from (10, 30) to (10, 3)
        condensed_probs = np.zeros((10, 3))
        condensed_probs[:, 0] = posteriors[:, :10].sum(axis=1)
        condensed_probs[:, 1] = posteriors[:, 10:20].sum(axis=1)
        condensed_probs[:, 2] = posteriors[:, 20:].sum(axis=1)
        return condensed_probs

    # train a model
    def fit(self, X, lengths=None):
        import pdb; pdb.set_trace()
        self._init(X, lengths=lengths)

        self.monitor_._reset()
        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for i, j in iter_from_X_lengths(X, lengths):
                framelogprob = self._compute_log_likelihood(X[i:j])
                logprob, fwdlattice = self._do_forward_pass(framelogprob)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self._accumulate_sufficient_statistics(
                    stats, X[i:j], framelogprob, posteriors, fwdlattice,
                    bwdlattice)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)

            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break

        return self
    
    def predict(self, X, lengths=None):
        import pdb; pdb.set_trace()
        _, state_sequence = self._decode(X, lengths)
        return state_sequence

    def _decode(X, lengths):
        self._check()
        # only supports viterbi algorithm!
        X = check_array(X)
        n_samples = X.shape[0]
        logprob = 0
        state_sequence = np.empty(n_samples, dtype=int)
        for i, j in iter_from_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            # these two lines used to be _decode_viterbi in base.py
            framelogprob = self._compute_log_likelihood(X[i:j])
            self._do_viterbi_pass(framelogprob)
            
            
            logprobij, state_sequenceij = decoder(X[i:j])
            logprob += logprobij
            state_sequence[i:j] = state_sequenceij

        return logprob, state_sequence
    

    def _do_viterbi_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        log_startprob = log_mask_zero(self.startprob_)
        log_transmat = log_mask_zero(self.transmat_)

        n_samples = self.n_samples
        n_components = self.n_components
        state_sequence = \
        np.empty(n_samples, dtype=np.int32)
        n_possibilities = n_samples * n_components
        viterbi_lattice = \
            np.zeros((n_samples, n_possibilities))
        work_buffer = np.empty(n_possibilities)

        viterbi_lattice[0] = log_startprob  # only include startprob because -np.inf overwhelms everything else

        # Induction
        for t in range(1, n_samples):
            for i in range(n_possibilities):
                for j in range(n_possibilities):
                    work_buffer[j] = (log_transmat[j, i]
                                      + viterbi_lattice[t - 1, j])

                viterbi_lattice[t, i] = np.max(work_buffer) + framelogprob[t, i]

        # Observation traceback
        state_sequence[-1] = where_from = \
            np.argmax(viterbi_lattice[-1])
        logprob = viterbi_lattice[-1, where_from]

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_possibilities):
                work_buffer[i] = (viterbi_lattice[t, i]
                                  + log_transmat[i, where_from])

            state_sequence[t] = where_from = np.argmax(work_buffer)
        return logprob, np.asarray(state_sequence)

    def _compute_posteriors(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        log_normalize(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)

    def get_fwd_lattice(self, X, lengths=None):
        framelogprob = self._compute_log_likelihood(X)
        logprobij, fwdlattice = self._do_forward_pass(framelogprob)
        return fwdlattice


    # @covars_.setter
    # def covars_(self, covars):
    #     import pdb; pdb.set_trace()
    #     covars = np.array(covars, copy=True)
    #     _utils._validate_covars(covars, self.covariance_type,
    #                             self.n_components)
    #     self._covars_ = covars

    def check_all(self, X):
        # make sure we have everything
        assert hasattr(self, 'startprob_')
        assert hasattr(self, 'transmat_')
        assert hasattr(self, 'covars_')
        assert hasattr(self, 'means_')
        
        # make sure everything is the right shape
        assert self.startprob_.shape == (self.n_possibilities,)
        assert np.allclose(self.startprob_.sum(), 1.0)
        assert self.transmat_.shape == (self.n_possibilities, self.n_possibilities)
        assert np.allclose(self.transmat_.sum(), self.n_possibilities)

        # other random shit
        self.n_features = self.means_.shape[1]
        assert self.covariance_type in COVARIANCE_TYPES

        # If you don't give me legit inputs I will end you
        checked_x = checkarray(X)
        assert np.arrayequal(checked_x, X)

        
    def _get_n_fit_scalars_per_param(self):
        import pdb; pdb.set_trace()
        nc = self.n_components
        nf = self.n_features
        return {
            "s": nc - 1,
            "t": nc * (nc - 1),
            "m": nc * nf,
            "c": {
                "spherical": nc,
                "diag": nc * nf,
                "full": nc * nf * (nf + 1) // 2,
                "tied": nf * (nf + 1) // 2,
            }[self.covariance_type],
        }

    def _init(self, X, lengths=None):
        # look into this?
        _check_and_set_gaussian_n_features(self, X)

        if 'm' in self.init_params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        if 'c' in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = \
                _utils.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components).copy()

    def _compute_log_likelihood(self, X):
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    # def _generate_sample_from_state(self, state, random_state=None):
    #     random_state = check_random_state(random_state)
    #     return random_state.multivariate_normal(
    #         self.means_[state], self.covars_[state]
    #     )

    def _initialize_sufficient_statistics(self):
        # stats = super()._initialize_sufficient_statistics()  # for startprob, transmat
        stats = dict({'nobs': 0})
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        # super()._accumulate_sufficient_statistics(
        #     stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)  # to update startprob, transmat
        stats['nobs'] += 1

        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)

    # returns logprob of last row, fwdlattice
    def _do_forward_pass(self, framelogprob):
        log_startprob = log_mask_zero(self.startprob_)
        log_transmat = log_mask_zero(self.transmat_)

        fwdlattice = np.zeros((self.n_samples, self.n_possibilities))
        fwdlattice[0] = log_startprob
        work_buffer = np.zeros(self.n_possibilities)
        
        for t in range(1, self.n_samples):
            for to_idx in range(self.n_possibilities):
                for from_idx in range(self.n_possibilities):
                    work_buffer[from_idx] = fwdlattice[t - 1, from_idx] + log_transmat[to_idx, from_idx]
                to_hidden_state = int(to_idx // 10)
                fwdlattice[t, to_idx] = logsumexp(work_buffer) + framelogprob[t, to_hidden_state]

        with np.errstate(under="ignore"):
            return logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob):
        log_startprob = log_mask_zero(self.startprob_)
        log_transmat = log_mask_zero(self.transmat_)
        bwdlattice = np.zeros((self.n_samples, self.n_possibilities))
        work_buffer = np.zeros((self.n_possibilities, self.n_samples))

        # Set uniform probability of spending 1-7 timesteps in the final state
        bwdlattice[-1] = -np.inf
        # states 20-29 represent hidden state 2
        bwdlattice[-1, 20:27] = np.log(1./7)

        for t in range(self.n_samples - 2, -1, -1):
            for from_idx in range(self.n_possibilities):
                for to_idx in range(self.n_possibilities):
                    to_hidden_state = to_idx // 10
                    save_value = (log_transmat[to_idx, from_idx]
                                           + framelogprob[t + 1, to_hidden_state]
                                           + bwdlattice[t + 1, to_idx])
                    work_buffer[to_idx] = (log_transmat[to_idx, from_idx]
                                           + framelogprob[t + 1, to_hidden_state]
                                           + bwdlattice[t + 1, to_idx])
                bwdlattice[t, from_idx] = logsumexp(work_buffer)
        return bwdlattice


    def _do_mstep(self, stats):
        # super()._do_mstep(stats)  # necessary if I'm updating startprob or transmat

        means_prior = self.means_prior
        means_weight = self.means_weight

        denom = stats['post'][:, np.newaxis]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))

        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type in ('spherical', 'diag'):
                cv_num = (means_weight * meandiff**2
                          + stats['obs**2']
                          - 2 * self.means_ * stats['obs']
                          + self.means_**2 * denom)
                cv_den = max(covars_weight - 1, 0) + denom
                self._covars_ = \
                    (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
                if self.covariance_type == 'spherical':
                    self._covars_ = np.tile(
                        self._covars_.mean(1)[:, np.newaxis],
                        (1, self._covars_.shape[1]))
            elif self.covariance_type in ('tied', 'full'):
                cv_num = np.empty((self.n_components, self.n_features,
                                  self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self.means_[c])

                    cv_num[c] = (means_weight * np.outer(meandiff[c],
                                                         meandiff[c])
                                 + stats['obs*obs.T'][c]
                                 - obsmean - obsmean.T
                                 + np.outer(self.means_[c], self.means_[c])
                                 * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self.covariance_type == 'tied':
                    self._covars_ = ((covars_prior + cv_num.sum(axis=0)) /
                                     (cvweight + stats['post'].sum()))
                elif self.covariance_type == 'full':
                    self._covars_ = ((covars_prior + cv_num) /
                                     (cvweight + stats['post'][:, None, None]))



# class GaussianHMM(_BaseHMM):
#     r"""Hidden Markov Model with Gaussian emissions.

#     Parameters
#     ----------
#     n_components : int
#         Number of states.

#     covariance_type : string, optional
#         String describing the type of covariance parameters to
#         use.  Must be one of

#         * "spherical" --- each state uses a single variance value that
#           applies to all features.
#         * "diag" --- each state uses a diagonal covariance matrix.
#         * "full" --- each state uses a full (i.e. unrestricted)
#           covariance matrix.
#         * "tied" --- all states use **the same** full covariance matrix.

#         Defaults to "diag".

#     min_covar : float, optional
#         Floor on the diagonal of the covariance matrix to prevent
#         overfitting. Defaults to 1e-3.

#     startprob_prior : array, shape (n_components, ), optional
#         Parameters of the Dirichlet prior distribution for
#         :attr:`startprob_`.

#     transmat_prior : array, shape (n_components, n_components), optional
#         Parameters of the Dirichlet prior distribution for each row
#         of the transition probabilities :attr:`transmat_`.

#     means_prior, means_weight : array, shape (n_components, ), optional
#         Mean and precision of the Normal prior distribtion for
#         :attr:`means_`.

#     covars_prior, covars_weight : array, shape (n_components, ), optional
#         Parameters of the prior distribution for the covariance matrix
#         :attr:`covars_`.

#         If :attr:`covariance_type` is "spherical" or "diag" the prior is
#         the inverse gamma distribution, otherwise --- the inverse Wishart
#         distribution.

#     algorithm : string, optional
#         Decoder algorithm. Must be one of "viterbi" or`"map".
#         Defaults to "viterbi".

#     random_state: RandomState or an int seed, optional
#         A random number generator instance.

#     n_iter : int, optional
#         Maximum number of iterations to perform.

#     tol : float, optional
#         Convergence threshold. EM will stop if the gain in log-likelihood
#         is below this value.

#     verbose : bool, optional
#         When ``True`` per-iteration convergence reports are printed
#         to :data:`sys.stderr`. You can diagnose convergence via the
#         :attr:`monitor_` attribute.

#     params : string, optional
#         Controls which parameters are updated in the training
#         process.  Can contain any combination of 's' for startprob,
#         't' for transmat, 'm' for means and 'c' for covars. Defaults
#         to all parameters.

#     init_params : string, optional
#         Controls which parameters are initialized prior to
#         training.  Can contain any combination of 's' for
#         startprob, 't' for transmat, 'm' for means and 'c' for covars.
#         Defaults to all parameters.

#     Attributes
#     ----------
#     n_features : int
#         Dimensionality of the Gaussian emissions.

#     monitor\_ : ConvergenceMonitor
#         Monitor object used to check the convergence of EM.

#     startprob\_ : array, shape (n_components, )
#         Initial state occupation distribution.

#     transmat\_ : array, shape (n_components, n_components)
#         Matrix of transition probabilities between states.

#     means\_ : array, shape (n_components, n_features)
#         Mean parameters for each state.

#     covars\_ : array
#         Covariance parameters for each state.

#         The shape depends on :attr:`covariance_type`::

#             (n_components, )                        if "spherical",
#             (n_components, n_features)              if "diag",
#             (n_components, n_features, n_features)  if "full"
#             (n_features, n_features)                if "tied",

#     Examples
#     --------
#     >>> from hmmlearn.hmm import GaussianHMM
#     >>> GaussianHMM(n_components=2)  #doctest: +ELLIPSIS
#     GaussianHMM(algorithm='viterbi',...
#     """
#     def __init__(self, n_components=1, covariance_type='diag',
#                  min_covar=1e-3,
#                  startprob_prior=1.0, transmat_prior=1.0,
#                  means_prior=0, means_weight=0,
#                  covars_prior=1e-2, covars_weight=1,
#                  algorithm="viterbi", random_state=None,
#                  n_iter=10, tol=1e-2, verbose=False,
#                  params="stmc", init_params="stmc"):
#         _BaseHMM.__init__(self, n_components,
#                           startprob_prior=startprob_prior,
#                           transmat_prior=transmat_prior, algorithm=algorithm,
#                           random_state=random_state, n_iter=n_iter,
#                           tol=tol, params=params, verbose=verbose,
#                           init_params=init_params)

#         self.covariance_type = covariance_type
#         self.min_covar = min_covar
#         self.means_prior = means_prior
#         self.means_weight = means_weight
#         self.covars_prior = covars_prior
#         self.covars_weight = covars_weight

#     @property
#     def covars_(self):
#         """Return covars as a full matrix."""
#         return fill_covars(self._covars_, self.covariance_type,
#                            self.n_components, self.n_features)

#     @covars_.setter
#     def covars_(self, covars):
#         covars = np.array(covars, copy=True)
#         _utils._validate_covars(covars, self.covariance_type,
#                                 self.n_components)
#         self._covars_ = covars

#     def _check(self):
#         super()._check()

#         self.means_ = np.asarray(self.means_)
#         self.n_features = self.means_.shape[1]

#         if self.covariance_type not in COVARIANCE_TYPES:
#             raise ValueError('covariance_type must be one of {}'
#                              .format(COVARIANCE_TYPES))

#     def _get_n_fit_scalars_per_param(self):
#         nc = self.n_components
#         nf = self.n_features
#         return {
#             "s": nc - 1,
#             "t": nc * (nc - 1),
#             "m": nc * nf,
#             "c": {
#                 "spherical": nc,
#                 "diag": nc * nf,
#                 "full": nc * nf * (nf + 1) // 2,
#                 "tied": nf * (nf + 1) // 2,
#             }[self.covariance_type],
#         }

#     def _init(self, X, lengths=None):
#         _check_and_set_gaussian_n_features(self, X)
#         super()._init(X, lengths=lengths)

#         if 'm' in self.init_params or not hasattr(self, "means_"):
#             kmeans = cluster.KMeans(n_clusters=self.n_components,
#                                     random_state=self.random_state)
#             kmeans.fit(X)
#             self.means_ = kmeans.cluster_centers_
#         if 'c' in self.init_params or not hasattr(self, "covars_"):
#             cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
#             if not cv.shape:
#                 cv.shape = (1, 1)
#             self.covars_ = \
#                 _utils.distribute_covar_matrix_to_match_covariance_type(
#                     cv, self.covariance_type, self.n_components).copy()
        
#     def _compute_log_likelihood(self, X):
#         return log_multivariate_normal_density(
#             X, self.means_, self._covars_, self.covariance_type)

#     def _generate_sample_from_state(self, state, random_state=None):
#         random_state = check_random_state(random_state)
#         return random_state.multivariate_normal(
#             self.means_[state], self.covars_[state]
#         )

#     def _initialize_sufficient_statistics(self):
#         stats = super()._initialize_sufficient_statistics()
#         stats['post'] = np.zeros(self.n_components)
#         stats['obs'] = np.zeros((self.n_components, self.n_features))
#         stats['obs**2'] = np.zeros((self.n_components, self.n_features))
#         if self.covariance_type in ('tied', 'full'):
#             stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
#                                            self.n_features))
#         return stats

#     def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
#                                           posteriors, fwdlattice, bwdlattice):
#         super()._accumulate_sufficient_statistics(
#             stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

#         if 'm' in self.params or 'c' in self.params:
#             stats['post'] += posteriors.sum(axis=0)
#             stats['obs'] += np.dot(posteriors.T, obs)

#         if 'c' in self.params:
#             if self.covariance_type in ('spherical', 'diag'):
#                 stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
#             elif self.covariance_type in ('tied', 'full'):
#                 # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
#                 # -> (nc, nf, nf)
#                 stats['obs*obs.T'] += np.einsum(
#                     'ij,ik,il->jkl', posteriors, obs, obs)

#     def _do_mstep(self, stats):
#         super()._do_mstep(stats)

#         means_prior = self.means_prior
#         means_weight = self.means_weight

#         # TODO: find a proper reference for estimates for different
#         #       covariance models.
#         # Based on Huang, Acero, Hon, "Spoken Language Processing",
#         # p. 443 - 445
#         denom = stats['post'][:, np.newaxis]
#         if 'm' in self.params:
#             self.means_ = ((means_weight * means_prior + stats['obs'])
#                            / (means_weight + denom))

#         if 'c' in self.params:
#             covars_prior = self.covars_prior
#             covars_weight = self.covars_weight
#             meandiff = self.means_ - means_prior

#             if self.covariance_type in ('spherical', 'diag'):
#                 cv_num = (means_weight * meandiff**2
#                           + stats['obs**2']
#                           - 2 * self.means_ * stats['obs']
#                           + self.means_**2 * denom)
#                 cv_den = max(covars_weight - 1, 0) + denom
#                 self._covars_ = \
#                     (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
#                 if self.covariance_type == 'spherical':
#                     self._covars_ = np.tile(
#                         self._covars_.mean(1)[:, np.newaxis],
#                         (1, self._covars_.shape[1]))
#             elif self.covariance_type in ('tied', 'full'):
#                 cv_num = np.empty((self.n_components, self.n_features,
#                                   self.n_features))
#                 for c in range(self.n_components):
#                     obsmean = np.outer(stats['obs'][c], self.means_[c])

#                     cv_num[c] = (means_weight * np.outer(meandiff[c],
#                                                          meandiff[c])
#                                  + stats['obs*obs.T'][c]
#                                  - obsmean - obsmean.T
#                                  + np.outer(self.means_[c], self.means_[c])
#                                  * stats['post'][c])
#                 cvweight = max(covars_weight - self.n_features, 0)
#                 if self.covariance_type == 'tied':
#                     self._covars_ = ((covars_prior + cv_num.sum(axis=0)) /
#                                      (cvweight + stats['post'].sum()))
#                 elif self.covariance_type == 'full':
#                     self._covars_ = ((covars_prior + cv_num) /
#                                      (cvweight + stats['post'][:, None, None]))


# class MultinomialHMM(_BaseHMM):
#     r"""Hidden Markov Model with multinomial (discrete) emissions

#     Parameters
#     ----------

#     n_components : int
#         Number of states.

#     startprob_prior : array, shape (n_components, ), optional
#         Parameters of the Dirichlet prior distribution for
#         :attr:`startprob_`.

#     transmat_prior : array, shape (n_components, n_components), optional
#         Parameters of the Dirichlet prior distribution for each row
#         of the transition probabilities :attr:`transmat_`.

#     algorithm : string, optional
#         Decoder algorithm. Must be one of "viterbi" or "map".
#         Defaults to "viterbi".

#     random_state: RandomState or an int seed, optional
#         A random number generator instance.

#     n_iter : int, optional
#         Maximum number of iterations to perform.

#     tol : float, optional
#         Convergence threshold. EM will stop if the gain in log-likelihood
#         is below this value.

#     verbose : bool, optional
#         When ``True`` per-iteration convergence reports are printed
#         to :data:`sys.stderr`. You can diagnose convergence via the
#         :attr:`monitor_` attribute.

#     params : string, optional
#         Controls which parameters are updated in the training
#         process.  Can contain any combination of 's' for startprob,
#         't' for transmat, 'e' for emissionprob.
#         Defaults to all parameters.

#     init_params : string, optional
#         Controls which parameters are initialized prior to
#         training.  Can contain any combination of 's' for
#         startprob, 't' for transmat, 'e' for emissionprob.
#         Defaults to all parameters.

#     Attributes
#     ----------
#     n_features : int
#         Number of possible symbols emitted by the model (in the samples).

#     monitor\_ : ConvergenceMonitor
#         Monitor object used to check the convergence of EM.

#     startprob\_ : array, shape (n_components, )
#         Initial state occupation distribution.

#     transmat\_ : array, shape (n_components, n_components)
#         Matrix of transition probabilities between states.

#     emissionprob\_ : array, shape (n_components, n_features)
#         Probability of emitting a given symbol when in each state.

#     Examples
#     --------
#     >>> from hmmlearn.hmm import MultinomialHMM
#     >>> MultinomialHMM(n_components=2)  #doctest: +ELLIPSIS
#     MultinomialHMM(algorithm='viterbi',...
#     """
#     # TODO: accept the prior on emissionprob_ for consistency.
#     def __init__(self, n_components=1,
#                  startprob_prior=1.0, transmat_prior=1.0,
#                  algorithm="viterbi", random_state=None,
#                  n_iter=10, tol=1e-2, verbose=False,
#                  params="ste", init_params="ste"):
#         _BaseHMM.__init__(self, n_components,
#                           startprob_prior=startprob_prior,
#                           transmat_prior=transmat_prior,
#                           algorithm=algorithm,
#                           random_state=random_state,
#                           n_iter=n_iter, tol=tol, verbose=verbose,
#                           params=params, init_params=init_params)

#     def _get_n_fit_scalars_per_param(self):
#         nc = self.n_components
#         nf = self.n_features
#         return {
#             "s": nc - 1,
#             "t": nc * (nc - 1),
#             "e": nc * (nf - 1),
#         }

#     def _init(self, X, lengths=None):
#         self._check_and_set_n_features(X)
#         super()._init(X, lengths=lengths)
#         self.random_state = check_random_state(self.random_state)

#         if 'e' in self.init_params:
#             self.emissionprob_ = self.random_state \
#                 .rand(self.n_components, self.n_features)
#             normalize(self.emissionprob_, axis=1)

#     def _check(self):
#         super()._check()

#         self.emissionprob_ = np.atleast_2d(self.emissionprob_)
#         n_features = getattr(self, "n_features", self.emissionprob_.shape[1])
#         if self.emissionprob_.shape != (self.n_components, n_features):
#             raise ValueError(
#                 "emissionprob_ must have shape (n_components, n_features)")
#         else:
#             self.n_features = n_features

#     def _compute_log_likelihood(self, X):
#         return log_mask_zero(self.emissionprob_)[:, np.concatenate(X)].T

#     def _generate_sample_from_state(self, state, random_state=None):
#         cdf = np.cumsum(self.emissionprob_[state, :])
#         random_state = check_random_state(random_state)
#         return [(cdf > random_state.rand()).argmax()]

#     def _initialize_sufficient_statistics(self):
#         stats = super()._initialize_sufficient_statistics()
#         stats['obs'] = np.zeros((self.n_components, self.n_features))
#         return stats

#     def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
#                                           posteriors, fwdlattice, bwdlattice):
#         super()._accumulate_sufficient_statistics(
#             stats, X, framelogprob, posteriors, fwdlattice, bwdlattice)
#         if 'e' in self.params:
#             for t, symbol in enumerate(np.concatenate(X)):
#                 stats['obs'][:, symbol] += posteriors[t]

#     def _do_mstep(self, stats):
#         super()._do_mstep(stats)
#         if 'e' in self.params:
#             self.emissionprob_ = (stats['obs']
#                                   / stats['obs'].sum(axis=1)[:, np.newaxis])

#     def _check_and_set_n_features(self, X):
#         """
#         Check if ``X`` is a sample from a Multinomial distribution, i.e. an
#         array of non-negative integers.
#         """
#         if not np.issubdtype(X.dtype, np.integer):
#             raise ValueError("Symbols should be integers")
#         if X.min() < 0:
#             raise ValueError("Symbols should be nonnegative")
#         if hasattr(self, "n_features"):
#             if self.n_features - 1 < X.max():
#                 raise ValueError(
#                     "Largest symbol is {} but the model only emits "
#                     "symbols up to {}"
#                     .format(X.max(), self.n_features - 1))
#         self.n_features = X.max() + 1


# class GMMHMM(_BaseHMM):
#     r"""Hidden Markov Model with Gaussian mixture emissions.

#     Parameters
#     ----------
#     n_components : int
#         Number of states in the model.

#     n_mix : int
#         Number of states in the GMM.

#     covariance_type : string, optional
#         String describing the type of covariance parameters to
#         use.  Must be one of

#         * "spherical" --- each state uses a single variance value that
#           applies to all features.
#         * "diag" --- each state uses a diagonal covariance matrix.
#         * "full" --- each state uses a full (i.e. unrestricted)
#           covariance matrix.
#         * "tied" --- all mixture components of each state use **the same** full
#           covariance matrix (note that this is not the same as for
#           `GaussianHMM`).

#         Defaults to "diag".

#     min_covar : float, optional
#         Floor on the diagonal of the covariance matrix to prevent
#         overfitting. Defaults to 1e-3.

#     startprob_prior : array, shape (n_components, ), optional
#         Parameters of the Dirichlet prior distribution for
#         :attr:`startprob_`.

#     transmat_prior : array, shape (n_components, n_components), optional
#         Parameters of the Dirichlet prior distribution for each row
#         of the transition probabilities :attr:`transmat_`.

#     weights_prior : array, shape (n_mix, ), optional
#         Parameters of the Dirichlet prior distribution for
#         :attr:`weights_`.

#     means_prior, means_weight : array, shape (n_mix, ), optional
#         Mean and precision of the Normal prior distribtion for
#         :attr:`means_`.

#     covars_prior, covars_weight : array, shape (n_mix, ), optional
#         Parameters of the prior distribution for the covariance matrix
#         :attr:`covars_`.

#         If :attr:`covariance_type` is "spherical" or "diag" the prior is
#         the inverse gamma distribution, otherwise --- the inverse Wishart
#         distribution.

#     algorithm : string, optional
#         Decoder algorithm. Must be one of "viterbi" or "map".
#         Defaults to "viterbi".

#     random_state: RandomState or an int seed, optional
#         A random number generator instance.

#     n_iter : int, optional
#         Maximum number of iterations to perform.

#     tol : float, optional
#         Convergence threshold. EM will stop if the gain in log-likelihood
#         is below this value.

#     verbose : bool, optional
#         When ``True`` per-iteration convergence reports are printed
#         to :data:`sys.stderr`. You can diagnose convergence via the
#         :attr:`monitor_` attribute.

#     init_params : string, optional
#         Controls which parameters are initialized prior to training. Can
#         contain any combination of 's' for startprob, 't' for transmat, 'm'
#         for means, 'c' for covars, and 'w' for GMM mixing weights.
#         Defaults to all parameters.

#     params : string, optional
#         Controls which parameters are updated in the training process.  Can
#         contain any combination of 's' for startprob, 't' for transmat, 'm' for
#         means, and 'c' for covars, and 'w' for GMM mixing weights.
#         Defaults to all parameters.

#     Attributes
#     ----------
#     monitor\_ : ConvergenceMonitor
#         Monitor object used to check the convergence of EM.

#     startprob\_ : array, shape (n_components, )
#         Initial state occupation distribution.

#     transmat\_ : array, shape (n_components, n_components)
#         Matrix of transition probabilities between states.

#     weights\_ : array, shape (n_components, n_mix)
#         Mixture weights for each state.

#     means\_ : array, shape (n_components, n_mix)
#         Mean parameters for each mixture component in each state.

#     covars\_ : array
#         Covariance parameters for each mixture components in each state.

#         The shape depends on :attr:`covariance_type`::

#             (n_components, n_mix)                          if "spherical",
#             (n_components, n_mix, n_features)              if "diag",
#             (n_components, n_mix, n_features, n_features)  if "full"
#             (n_components, n_features, n_features)         if "tied",
#     """

#     def __init__(self, n_components=1, n_mix=1,
#                  min_covar=1e-3, startprob_prior=1.0, transmat_prior=1.0,
#                  weights_prior=1.0, means_prior=0.0, means_weight=0.0,
#                  covars_prior=None, covars_weight=None,
#                  algorithm="viterbi", covariance_type="diag",
#                  random_state=None, n_iter=10, tol=1e-2,
#                  verbose=False, params="stmcw",
#                  init_params="stmcw"):
#         _BaseHMM.__init__(self, n_components,
#                           startprob_prior=startprob_prior,
#                           transmat_prior=transmat_prior,
#                           algorithm=algorithm, random_state=random_state,
#                           n_iter=n_iter, tol=tol, verbose=verbose,
#                           params=params, init_params=init_params)
#         self.covariance_type = covariance_type
#         self.min_covar = min_covar
#         self.n_mix = n_mix
#         self.weights_prior = weights_prior
#         self.means_prior = means_prior
#         self.means_weight = means_weight
#         self.covars_prior = covars_prior
#         self.covars_weight = covars_weight

#     def _get_n_fit_scalars_per_param(self):
#         nc = self.n_components
#         nf = self.n_features
#         nm = self.n_mix
#         return {
#             "s": nc - 1,
#             "t": nc * (nc - 1),
#             "m": nc * nm * nf,
#             "c": {
#                 "spherical": nc * nm,
#                 "diag": nc * nm * nf,
#                 "full": nc * nm * nf * (nf + 1) // 2,
#                 "tied": nc * nf * (nf + 1) // 2,
#             }[self.covariance_type],
#             "w": nm - 1,
#         }

#     def _init(self, X, lengths=None):
#         _check_and_set_gaussian_n_features(self, X)
#         super()._init(X, lengths=lengths)
#         nc = self.n_components
#         nf = self.n_features
#         nm = self.n_mix

#         # Default values for covariance prior parameters
#         self._init_covar_priors()
#         self._fix_priors_shape()

#         main_kmeans = cluster.KMeans(n_clusters=nc,
#                                      random_state=self.random_state)
#         labels = main_kmeans.fit_predict(X)
#         kmeanses = []
#         for label in range(nc):
#             kmeans = cluster.KMeans(n_clusters=nm,
#                                     random_state=self.random_state)
#             kmeans.fit(X[np.where(labels == label)])
#             kmeanses.append(kmeans)

#         if 'w' in self.init_params or not hasattr(self, "weights_"):
#             self.weights_ = np.ones((nc, nm)) / (np.ones((nc, 1)) * nm)

#         if 'm' in self.init_params or not hasattr(self, "means_"):
#             self.means_ = np.stack(
#                 [kmeans.cluster_centers_ for kmeans in kmeanses])

#         if 'c' in self.init_params or not hasattr(self, "covars_"):
#             cv = np.cov(X.T) + self.min_covar * np.eye(nf)
#             if not cv.shape:
#                 cv.shape = (1, 1)
#             if self.covariance_type == 'tied':
#                 self.covars_ = np.zeros((nc, nf, nf))
#                 self.covars_[:] = cv
#             elif self.covariance_type == 'full':
#                 self.covars_ = np.zeros((nc, nm, nf, nf))
#                 self.covars_[:] = cv
#             elif self.covariance_type == 'diag':
#                 self.covars_ = np.zeros((nc, nm, nf))
#                 self.covars_[:] = np.diag(cv)
#             elif self.covariance_type == 'spherical':
#                 self.covars_ = np.zeros((nc, nm))
#                 self.covars_[:] = cv.mean()

#     def _init_covar_priors(self):
#         if self.covariance_type == "full":
#             if self.covars_prior is None:
#                 self.covars_prior = 0.0
#             if self.covars_weight is None:
#                 self.covars_weight = -(1.0 + self.n_features + 1.0)
#         elif self.covariance_type == "tied":
#             if self.covars_prior is None:
#                 self.covars_prior = 0.0
#             if self.covars_weight is None:
#                 self.covars_weight = -(self.n_mix + self.n_features + 1.0)
#         elif self.covariance_type == "diag":
#             if self.covars_prior is None:
#                 self.covars_prior = -1.5
#             if self.covars_weight is None:
#                 self.covars_weight = 0.0
#         elif self.covariance_type == "spherical":
#             if self.covars_prior is None:
#                 self.covars_prior = -(self.n_mix + 2.0) / 2.0
#             if self.covars_weight is None:
#                 self.covars_weight = 0.0

#     def _fix_priors_shape(self):
#         nc = self.n_components
#         nf = self.n_features
#         nm = self.n_mix

#         # If priors are numbers, this function will make them into a
#         # matrix of proper shape
#         self.weights_prior = np.broadcast_to(
#             self.weights_prior, (nc, nm)).copy()
#         self.means_prior = np.broadcast_to(
#             self.means_prior, (nc, nm, nf)).copy()
#         self.means_weight = np.broadcast_to(
#             self.means_weight, (nc, nm)).copy()

#         if self.covariance_type == "full":
#             self.covars_prior = np.broadcast_to(
#                 self.covars_prior, (nc, nm, nf, nf)).copy()
#             self.covars_weight = np.broadcast_to(
#                 self.covars_weight, (nc, nm)).copy()
#         elif self.covariance_type == "tied":
#             self.covars_prior = np.broadcast_to(
#                 self.covars_prior, (nc, nf, nf)).copy()
#             self.covars_weight = np.broadcast_to(
#                 self.covars_weight, nc).copy()
#         elif self.covariance_type == "diag":
#             self.covars_prior = np.broadcast_to(
#                 self.covars_prior, (nc, nm, nf)).copy()
#             self.covars_weight = np.broadcast_to(
#                 self.covars_weight, (nc, nm, nf)).copy()
#         elif self.covariance_type == "spherical":
#             self.covars_prior = np.broadcast_to(
#                 self.covars_prior, (nc, nm)).copy()
#             self.covars_weight = np.broadcast_to(
#                 self.covars_weight, (nc, nm)).copy()

#     def _check(self):
#         super()._check()
#         if not hasattr(self, "n_features"):
#             self.n_features = self.means_.shape[2]
#         nc = self.n_components
#         nf = self.n_features
#         nm = self.n_mix

#         self._init_covar_priors()
#         self._fix_priors_shape()

#         # Checking covariance type
#         if self.covariance_type not in COVARIANCE_TYPES:
#             raise ValueError("covariance_type must be one of {}"
#                              .format(COVARIANCE_TYPES))

#         self.weights_ = np.array(self.weights_)
#         # Checking mixture weights' shape
#         if self.weights_.shape != (nc, nm):
#             raise ValueError("mixture weights must have shape "
#                              "(n_components, n_mix), actual shape: {}"
#                              .format(self.weights_.shape))

#         # Checking mixture weights' mathematical correctness
#         if not np.allclose(np.sum(self.weights_, axis=1), np.ones(nc)):
#             raise ValueError("mixture weights must sum up to 1")

#         # Checking means' shape
#         self.means_ = np.array(self.means_)
#         if self.means_.shape != (nc, nm, nf):
#             raise ValueError("mixture means must have shape "
#                              "(n_components, n_mix, n_features), "
#                              "actual shape: {}".format(self.means_.shape))

#         # Checking covariances' shape
#         self.covars_ = np.array(self.covars_)
#         covars_shape = self.covars_.shape
#         needed_shapes = {
#             "spherical": (nc, nm),
#             "tied": (nc, nf, nf),
#             "diag": (nc, nm, nf),
#             "full": (nc, nm, nf, nf),
#         }
#         needed_shape = needed_shapes[self.covariance_type]
#         if covars_shape != needed_shape:
#             raise ValueError("{!r} mixture covars must have shape {}, "
#                              "actual shape: {}"
#                              .format(self.covariance_type,
#                                      needed_shape, covars_shape))

#         # Checking covariances' mathematical correctness
#         from scipy import linalg

#         if (self.covariance_type == "spherical" or
#                 self.covariance_type == "diag"):
#             if np.any(self.covars_ < 0):
#                 raise ValueError("{!r} mixture covars must be non-negative"
#                                  .format(self.covariance_type))
#             if np.any(self.covars_ == 0):
#                 _log.warning("Degenerate mixture covariance")
#         elif self.covariance_type == "tied":
#             for i, covar in enumerate(self.covars_):
#                 if not np.allclose(covar, covar.T):
#                     raise ValueError("Covariance of state #{} is not symmetric"
#                                      .format(i))
#                 min_eigvalsh = np.linalg.eigvalsh(covar).min()
#                 if min_eigvalsh < 0:
#                     raise ValueError("Covariance of state #{} is not positive "
#                                      "definite".format(i))
#                 if min_eigvalsh == 0:
#                     _log.warning("Covariance of state #%d has a null "
#                                  "eigenvalue.", i)
#         elif self.covariance_type == "full":
#             for i, mix_covars in enumerate(self.covars_):
#                 for j, covar in enumerate(mix_covars):
#                     if not np.allclose(covar, covar.T):
#                         raise ValueError(
#                             "Covariance of state #{}, mixture #{} is not "
#                             "symmetric".format(i, j))
#                     min_eigvalsh = np.linalg.eigvalsh(covar).min()
#                     if min_eigvalsh < 0:
#                         raise ValueError(
#                             "Covariance of state #{}, mixture #{} is not "
#                             "positive definite".format(i, j))
#                     if min_eigvalsh == 0:
#                         _log.warning("Covariance of state #%d, mixture #%d "
#                                      "has a null eigenvalue.", i, j)

#     def _generate_sample_from_state(self, state, random_state=None):
#         if random_state is None:
#             random_state = self.random_state
#         random_state = check_random_state(random_state)

#         cur_weights = self.weights_[state]
#         i_gauss = random_state.choice(self.n_mix, p=cur_weights)
#         if self.covariance_type == 'tied':
#             # self.covars_.shape == (n_components, n_features, n_features)
#             # shouldn't that be (n_mix, ...)?
#             covs = self.covars_
#         else:
#             covs = self.covars_[:, i_gauss]
#             covs = fill_covars(covs, self.covariance_type,
#                                self.n_components, self.n_features)
#         return random_state.multivariate_normal(
#             self.means_[state, i_gauss], covs[state]
#         )

#     def _compute_log_weighted_gaussian_densities(self, X, i_comp):
#         cur_means = self.means_[i_comp]
#         cur_covs = self.covars_[i_comp]
#         if self.covariance_type == 'spherical':
#             cur_covs = cur_covs[:, np.newaxis]
#         log_cur_weights = np.log(self.weights_[i_comp])

#         return log_multivariate_normal_density(
#             X, cur_means, cur_covs, self.covariance_type
#         ) + log_cur_weights

#     def _compute_log_likelihood(self, X):
#         n_samples, _ = X.shape
#         res = np.zeros((n_samples, self.n_components))

#         for i in range(self.n_components):
#             log_denses = self._compute_log_weighted_gaussian_densities(X, i)
#             with np.errstate(under="ignore"):
#                 res[:, i] = logsumexp(log_denses, axis=1)

#         return res

#     def _initialize_sufficient_statistics(self):
#         stats = super()._initialize_sufficient_statistics()
#         stats['n_samples'] = 0
#         stats['post_comp_mix'] = None
#         stats['post_mix_sum'] = np.zeros((self.n_components, self.n_mix))
#         stats['post_sum'] = np.zeros(self.n_components)
#         stats['samples'] = None
#         stats['centered'] = None
#         return stats

#     def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
#                                           post_comp, fwdlattice, bwdlattice):

#         # TODO: support multiple frames

#         super()._accumulate_sufficient_statistics(
#             stats, X, framelogprob, post_comp, fwdlattice, bwdlattice
#         )

#         n_samples, _ = X.shape

#         stats['n_samples'] = n_samples
#         stats['samples'] = X

#         post_mix = np.zeros((n_samples, self.n_components, self.n_mix))
#         for p in range(self.n_components):
#             log_denses = self._compute_log_weighted_gaussian_densities(X, p)
#             log_normalize(log_denses, axis=-1)
#             with np.errstate(under="ignore"):
#                 post_mix[:, p, :] = np.exp(log_denses)

#         with np.errstate(under="ignore"):
#             post_comp_mix = post_comp[:, :, np.newaxis] * post_mix
#         stats['post_comp_mix'] = post_comp_mix

#         stats['post_mix_sum'] = np.sum(post_comp_mix, axis=0)
#         stats['post_sum'] = np.sum(post_comp, axis=0)

#         stats['centered'] = X[:, np.newaxis, np.newaxis, :] - self.means_

#     def _do_mstep(self, stats):
#         super()._do_mstep(stats)
#         nc = self.n_components
#         nf = self.n_features
#         nm = self.n_mix

#         n_samples = stats['n_samples']

#         # Maximizing weights
#         alphas_minus_one = self.weights_prior - 1
#         new_weights_numer = stats['post_mix_sum'] + alphas_minus_one
#         new_weights_denom = (
#             stats['post_sum'] + np.sum(alphas_minus_one, axis=1)
#         )[:, np.newaxis]
#         new_weights = new_weights_numer / new_weights_denom

#         # Maximizing means
#         lambdas, mus = self.means_weight, self.means_prior
#         new_means_numer = (
#             np.einsum('ijk,il->jkl', stats['post_comp_mix'], stats['samples'])
#             + lambdas[:, :, np.newaxis] * mus
#         )
#         new_means_denom = (stats['post_mix_sum'] + lambdas)[:, :, np.newaxis]
#         new_means = new_means_numer / new_means_denom

#         # Maximizing covariances
#         centered_means = self.means_ - mus

#         if self.covariance_type == 'full':
#             centered = stats['centered'].reshape((n_samples, nc, nm, nf, 1))
#             centered_t = stats['centered'].reshape((n_samples, nc, nm, 1, nf))
#             centered_dots = centered * centered_t

#             psis_t = np.transpose(self.covars_prior, axes=(0, 1, 3, 2))
#             nus = self.covars_weight

#             centr_means_resh = centered_means.reshape((nc, nm, nf, 1))
#             centr_means_resh_t = centered_means.reshape((nc, nm, 1, nf))
#             centered_means_dots = centr_means_resh * centr_means_resh_t

#             new_cov_numer = (
#                 np.einsum(
#                     'ijk,ijklm->jklm', stats['post_comp_mix'], centered_dots)
#                 + psis_t
#                 + lambdas[:, :, np.newaxis, np.newaxis] * centered_means_dots
#             )
#             new_cov_denom = (
#                 stats['post_mix_sum'] + 1 + nus + nf + 1
#             )[:, :, np.newaxis, np.newaxis]
#             new_cov = new_cov_numer / new_cov_denom

#         elif self.covariance_type == 'diag':
#             centered2 = stats['centered'] ** 2
#             centered_means2 = centered_means ** 2

#             alphas = self.covars_prior
#             betas = self.covars_weight

#             new_cov_numer = (
#                 np.einsum('ijk,ijkl->jkl', stats['post_comp_mix'], centered2)
#                 + lambdas[:, :, np.newaxis] * centered_means2
#                 + 2 * betas
#             )
#             new_cov_denom = (
#                 stats['post_mix_sum'][:, :, np.newaxis] + 1 + 2 * (alphas + 1)
#             )
#             new_cov = new_cov_numer / new_cov_denom

#         elif self.covariance_type == 'spherical':
#             centered_norm2 = np.sum(stats['centered'] ** 2, axis=-1)

#             alphas = self.covars_prior
#             betas = self.covars_weight

#             centered_means_norm2 = np.sum(centered_means ** 2, axis=-1)

#             new_cov_numer = (
#                 np.einsum(
#                     'ijk,ijk->jk', stats['post_comp_mix'], centered_norm2)
#                 + lambdas * centered_means_norm2
#                 + 2 * betas
#             )
#             new_cov_denom = nf * (stats['post_mix_sum'] + 1) + 2 * (alphas + 1)
#             new_cov = new_cov_numer / new_cov_denom

#         elif self.covariance_type == 'tied':
#             centered = stats['centered'].reshape((n_samples, nc, nm, nf, 1))
#             centered_t = stats['centered'].reshape((n_samples, nc, nm, 1, nf))
#             centered_dots = centered * centered_t

#             psis_t = np.transpose(self.covars_prior, axes=(0, 2, 1))
#             nus = self.covars_weight

#             centr_means_resh = centered_means.reshape((nc, nm, nf, 1))
#             centr_means_resh_t = centered_means.reshape((nc, nm, 1, nf))
#             centered_means_dots = centr_means_resh * centr_means_resh_t

#             lambdas_cmdots_prod_sum = (
#                 np.einsum('ij,ijkl->ikl', lambdas, centered_means_dots))

#             new_cov_numer = (
#                 np.einsum(
#                     'ijk,ijklm->jlm', stats['post_comp_mix'], centered_dots)
#                 + lambdas_cmdots_prod_sum + psis_t)
#             new_cov_denom = (
#                 stats['post_sum'] + nm + nus + nf + 1
#             )[:, np.newaxis, np.newaxis]
#             new_cov = new_cov_numer / new_cov_denom

#         # Assigning new values to class members
#         self.weights_ = new_weights
#         self.means_ = new_means
#         self.covars_ = new_cov