import numpy as np
from scipy.special import logsumexp

# all modifications from the original code are labeled!


def _forward(n_samples, n_components, log_startprob, log_transmat, framelogprob, fwdlattice):
    work_buffer = np.zeros(n_components)

    # modification!
    fwdlattice[0] = np.full(n_components, -np.inf)
    fwdlattice[0, 0] = 0.

    for t in range(1, n_samples):
        for j in range(n_components):
            for i in range(n_components):
                work_buffer[i] = fwdlattice[t - 1, i] + log_transmat[i, j]

            fwdlattice[t, j] = _logsumexp(work_buffer) + framelogprob[t, j]


def _backward(n_samples, n_components, log_startprob, log_transmat, framelogprob, bwdlattice):

    work_buffer = np.zeros(n_components)

    bwdlattice[-1] = 0.

    for t in range(n_samples - 2, -1, -1):
        for i in range(n_components):
            for j in range(n_components):
                work_buffer[j] = (log_transmat[i, j]
                                  + framelogprob[t + 1, j]
                                  + bwdlattice[t + 1, j])
            bwdlattice[t, i] = _logsumexp(work_buffer)


def _compute_log_xi_sum(n_samples, n_components, fwdlattice, log_transmat, bwdlattice, framelogprob, log_xi_sum):

    work_buffer = np.full((n_components, n_components), -INFINITY)
    logprob = logsumexp(fwdlattice[n_samples - 1])

    for t in range(n_samples - 1):
        for i in range(n_components):
            for j in range(n_components):
                work_buffer[i, j] = (fwdlattice[t, i]
                                     + log_transmat[i, j]
                                     + framelogprob[t + 1, j]
                                     + bwdlattice[t + 1, j]
                                     - logprob)

        for i in range(n_components):
            for j in range(n_components):
                log_xi_sum[i, j] = logsumexp(log_xi_sum[i, j],
                                              work_buffer[i, j])


def _viterbi(int n_samples, int n_components,
             dtype_t[:] log_startprob,
             dtype_t[:, :] log_transmat,
             dtype_t[:, :] framelogprob):

    cdef int i, j, t, where_from
    cdef dtype_t logprob

    cdef int[::view.contiguous] state_sequence = \
        np.empty(n_samples, dtype=np.int32)
    cdef dtype_t[:, ::view.contiguous] viterbi_lattice = \
        np.zeros((n_samples, n_components))
    cdef dtype_t[::view.contiguous] work_buffer = np.empty(n_components)

    with nogil:
        for i in range(n_components):
            viterbi_lattice[0, i] = log_startprob[i] + framelogprob[0, i]

        # Induction
        for t in range(1, n_samples):
            for i in range(n_components):
                for j in range(n_components):
                    work_buffer[j] = (log_transmat[j, i]
                                      + viterbi_lattice[t - 1, j])

                viterbi_lattice[t, i] = _max(work_buffer) + framelogprob[t, i]

        # Observation traceback
        state_sequence[n_samples - 1] = where_from = \
            _argmax(viterbi_lattice[n_samples - 1])
        logprob = viterbi_lattice[n_samples - 1, where_from]

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_components):
                work_buffer[i] = (viterbi_lattice[t, i]
                                  + log_transmat[i, where_from])

            state_sequence[t] = where_from = _argmax(work_buffer)

    return np.asarray(state_sequence), logprob
