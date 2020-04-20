import numpy as np
from scipy.special import logsumexp

# all modifications from the original code are labeled!


def _forward(n_samples, n_components, log_startprob, log_transmat, framelogprob, fwdlattice):
    # modification!
    # work_buffer = np.zeros(n_components)
    n_possibilities = n_components * n_samples
    work_buffer = np.zeros(n_possibilities)

    # modification!
    import pdb; pdb.set_trace()
    fwdlattice[0] = -np.inf
    fwdlattice[0, 0] = 0.

    import pdb; pdb.set_trace()
    for t in range(1, n_samples):
        for j in range(n_possibilities):
            for i in range(n_possibilities):
                work_buffer[i] = fwdlattice[t - 1, i] + log_transmat[i, j]

            fwdlattice[t, j] = _logsumexp(work_buffer) + framelogprob[t, j]


def _backward(n_samples, n_components, log_startprob, log_transmat, framelogprob, bwdlattice):
    # modification
    # work_buffer = np.zeros(n_components)
    n_possibilities = n_samples * n_components
    work_buffer = np.zeros(n_possibilities, n_samples)

    import pdb; pdb.set_trace()
    bwdlattice[-1] = -np.inf
    bwdlattice[-1, -1] = 0.

    import pdb; pdb.set_trace()
    for t in range(n_samples - 2, -1, -1):
        for i in range(n_possibilities):
            for j in range(n_possibilities):
                work_buffer[j] = (log_transmat[i, j]
                                  + framelogprob[t + 1, j]
                                  + bwdlattice[t + 1, j])
            bwdlattice[t, i] = _logsumexp(work_buffer)


def _compute_log_xi_sum(n_samples, n_components, fwdlattice, log_transmat, bwdlattice, framelogprob, log_xi_sum):

    # modification!
    # work_buffer = np.full((n_components, n_components), -np.inf)
    n_possibilities = n_components * n_samples
    work_buffer = np.full((n_possibilities, n_possibilities), -np.inf)
    logprob = logsumexp(fwdlattice[-1])

    import pdb; pdb.set_trace()
    for t in range(n_samples):
        for i in range(n_possibiliites):
            for j in range(n_possibilities):
                work_buffer[i, j] = (fwdlattice[t, i]
                                     + log_transmat[i, j]
                                     + framelogprob[t + 1, j]
                                     + bwdlattice[t + 1, j]
                                     - logprob)

        for i in range(n_possibilites):
            for j in range(n_possibilities):
                log_xi_sum[i, j] = logsumexp(log_xi_sum[i, j],
                                              work_buffer[i, j])


def _viterbi(n_samples, n_components, log_startprob, log_transmat, framelogprob):
    state_sequence = \
        np.empty(n_samples, dtype=np.int32)
    n_possibilities = n_samples * n_components
    viterbi_lattice = \
        np.zeros((n_samples, n_possibilities))
    work_buffer = np.empty(n_possibilities)

    # modification!
    # for i in range(n_components):
    #     viterbi_lattice[0, i] = log_startprob[i] + framelogprob[0, i]
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

    return np.asarray(state_sequence), logprob
