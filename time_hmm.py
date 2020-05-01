import numpy as np
from scipy.stats import norm

# take in a matrix, fill in the k-th diagonal with a value or array
def fill_diagonal(matrix, value, k):
    assert len(matrix.shape) == 2
    length_to_fill = matrix.shape[1] - abs(k)
    if isinstance(value, int) or isinstance(value, float):
        fill_array = np.full(length_to_fill, value)
    else:
        assert value.shape == (length_to_fill,)
        fill_array = value
    fill_counter = 0
    for vert_idx in range(matrix.shape[0]):
        for horiz_idx in range(matrix.shape[1]):
            if (vert_idx + k) == horiz_idx:
                matrix[vert_idx, horiz_idx] = fill_array[fill_counter]
                fill_counter += 1
    return matrix


def make_safety_mask(num_frames, num_states):
    matrix_width = num_frames * num_states
    safety_mask = np.zeros([matrix_width, matrix_width])
    block_cutoffs = np.arange(0, matrix_width + num_frames, num_frames)
    vert_lower_cutoff = 0
    for vert_idx, vert_upper_cutoff in enumerate(block_cutoffs[1:]):
        horiz_lower_cutoff = 0
        for horiz_idx, horiz_upper_cutoff in enumerate(block_cutoffs[1:]):
            block = safety_mask[vert_lower_cutoff:vert_upper_cutoff, horiz_lower_cutoff:horiz_upper_cutoff]
            # if this block is on the main diagonal
            if vert_idx == horiz_idx:
                fill_diagonal(block, value=1, k=-1)
            # if this block is on the -1 diagonal
            if (vert_idx - 1) == horiz_idx:
                block[0][:-1] = 1
            horiz_lower_cutoff = horiz_upper_cutoff
        vert_lower_cutoff = vert_upper_cutoff
    return safety_mask

# customize this as needed
def get_transition_probs(num_frames=10):
    # this will always be one fewer than the number of frames because we're filling the -1 diagonal
    normal = norm(loc=3, scale=5)
    probs = [normal.pdf(duration) for duration in range(1, num_frames)]
    probs = np.array(probs)
    probs /= np.max(probs)
    probs *= 0.99
    # probs = [0.97 ** duration for duration in range(1, num_frames)]
    return np.array(probs)


def make_transition_matrix(num_frames, num_states):
    matrix_width = num_frames * num_states
    transition_matrix = np.zeros([matrix_width, matrix_width])
    block_cutoffs = np.arange(0, matrix_width + num_frames, num_frames)
    vert_lower_cutoff = 0
    transition_probs = get_transition_probs(num_frames)
    comp_transition_probs = 1. - transition_probs
    for vert_idx, vert_upper_cutoff in enumerate(block_cutoffs[1:]):
        horiz_lower_cutoff = 0
        for horiz_idx, horiz_upper_cutoff in enumerate(block_cutoffs[1:]):
            block = transition_matrix[vert_lower_cutoff:vert_upper_cutoff, horiz_lower_cutoff:horiz_upper_cutoff]
            # if this block is on the main diagonal
            if vert_idx == horiz_idx:
                fill_diagonal(block, value=transition_probs, k=-1)
            # if this block is on the -1 diagonal
            if (vert_idx - 1) == horiz_idx:
                block[0][:-1] = comp_transition_probs
            horiz_lower_cutoff = horiz_upper_cutoff
        vert_lower_cutoff = vert_upper_cutoff
    safety_mask = make_safety_mask(num_frames, num_states)
    assert np.all(transition_matrix * safety_mask == transition_matrix)
    return transition_matrix


