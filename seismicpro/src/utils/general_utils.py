import numpy as np
from numba import njit, prange


def to_list(obj):
    """Cast an object to a list. Almost identical to `list(obj)` for 1-D
    objects, except for `str`, which won't be split into separate letters but
    transformed into a list of a single element.
    """
    return np.array(obj).ravel().tolist()


def to_str(obj, sep=' '):
    return sep.join(map(str, to_list(obj)))

def create_separator(indent, block_name, print_sep=True):
    full_length = 78 - indent * 4
    n_spaces = full_length - len(block_name)

    sep = f"#{'-'*(full_length-2)}#"
    left_side = '#' + ' ' * (n_spaces // 2 - 1)
    right_side = ' '*(np.ceil(n_spaces/2).astype(int) - 1)+'#'
    msg = left_side+block_name+right_side
    if print_sep:
        print(sep)
        print(msg)
        print(sep)

    return sep + '\n' + msg + '\n' + sep

@njit
def calculate_stats(trace):
    trace_min, trace_max = np.inf, -np.inf
    trace_sum, trace_sq_sum = 0, 0
    for sample in trace:
        trace_min = min(sample, trace_min)
        trace_max = max(sample, trace_max)
        trace_sum += sample
        trace_sq_sum += sample**2
    return trace_min, trace_max, trace_sum, trace_sq_sum


@njit
def create_supergather_index(lines, size):
    area_size = size[0] * size[1]
    shifts_i = np.arange(size[0]) - size[0] // 2
    shifts_x = np.arange(size[1]) - size[1] // 2
    supergather_lines = np.empty((len(lines) * area_size, 4), dtype=lines.dtype)
    for ix, (i, x) in enumerate(lines):
        for ix_i, shift_i in enumerate(shifts_i):
            for ix_x, shift_x in enumerate(shifts_x):
                row = np.array([i, x, i + shift_i, x + shift_x])
                supergather_lines[ix * area_size + ix_i * size[1] + ix_x] = row
    return supergather_lines


@njit(parallel=True)
def convert_mask_to_pick(mask, threshold):
    picking_array = np.empty(len(mask), dtype=np.int32)
    for i in prange(len(mask)):
        trace = mask[i]
        max_len, curr_len, picking_ix = 0, 0, 0
        for j, sample in enumerate(trace):
            # Count length of current sequence of ones
            if sample >= threshold:
                curr_len += 1
            else:
                # If the new longest sequence found
                if curr_len > max_len:
                    max_len = curr_len
                    picking_ix = j
                curr_len = 0
        # If the longest sequence found in the end of the trace
        if curr_len > max_len:
            picking_ix = len(trace)
            max_len = curr_len
        picking_array[i] = picking_ix - max_len
    return picking_array
