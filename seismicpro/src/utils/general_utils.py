"""Miscellaneous general utility functions"""

import numpy as np
from numba import njit, prange

from .interpolation import interpolate


def to_list(obj):
    """Cast an object to a list. Almost identical to `list(obj)` for 1-D objects, except for `str`, which won't be
    split into separate letters but transformed into a list of a single element."""
    return np.array(obj).ravel().tolist()


def maybe_copy(obj, inplace=False, **kwargs):
    """Copy an object if `inplace` flag is set to `False`. Otherwise return the object unchanged."""
    return obj if inplace else obj.copy(**kwargs)


def unique_indices_sorted(arr):
    """Return indices of the first occurrences of the unique values in a sorted array."""
    mask = np.empty(len(arr), dtype=np.bool_)
    mask[:1] = True
    mask[1:] = (arr[1:] != arr[:-1]).any(axis=1)
    return np.where(mask)[0]


@njit(nogil=True)
def calculate_stats(trace):
    """Calculate min, max, sum and sum of squares of the trace amplitudes."""
    trace_min, trace_max = np.inf, -np.inf
    trace_sum, trace_sq_sum = 0, 0
    for sample in trace:
        trace_min = min(sample, trace_min)
        trace_max = max(sample, trace_max)
        trace_sum += sample
        trace_sq_sum += sample**2
    return trace_min, trace_max, trace_sum, trace_sq_sum


@njit(nogil=True)
def create_supergather_index(centers, size):
    """Create a mapping from supergather centers to coordinates of gathers in them.

    Examples
    --------
    >>> centers = np.array([[5, 5], [8, 9]])
    >>> size = (3, 3)
    >>> create_supergather_index(centers, size)
    array([[ 5,  5,  4,  4],
           [ 5,  5,  4,  5],
           [ 5,  5,  4,  6],
           [ 5,  5,  5,  4],
           [ 5,  5,  5,  5],
           [ 5,  5,  5,  6],
           [ 5,  5,  6,  4],
           [ 5,  5,  6,  5],
           [ 5,  5,  6,  6],
           [ 8,  9,  7,  8],
           [ 8,  9,  7,  9],
           [ 8,  9,  7, 10],
           [ 8,  9,  8,  8],
           [ 8,  9,  8,  9],
           [ 8,  9,  8, 10],
           [ 8,  9,  9,  8],
           [ 8,  9,  9,  9],
           [ 8,  9,  9, 10]])

    Parameters
    ----------
    centers : 2d np.ndarray with 2 columns
        Coordinates of supergather centers.
    size : tuple with 2 elements
        Supergather size along inline and crossline axes. Measured in lines.

    Returns
    -------
    mapping : 2d np.ndarray with 4 columns
        Coordinates of supergather centers in the first 2 columns and coordinates of the included gathers in the last
        two columns.
    """
    area_size = size[0] * size[1]
    shifts_i = np.arange(size[0]) - size[0] // 2
    shifts_x = np.arange(size[1]) - size[1] // 2
    mapping = np.empty((len(centers) * area_size, 4), dtype=centers.dtype)
    for ix, (i, x) in enumerate(centers):
        for ix_i, shift_i in enumerate(shifts_i):
            for ix_x, shift_x in enumerate(shifts_x):
                row = np.array([i, x, i + shift_i, x + shift_x])
                mapping[ix * area_size + ix_i * size[1] + ix_x] = row
    return mapping


@njit(nogil=True)
def times_to_indices(times, samples, round=False):
    """Convert `times` to their indices in the increasing `samples` array. If some value of `times` is not present
    in `samples`, its index is linearly interpolated or extrapolated by the other indices of `samples`.

    Parameters
    ----------
    times : 1d np.ndarray of floats
        Time values to convert to indices.
    samples : 1d np.ndarray of floats
        Recording time for each trace value.
    round : bool, optional, defaults to False
        If `True`, round the obtained float indices to the nearest integer. Values exactly halfway between two adjacent
        integers are rounded to the nearest even one.

    Returns
    -------
    indices : 1d np.ndarray
        Array with positions of `times` in `samples`.

    Raises
    ------
    ValueError
        If `samples` is not increasing.
    """
    for i in range(len(samples) - 1):
        if samples[i+1] <= samples[i]:
            raise ValueError('The `samples` array must be increasing.')
    left_slope = 1 / (samples[1] - samples[0])
    right_slope = 1 / (samples[-1] - samples[-2])
    float_position = interpolate(times, samples, np.arange(len(samples), dtype=np.float32), left_slope, right_slope)
    return np.rint(float_position) if round else float_position


@njit(nogil=True)
def convert_times_to_mask(times, samples):
    """Convert `times` to indices by finding a nearest position in `samples` for each time in `times` and construct a
    boolean mask with shape (len(times), len(samples)) with `False` values before calculated time index for each row
    and `True` after.

    Examples
    --------
    >>> times = np.array([0, 4, 6])
    >>> samples = [0, 2, 4, 6, 8]
    >>> convert_times_to_mask(times, samples)
    array([[ True,  True,  True,  True,  True],
           [False, False,  True,  True,  True],
           [False, False, False,  True,  True]])

    Parameters
    ----------
    times : 1d np.ndarray
        Time values to construct the mask. Measured in milliseconds.
    samples : 1d np.ndarray of floats
        Recording time for each trace value. Measured in milliseconds.

    Returns
    -------
    mask : np.ndarray of bool
        Boolean mask with shape (len(times), len(samples)).
    """
    times_indices = times_to_indices(times, samples, round=True)
    return (np.arange(len(samples)) - times_indices.reshape(-1, 1)) >= 0


@njit(nogil=True, parallel=True)
def convert_mask_to_pick(mask, samples, threshold):
    """Convert a first breaks `mask` into an array of arrival times.

    The mask has shape (n_traces, trace_length), each its value represents a probability of corresponding index along
    the trace to follow the first break. A naive approach is to define the first break time index as the location of
    the first trace value exceeding the `threshold`. Unfortunately, it results in noisy predictions, so the following
    conversion procedure is proposed as it appears to be more stable:
    1. Binarize the mask according to the specified `threshold`,
    2. Find the longest sequence of ones in the `mask` for each trace and save indices of the first elements of the
       found sequences,
    3. Return an array of `samples` values corresponding to the obtained indices.

    Examples
    --------
    >>> mask = np.array([[  1, 1, 1, 1, 1],
    ...                  [  0, 0, 1, 1, 1],
    ...                  [0.6, 0, 0, 1, 1]])
    >>> samples = [0, 2, 4, 6, 8]
    >>> threshold = 0.5
    >>> convert_mask_to_pick(mask, samples, threshold)
    array([0, 4, 6])

    Parameters
    ----------
    mask : 2d np.ndarray
        An array with shape (n_traces, trace_length), with each value representing a probability of corresponding index
        along the trace to follow the first break.
    samples : 1d np.ndarray of floats
        Recording time for each trace value. Measured in milliseconds.
    threshold : float
        A threshold for trace mask value to refer its index to be either pre- or post-first break.

    Returns
    -------
    times : np.ndarray with length len(mask)
        Start time of the longest sequence with `mask` values greater than the `threshold` for each trace. Measured in
        milliseconds.
    """
    picking_times = np.empty(len(mask), dtype=np.int32)
    for i in prange(len(mask)):  # pylint: disable=not-an-iterable
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
        picking_times[i] = samples[picking_ix - max_len]
    return picking_times


@njit(nogil=True)
def mute_gather(gather_data, muting_times, samples, fill_value):
    """Fill area before `muting_times` with `fill_value`.

    Parameters
    ----------
    gather_data : 2d np.ndarray
        Gather data to mute.
    muting_times : 1d np.ndarray
        Time values up to which muting is performed. Its length must match `gather_data.shape[0]`. Measured in
        milliseconds.
    samples : 1d np.ndarray of floats
        Recording time for each trace value. Measured in milliseconds.
    fill_value : float
         A value to fill the muted part of the gather with.

    Returns
    -------
    gather_data : 2d np.ndarray
        Muted gather data.
    """
    mask = convert_times_to_mask(times=muting_times, samples=samples)
    data_shape = gather_data.shape
    gather_data = gather_data.reshape(-1)
    mask = mask.reshape(-1)
    gather_data[~mask] = fill_value
    return gather_data.reshape(data_shape)


@njit(nogil=True)
def clip(data, data_min, data_max):
    """Limit the `data` values.

    `data` values outside [`data_min`, `data_max`] interval are clipped to the interval edges.

    Parameters
    ----------
    data : np.ndarray
        Data to clip.
    data_min : int, float
        Minimum value of the interval.
    data_max : int, float
        Maximum value of the interval.

    Returns
    -------
    data : np.ndarray
        Clipped data with the same shape.
    """
    data_shape = data.shape
    data = data.reshape(-1)
    for i in range(len(data)):  # pylint: disable=consider-using-enumerate
        data[i] = min(max(data[i], data_min), data_max)
    return data.reshape(data_shape)


def make_origins(origins, data_shape, crop_shape, n_crops=1, stride=None):
    """Calculate an array of origins or reformat given origins to a 2d `np.ndarray`.

    The returned array has shape [n_origins, 2], where each origin represents a top-left corner of a corresponding crop
    with the shape `crop_shape` from the source data.

    Parameters
    ----------
    origins : list, tuple, np.ndarray or str
        All array-like values are cast to an `np.ndarray` and treated as origins directly, except for a 2-element tuple
        of `int`, which will be treated as a single individual origin.
        If `str`, represents a mode to calculate origins. Two options are supported:
        - "random": calculate `n_crops` crops selected randomly using a uniform distribution over the source data, so
          that no crop crosses data boundaries,
        - "grid": calculate a deterministic uniform grid of origins, whose density is determined by `stride`.
    data_shape : tuple with 2 elements
        Shape of the data to be cropped.
    crop_shape : tuple with 2 elements
        Shape of the resulting crops.
    n_crops : int, optional, defaults to 1
        The number of generated crops if `origins` is "random".
    stride : tuple with 2 elements, optional, defaults to `crop_shape`
        Steps between two adjacent crops along both axes. The lower the value is, the more dense the grid of crops will
        be. An extra origin will always be placed so that the corresponding crop will fit in the very end of an axis to
        guarantee complete data coverage with crops regardless of passed `crop_shape` and `stride`.

    Returns
    -------
    origins : 2d np.ndarray
        An array of absolute coordinates of top-left corners of crops.

    Raises
    ------
    ValueError
        If `origins` is `str`, but not "random" or "grid".
        If `origins` is array-like, but can not be cast to a 2d `np.ndarray` with shape [n_origins, 2].
    """
    if isinstance(origins, str):
        if origins == 'random':
            return np.column_stack((np.random.randint(1 + max(0, data_shape[0] - crop_shape[0]), size=n_crops),
                                    np.random.randint(1 + max(0, data_shape[1] - crop_shape[1]), size=n_crops)))
        if origins == 'grid':
            stride = crop_shape if stride is None else stride
            origins_x = _make_grid_origins(data_shape[0], crop_shape[0], stride[0])
            origins_y = _make_grid_origins(data_shape[1], crop_shape[1], stride[1])
            return np.array(np.meshgrid(origins_x, origins_y)).T.reshape(-1, 2)
        raise ValueError(f"If str, origin should be either 'random' or 'grid' but {origins} was given.")

    origins = np.atleast_2d(origins)
    if origins.ndim == 2 and origins.shape[1] == 2:
        return origins
    raise ValueError("If array-like, origins must be of a shape [n_origins, 2].")


def _make_grid_origins(data_shape, crop_shape, stride):
    """Calculate evenly-spaced origins along a single axis.

    Parameters
    ----------
    data_shape : int
        Shape of the data to be cropped.
    crop_shape : int
        Shape of the resulting crops.
    stride : int
        A step between two adjacent crops.

    Returns
    -------
    origins : 1d np.ndarray
        An array of crop origins.
    """
    max_origin = max(data_shape - crop_shape, 0)
    return np.array(list(range(0, max_origin, stride)) + [max_origin], dtype=np.int32)
