"""Implements classes for velocity analysis: StackingVelocity and VelocityCube"""

import warnings

import numpy as np
import cv2
from scipy.interpolate import interp1d, LinearNDInterpolator
from sklearn.neighbors import NearestNeighbors

from .utils import to_list, read_vfunc, read_single_vfunc, dump_vfunc


class VelocityInterpolator:
    def __init__(self, stacking_velocities_dict):
        self.coords = np.array(list(stacking_velocities_dict.keys()))
        self.laws = np.array(list(stacking_velocities_dict.values()))
        
        max_dx = np.diff(np.sort(np.unique(self.coords.T[0]))).max(initial=0)
        max_dy = np.diff(np.sort(np.unique(self.coords.T[1]))).max(initial=0)
        max_r = np.sqrt( max_dx ** 2 + max_dy ** 2)

        self.knn = NearestNeighbors(n_neighbors=1, radius=max_r)
        self.knn.fit(self.coords)

    def _azimuth(self, x, y):
        res = []
        if x >= 0 and y >= 0: res.append(1)
        if x <= 0 and y >= 0: res.append(2)
        if x <= 0 and y <= 0: res.append(3)
        if x >= 0 and y <= 0: res.append(4)
        return res

    def point_surrounders(self, inline, crossline):
        surrounders = {}
        _, indices = self.knn.radius_neighbors([(inline, crossline),], return_distance=True, sort_results=True)
        for _, ind in enumerate(indices[0]):
            centered_coord = self.coords[ind] - (inline, crossline)
            azimuth = self._azimuth(*centered_coord)
            for az in azimuth:
                if az not in surrounders: 
                    surrounders[az] = self.laws[ind]
            if len(surrounders) == 4:
                return surrounders
        return surrounders

    def _interpolate_linear(self, inline, crossline, surrounders):
        x_left, x_right = surrounders[2].inline, surrounders[1].inline
        y_bot, y_top = surrounders[3].crossline, surrounders[1].crossline
        w_x = 0.5 if x_right==x_left else (inline - x_left) / (x_right - x_left)
        w_y = 0.5 if y_bot==y_top else (crossline - y_bot) / (y_top - y_bot)
        w = [w_x*w_y, (1-w_x)*w_y, (1-w_x)*(1-w_y), w_x*(1-w_y)]

        times_union = np.unique([times for law in surrounders.values() for times in law.times])
        interp_t = np.array([law(times_union) for law in surrounders.values()])
        interp_xy = np.average(interp_t, 0, w)
        return StackingVelocity.from_points(times_union, interp_xy, inline, crossline)

    def _interpolate_nearest(self, inline, crossline):
        """Return the closest known stacking velocity to given `inline` and `crossline`."""
        index = self.knn.kneighbors([(inline, crossline),], return_distance=False).item()
        nearest_stacking_velocity = self.laws[index]
        return StackingVelocity.from_points(nearest_stacking_velocity.times, nearest_stacking_velocity.velocities,
                                            inline=inline, crossline=crossline)

    def __call__(self, inline, crossline):
        surrounders = self.point_surrounders(inline, crossline)
        if len(surrounders) == 4:
            return self._interpolate_linear(inline, crossline, surrounders)
        return self._interpolate_nearest(inline, crossline)


class StackingVelocity:
    """A class representing stacking velocity in a certain place of a field.

    Stacking velocity is the value of the seismic velocity obtained from the best fit of the traveltime curve by a
    hyperbola for each timestamp. It is used to correct the arrival times of reflection events in the traces for their
    varying offsets prior to stacking.

    It can be created from three different types of data by calling a corresponding `classmethod`:
    * `from_points` - create a stacking velocity from 1d arrays of times and velocities,
    * `from_file` - create a stacking velocity from a file in VFUNC format with time-velocity pairs,
    * `from_interpolator` - create a stacking velocity from a callable that returns velocity value by given time.

    However, usually a stacking velocity instance is not created directly, but is obtained as a result of calling the
    following methods:
    * :func:`~semblance.Semblance.calculate_stacking_velocity` - to run an automatic algorithm for stacking velocity
      computation,
    * :func:`VelocityCube.__call__` - to interpolate a stacking velocity at passed field coordinates given a created or
      loaded velocity cube.

    The resulting object is callable and returns stacking velocities for given times.

    Examples
    --------
    Stacking velocity can be automatically calculated for a CDP gather by its semblance:
    >>> survey = Survey(path, header_index=["INLINE_3D", "CROSSLINE_3D"], header_cols="offset")
    >>> gather = survey.sample_gather().sort(by="offset")
    >>> semblance = gather.calculate_semblance(velocities=np.linspace(1400, 5000, 200), win_size=8)
    >>> velocity = semblance.calculate_stacking_velocity()

    Or it can be interpolated from a velocity cube (loaded from a file in our case):
    >>> cube = VelocityCube(path=cube_path).create_interpolator()
    >>> velocity = cube(inline, crossline)

    Attributes
    ----------
    interpolator : callable
        An interpolator returning velocity value by given time.
    times : 1d array-like
        An array with time values for which stacking velocity was picked. Measured in milliseconds. Can be `None` if
        the instance was created from interpolator.
    velocities : 1d array-like
        An array with stacking velocity values, matching the length of `times`. Measured in meters/seconds. Can be
        `None` if the instance was created from interpolator.
    inline : int
        An inline of the stacking velocity. If `None`, the created instance won't be able to be added to a
        `VelocityCube`.
    crossline : int
        A crossline of the stacking velocity. If `None`, the created instance won't be able to be added to a
        `VelocityCube`.
    """
    def __init__(self):
        self.interpolator = lambda times: np.zeros_like(times, dtype=np.float32)
        self.times = None
        self.velocities = None
        self.inline = None
        self.crossline = None

    @classmethod
    def from_points(cls, times, velocities, inline=None, crossline=None):
        """Init stacking velocity from arrays of times and corresponding velocities.

        The resulting object performs linear velocity interpolation between points given. Linear extrapolation is
        performed outside of the defined times range.

        Parameters
        ----------
        times : 1d array-like
            An array with time values for which stacking velocity was picked. Measured in milliseconds.
        velocities : 1d array-like
            An array with stacking velocity values, matching the length of `times`. Measured in meters/seconds.
        inline : int, optional, defaults to None
            An inline of the created stacking velocity. If `None`, the created instance won't be able to be added to a
            `VelocityCube`.
        crossline : int, optional, defaults to None
            A crossline of the created stacking velocity. If `None`, the created instance won't be able to be added to
            a `VelocityCube`.

        Returns
        -------
        self : StackingVelocity
            Created stacking velocity instance.

        Raises
        ------
        ValueError
            If shapes of `times` and `velocities` are inconsistent or some velocity values are negative.
        """
        times = np.array(times, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)
        if times.ndim != 1 or times.shape != velocities.shape:
            raise ValueError("Inconsistent shapes of times and velocities")
        if (velocities < 0).any():
            raise ValueError("Velocity values must be positive")
        self = cls.from_interpolator(interp1d(times, velocities, fill_value="extrapolate"), inline, crossline)
        self.times = times
        self.velocities = velocities
        return self

    @classmethod
    def from_file(cls, path):
        """Init stacking velocity from a file with vertical functions in Paradigm Echos VFUNC format.

        The file must have exactly one record with the following structure:
        VFUNC [inline] [crossline]
        [time_1] [velocity_1] [time_2] [velocity_2] ... [time_n] [velocity_n]

        The resulting object performs linear velocity interpolation between points given. Linear extrapolation is
        performed outside of the defined times range.

        Parameters
        ----------
        path : str
            A path to the source file.

        Returns
        -------
        self : StackingVelocity
            Loaded stacking velocity instance.
        """
        inline, crossline, times, velocities = read_single_vfunc(path)
        return cls.from_points(times, velocities, inline, crossline)

    @classmethod
    def from_interpolator(cls, interpolator, inline=None, crossline=None):
        """Init stacking velocity from velocity interpolator.

        Parameters
        ----------
        interpolator : callable
            An interpolator returning velocity value by given time.
        inline : int, optional, defaults to None
            An inline of the created stacking velocity. If `None`, the created instance won't be able to be added to a
            `VelocityCube`.
        crossline : int, optional, defaults to None
            A crossline of the created stacking velocity. If `None`, the created instance won't be able to be added to
            a `VelocityCube`.

        Returns
        -------
        self : StackingVelocity
            Created stacking velocity instance.
        """
        self = cls()
        self.interpolator = interpolator
        self.inline = inline
        self.crossline = crossline
        return self

    def set_times(self, times):
        """Create a new `StackingVelocity` from given `times` and `velocities` recalculated as `self(times)`.
        Coordinates of the created instance will match those of `self`.

        Parameters
        ----------
        times : 1d np.ndarray
            Times to create a new stacking velocity instance for.

        Returns
        -------
        self : StackingVelocity
            Stacking velocities for given `times`.
        """
        return self.from_points(times, self(times), self.inline, self.crossline)

    def dump(self, path):
        """Dump stacking velocities to a file in VFUNC format.

        Notes
        -----
        See more about the format in :func:`~utils.file_utils.dump_vfunc`.

        Parameters
        ----------
        path : str
            A path to the created file.
        """
        if not self.has_coords or not self.has_points:
            raise ValueError("StackingVelocity instance can be dumped only if it was created from time and velocity "
                             "pairs with not-None inline and crossline")
        dump_vfunc(path, [(self.inline, self.crossline, self.times, self.velocities)])

    @property
    def has_points(self):
        """bool: Whether the instance was created from time and velocity pairs."""
        return (self.times is not None) and (self.velocities is not None)

    @property
    def has_coords(self):
        """bool: Whether stacking velocity inline and crossline are not-None."""
        return (self.inline is not None) and (self.crossline is not None)

    @property
    def interpolation_data(self):
        """2d np.ndarray with 4 columns: data to be passed to `VelocityInterpolator`. First two columns store
        duplicated inline and crossline, while two last store time-velocity pairs."""
        if not self.has_coords or not self.has_points:
            raise ValueError("Interpolation data can be obtained only for stacking velocities created from time and "
                             "velocity pairs with not-None inline and crossline")
        coords = [self.get_coords()] * len(self.times)
        return np.concatenate([coords, self.times.reshape(-1, 1), self.velocities.reshape(-1, 1)], axis=1)

    def get_coords(self):
        """Get spatial coordinates of the stacking velocity.

        Returns
        -------
        coords : tuple with 2 elements
            Stacking velocity spatial coordinates.
        """
        return (self.inline, self.crossline)

    def __call__(self, times):
        """Returns stacking velocities for given `times`.

        Parameters
        ----------
        times : 1d array-like
            An array with time values. Measured in milliseconds.

        Returns
        -------
        velocities : 1d array-like
            An array with stacking velocity values, matching the length of `times`. Measured in meters/seconds.
        """
        return np.maximum(self.interpolator(times), 0)


class VelocityCube:
    """A class for storing and interpolating stacking velocity data over a field.

    Velocities used for seismic cube stacking are usually picked on a sparse grid of inlines and crosslines and then
    interpolated over the whole field in order to reduce computational costs. Such interpolation can be performed by
    `VelocityCube` class which provides an interface to obtain a stacking velocity at given spatial coordinates via its
    `__call__` method. The cube can either be loaded from a file of vertical functions or created empty and iteratively
    updated with calculated stacking velocities.

    After all velocities are added, velocity interpolator should be created. It can be done either manually by
    calling :func:`~VelocityCube.create_interpolator` method or automatically during the first call to the cube. Manual
    interpolator creation is useful when the cube should be passed to different proccesses (e.g. in a pipeline with
    prefetch with `mpc` target) since otherwise the interpolator will be independently created in all the processes.

    Examples
    --------
    The cube can either be loaded from a file:
    >>> cube = VelocityCube(path=cube_path)

    Or created empty and updated with instances of `StackingVelocity` class:
    >>> cube = VelocityCube()
    >>> velocity = StackingVelocity.from_points(times=[0, 1000, 2000, 3000], velocities=[1500, 2000, 2800, 3400],
    ...                                         inline=20, crossline=40)
    >>> cube.update(velocity)

    Cube creation should be finalized with `create_interpolator` method call:
    >>> cube.create_interpolator()

    Parameters
    ----------
    path : str. optional
        A path to the source file with vertical functions to load the cube from. If not given, an empty cube is
        created.
    tmin : float, optional, defaults to 0
        Start time to clip all stacking velocities with to ensure that the convex hull of the point cloud passed to
        `LinearNDInterpolator` covers all timestamps for all spatial coordinates. Measured in milliseconds.
    tmax : float, optional, defaults to 6000
        End time to clip all stacking velocities with to ensure that the convex hull of the point cloud passed to
        `LinearNDInterpolator` covers all timestamps for all spatial coordinates. Measured in milliseconds.

    Attributes
    ----------
    stacking_velocities_dict : dict
        A dict of stacking velocities in the cube whose keys are tuples with their spatial coordinates and values are
        the instances themselves.
    interpolator : VelocityInterpolator
        Velocity interpolator over the field.
    is_dirty_interpolator : bool
        Whether the cube was updated after the interpolator was created.
    tmin : float, optional, defaults to 0
        Start time to clip all stacking velocities with to ensure that the convex hull of the point cloud passed to
        `LinearNDInterpolator` covers all timestamps for all spatial coordinates. Measured in milliseconds.
    tmax : float, optional, defaults to 6000
        End time to clip all stacking velocities with to ensure that the convex hull of the point cloud passed to
        `LinearNDInterpolator` covers all timestamps for all spatial coordinates. Measured in milliseconds.
    """
    def __init__(self, path=None, tmin=0, tmax=6000):
        self.stacking_velocities_dict = {}
        self.interpolator = None
        self.is_dirty_interpolator = True
        self.tmin = tmin
        self.tmax = tmax
        if path is not None:
            self.load(path)

    @property
    def has_interpolator(self):
        """bool: Whether the velocity interpolator was created."""
        return self.interpolator is not None

    def load(self, path):
        """Load a velocity cube from a file with vertical functions in Paradigm Echos VFUNC format.

        The file may have one or more records with the following structure:
        VFUNC [inline] [crossline]
        [time_1] [velocity_1] [time_2] [velocity_2] ... [time_n] [velocity_n]

        Parameters
        ----------
        path : str
            A path to the source file.

        Returns
        -------
        self : VelocityCube
            Self with loaded stacking velocities. Changes `stacking_velocities_dict` inplace and sets the
            `is_dirty_interpolator` flag to `True`.
        """
        for inline, crossline, times, velocities in read_vfunc(path):
            stacking_velocity = StackingVelocity.from_points(times, velocities, inline, crossline)
            self.stacking_velocities_dict[(inline, crossline)] = stacking_velocity
        self.is_dirty_interpolator = True
        return self

    def dump(self, path):
        """Dump all the vertical functions of the cube to a file in VFUNC format.

        Notes
        -----
        See more about the format in :func:`~utils.file_utils.dump_vfunc`.

        Parameters
        ----------
        path : str
            A path to the created file.
        """
        vfunc_list = []
        for (inline, crossline), stacking_velocity in self.stacking_velocities_dict.items():
            vfunc_list.append((inline, crossline, stacking_velocity.times, stacking_velocity.velocities))
        dump_vfunc(path, vfunc_list)

    def update(self, stacking_velocities):
        """Update a velocity cube with given stacking velocities.

        Notes
        -----
        All passed `StackingVelocity` instances must have not-None coordinates.

        Parameters
        ----------
        stacking_velocities : StackingVelocity or list of StackingVelocity
            Stacking velocities to update the cube with.

        Returns
        -------
        self : VelocityCube
            Self with loaded stacking velocities. Changes `stacking_velocities_dict` inplace and sets the
            `is_dirty_interpolator` flag to `True` if passed `stacking_velocities` is not empty.

        Raises
        ------
        TypeError
            If wrong type of `stacking_velocities` was passed.
        ValueError
            If any of the passed stacking velocities has `None` coordinates.
        """
        stacking_velocities = to_list(stacking_velocities)
        if not all(isinstance(vel, StackingVelocity) for vel in stacking_velocities):
            raise TypeError("The cube can be updated only with `StackingVelocity` instances")
        if not all(vel.has_coords for vel in stacking_velocities):
            raise ValueError("All passed `StackingVelocity` instances must have not-None coordinates")
        for vel in stacking_velocities:
            self.stacking_velocities_dict[vel.get_coords()] = vel
        if stacking_velocities:
            self.is_dirty_interpolator = True
        return self

    def create_interpolator(self):
        """Create velocity interpolator from stacking velocities in the cube.

        Notes
        -----
        Time range of all stacking velocities in the cube will be set to [`tmin`, `tmax`] to ensure that the convex
        hull of the point cloud passed to `LinearNDInterpolator` covers all timestamps for all spatial coordinates.

        Returns
        -------
        self : VelocityCube
            Self with created interploator. Updates `interpolator` inplace and sets the `is_dirty_interpolator` flag
            to `False`.

        Raises
        ------
        ValueError
            If velocity cube is empty.
        """
        if not self.stacking_velocities_dict:
            raise ValueError("No stacking velocities passed")

        # Set the time range of all stacking velocities to [tmin, tmax] in order to ensure that the convex hull of the
        # point cloud passed to LinearNDInterpolator covers all timestamps from tmin to tmax
        stacking_velocities_dict = {}
        for coord, stacking_velocity in self.stacking_velocities_dict.items():
            times_union = np.union1d(stacking_velocity.times, [self.tmin, self.tmax])
            stacking_velocities_dict[coord] = stacking_velocity.set_times(times_union)

        self.interpolator = VelocityInterpolator(stacking_velocities_dict)
        self.is_dirty_interpolator = False
        return self

    def __call__(self, inline, crossline, create_interpolator=True):
        """Interpolate stacking velocity at given `inline` and `crossline`.

        Parameters
        ----------
        inline : int
            An inline to interpolate stacking velocity at.
        crossline : int
            A crossline to interpolate stacking velocity at.
        create_interpolator : bool, optional, defaults to True
            Whether to create a velocity interpolator if it does not exist.

        Returns
        -------
        stacking_velocity : StackingVelocity
            Interpolated stacking velocity at (`inline`, `crossline`).

        Raises
        ------
        ValueError
            If velocity interpolator does not exist and `create_interpolator` flag is set to `False`.
        """
        if create_interpolator and (not self.has_interpolator or self.is_dirty_interpolator):
            self.create_interpolator()
        elif not create_interpolator:
            if not self.has_interpolator:
                raise ValueError("Velocity interpolator must be created first")
            if self.is_dirty_interpolator:
                warnings.warn("Dirty interpolator is being used", RuntimeWarning)
        return self.interpolator(inline, crossline)
