"""Implements MetricsAccumulator class for metrics accumulation over batches and MetricMap class for metric
visualization over a field map"""

# pylint: disable=no-name-in-module, import-error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from ipywidgets import widgets
from IPython.display import display

from .utils import to_list, plot_metrics_map
from ..batchflow.models.metrics import Metrics


class MetricsAccumulator(Metrics):
    """Accumulate metric values and their coordinates to further aggregate them into a metrics map.

    Parameters
    ----------
    coords : array-like
        Array of arrays or 2d array with coordinates for X and Y axes.
    kwargs : misc
        Metrics and their values to aggregate and plot on the map. The `kwargs` dict has the following structure:
        `{metric_name_1: metric_values_1,
          ...,
          metric_name_N: metric_values_N
         }`

        Here, `metric_name` is any `str` while `metric_values` should have one of the following formats:
        * If 1d array, each value corresponds to a pair of coordinates with the same index.
        * If an array of arrays, all values from each inner array correspond to a pair of coordinates with the same
          index as in the outer metrics array.
        In both cases the length of `metric_values` must match the length of coordinates array.

    Attributes
    ----------
    metrics_list : pd.DataFrame
        An array with shape (N, 2) which contains X and Y coordinates for each corresponding metric value.
        All keys from `kwargs` become instance attributes and contain the corresponding metric values.

    Raises
    ------
    ValueError
        If `kwargs` were not passed.
        If `ndim` for given coordinate is not equal to 2.
        If shape of the first dim of the coordinate array is not equal to 2.
        If the length of the metric array does not match the length of the array with coordinates.
    TypeError
        If given coordinates are not array-like.
        If given metrics are not array-like.
    """
    def __init__(self, coords, **kwargs):
        super().__init__()

        if not kwargs:
            raise ValueError("At least one metric should be passed.")

        if not isinstance(coords, (list, tuple, np.ndarray)):
            raise TypeError(f"coords must be array-like but {type(coords)} was given.")
        coords = np.asarray(coords)

        # If coords is an array of arrays, convert it to an array with numeric dtype and check its shape
        coords = np.array(coords.tolist()) if coords.ndim == 1 else coords
        if coords.ndim != 2:
            raise ValueError("Coordinates array must be 2-dimensional.")
        if coords.shape[1] != 2:
            raise ValueError("Coordinates array must have shape (N, 2), where N is the number of elements"
                             f" but an array with shape {coords.shape} was given")

        # Create a DataFrame with current metric values
        curr_metrics = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
        for metric_name, metric_values in kwargs.items():
            if not isinstance(metric_values, (list, tuple, np.ndarray)):
                raise TypeError(f"'{metric_name}' metric value must be array-like but {type(metric_values)} received")
            metric_values = np.asarray(metric_values)

            if len(curr_metrics) != len(metric_values):
                raise ValueError(f"The length of {metric_name} metric array must match the length of coordinates "
                                 f"array ({len(curr_metrics)}) but equals {len(metric_values)}")
            curr_metrics[metric_name] = metric_values

        self.metrics_names = sorted(kwargs.keys())
        self.metrics_list = [curr_metrics]
        self.map_kwargs = {}

    @property
    def metrics(self):
        if len(self.metrics_list) > 1:
            self.metrics_list = [pd.concat(self.metrics_list, ignore_index=True)]
        return self.metrics_list[0]

    def append(self, other):
        """Append coordinates and metric values to the global container."""
        self.metrics_names = sorted(set(self.metrics_names + other.metrics_names))
        self.metrics_list += other.metrics_list

    def memorize_map_kwargs(self, **kwargs):
        self.map_kwargs = kwargs

    def _process_metrics_agg(self, metrics, agg):
        is_single_metric = isinstance(metrics, str)
        metrics = to_list(metrics) if metrics is not None else self.metrics_names

        agg = to_list(agg)
        if len(agg) == 1:
            agg *= len(metrics)
        if len(agg) != len(metrics):
            raise ValueError("The number of aggregation functions must match the length of metrics to calculate")

        return metrics, agg, is_single_metric

    def evaluate(self, metrics=None, agg="mean"):
        metrics, agg, is_single_metric = self._process_metrics_agg(metrics, agg)
        metrics_vals = [self.metrics[metric].dropna().explode().agg(agg_func)
                        for metric, agg_func in zip(metrics, agg)]
        if is_single_metric:
            return metrics_vals[0]
        return metrics_vals

    def construct_map(self, metrics=None, agg="mean", bin_size=500, **map_kwargs):
        """Calculate and optionally plot a metrics map.

        The map is constructed in the following way:
        1. All stored coordinates are divided into bins of the specified `bin_size`.
        2. All metric values are grouped by their bin.
        3. An aggregation is performed by calling `agg_func` for values in each bin. If no metric values were assigned
           to a bin, `np.nan` is returned.
        As a result, each value of the constructed map represents an aggregated metric for a particular bin.

        Parameters
        ----------
        metric_name : str
            The name of a metric to construct a map for.
        bin_size : int, float or array-like with length 2, optional, defaults to 500
            Bin size for X and Y axes. If single `int` or `float`, the same bin size will be used for both axes.
        agg_func : str or callable, optional, defaults to 'mean'
            Function to aggregate metric values in a bin.
            If `str`, the function from `DEFAULT_METRICS` will be used.
            If `callable`, it will be used directly. Note, that the function must be wrapped with `njit` decorator.
            Its first argument is a 1d np.ndarray containing metric values in a bin, all other arguments can take any
            numeric values and must be passed using the `agg_func_kwargs`.
        agg_func_kwargs : dict, optional
            Additional keyword arguments to be passed to `agg_func`.
        plot : bool, optional, defaults to True
            Whether to plot the constructed map.
        plot_kwargs : misc, optional
            Additional keyword arguments to be passed to :func:`.plot_utils.plot_metrics_map`.

        Returns
        -------
        metrics_map : 2d np.ndarray
            A map with aggregated metric values.

        Raises
        ------
        TypeError
            If `agg_func` is not `str` or `callable`.
        ValueError
            If `agg_func` is `str` and is not in DEFAULT_METRICS.
            If `agg_func` is not wrapped with `njit` decorator.
        """
        metrics, agg, is_single_metric = self._process_metrics_agg(metrics, agg)
        if isinstance(bin_size, (int, float, np.number)):
            bin_size = (bin_size, bin_size)

        # Binarize metrics for further aggregation into maps
        metrics_df = self.metrics.copy(deep=False)
        metrics_df["x_bin"] = ((metrics_df["x"] - metrics_df["x"].min()) // bin_size[0]).astype(np.int32)
        metrics_df["y_bin"] = ((metrics_df["y"] - metrics_df["y"].min()) // bin_size[1]).astype(np.int32)
        x_range = metrics_df["x_bin"].max() + 1
        y_range = metrics_df["y_bin"].max() + 1
        metrics_df = metrics_df.set_index(["x_bin", "y_bin", "x", "y"]).sort_index()

        metrics_maps = []
        for metric, agg_func in zip(metrics, agg):
            metric_df = metrics_df[metric].dropna().explode()

            metric_agg = metric_df.groupby(["x_bin", "y_bin"]).agg(agg_func)
            x = metric_agg.index.get_level_values(0)
            y = metric_agg.index.get_level_values(1)
            metric_map = np.full((x_range, y_range), fill_value=np.nan)
            metric_map[x, y] = metric_agg

            bin_to_coords = metric_df.groupby(["x_bin", "y_bin", "x", "y"]).agg(agg_func)
            bin_to_coords = bin_to_coords.to_frame().reset_index(level=["x", "y"]).groupby(["x_bin", "y_bin"])

            agg_func = agg_func.__name__ if callable(agg_func) else agg_func
            extra_map_kwargs = {**self.map_kwargs.get(metric, {}), **map_kwargs.get(metric, {})}
            metric_map = MetricMap(metric_map, bin_to_coords, metric, agg_func, bin_size, **extra_map_kwargs)
            metrics_maps.append(metric_map)

        if is_single_metric:
            return metrics_maps[0]
        return metrics_maps


class MetricMap:
    def __init__(self, metric_map, bin_to_coords, metric, agg, bin_size, **extra_map_kwargs):
        self.metric_map = metric_map
        self.bin_to_coords = bin_to_coords
        self.metric = metric
        self.agg = agg
        self.bin_size = bin_size
        self.extra_map_kwargs = extra_map_kwargs

    def plot(self, interactive=False, **kwargs):
        title = f"{self.agg}({self.metric}) in {self.bin_size} bins"
        if not interactive:
            plot_metrics_map(metrics_map=self.metric_map.T)
        else:
            MapViewer(self, title=title, **self.extra_map_kwargs, **kwargs).plot()


class MapViewer:
    def __init__(self, metric_map, on_click_handler, title, is_lower_better=True, map_size=(4, 4), aux_size=(4, 4)):
        self.metric_map = metric_map
        self.on_click_handler = on_click_handler
        self.is_desc = is_lower_better

        # Current map state
        self.click_point = None
        self.curr_coords = None
        self.curr_ix = None

        # Figure setup
        self.output = widgets.Output()
        with self.output:
            self.map_fig, self.map_ax = plt.subplots(figsize=map_size, constrained_layout=True)
            self.aux_fig, self.aux_ax = plt.subplots(figsize=aux_size, constrained_layout=True)

        self.map_fig.canvas.toolbar_visible = False
        self.aux_fig.canvas.toolbar_visible = False
        self.map_fig.canvas.header_visible = False
        self.aux_fig.canvas.header_visible = False

        # Layout definition
        text_layout = widgets.Layout(height="35px", display="flex", justify_content="center", width="95%")
        button_layout = widgets.Layout(height="35px", width="35px", min_width="35px")

        # Widget definition
        title_style = "<style>p{word-wrap:normal; text-align:center; font-size:14px}</style>"
        self.title = widgets.HTML(value=f"{title_style} <b><p>{title}</p></b>", layout=text_layout)
        self.sort = widgets.Button(icon=self.sort_icon, layout=button_layout)
        self.prev = widgets.Button(icon="angle-left", layout=button_layout)
        self.drop = widgets.Dropdown(layout=text_layout)
        self.next = widgets.Button(icon="angle-right", layout=button_layout)
        self.buttons = widgets.HBox([self.sort, self.prev, self.drop, self.next])

        # Map layout
        self.figure_box = widgets.HBox([widgets.VBox([self.title, self.map_fig.canvas]),
                                        widgets.VBox([self.buttons, self.aux_fig.canvas])])

        # Handler definition
        self.sort.on_click(self.reverse_coords)
        self.prev.on_click(self.prev_coords)
        self.drop.observe(self.select_coords, names="value")
        self.next.on_click(self.next_coords)
        self.map_fig.canvas.mpl_connect("button_press_event", self.map_on_click)

    @property
    def sort_icon(self):
        return "sort-amount-desc" if self.is_desc else "sort-amount-asc"

    def gen_drop_options(self, coords):
        return [f"({x}, {y}) - {metric:.05f}" for x, y, metric in coords.itertuples(index=False)]

    def redraw_aux(self):
        curr_x, curr_y = self.curr_coords.iloc[self.curr_ix][["x", "y"]]
        self.aux_ax.clear()
        self.on_click_handler(curr_x, curr_y, self.aux_ax)

    def update_drop(self, ix, coords=None):
        self.drop.unobserve(self.select_coords, names="value")
        with self.drop.hold_sync():
            if coords is not None:
                self.drop.options = self.gen_drop_options(coords)
            self.drop.index = ix
        self.drop.observe(self.select_coords, names="value")

    def update_state(self, ix, coords=None, redraw=True):
        self.curr_ix = ix
        if coords is not None:
            self.curr_coords = coords
        self.update_drop(ix, coords)
        self.toggle_prev_next_buttons()
        if redraw:
            self.redraw_aux()

    def toggle_prev_next_buttons(self):
        self.prev.disabled = (self.curr_ix == 0)
        self.next.disabled = (self.curr_ix == (len(self.curr_coords) - 1))

    def reverse_coords(self, event):
        self.is_desc = not self.is_desc
        self.sort.icon = self.sort_icon
        self.update_state(len(self.curr_coords) - self.curr_ix - 1, self.curr_coords.iloc[::-1], redraw=False)

    def next_coords(self, event):
        self.update_state(min(self.curr_ix + 1, len(self.curr_coords) - 1))

    def prev_coords(self, event):
        self.update_state(max(self.curr_ix - 1, 0))

    def select_coords(self, change):
        self.update_state(self.drop.index)

    def process_click(self, *click_coords):
        # Handle clicks on an empty area
        if click_coords not in self.metric_map.bin_to_coords.groups:
            return

        if self.click_point is not None:
            self.click_point.remove()
        self.click_point = self.map_ax.scatter(*click_coords, color="black", marker="+")

        coords = self.metric_map.bin_to_coords.get_group(click_coords)
        coords = coords.sort_values(self.metric_map.metric, ascending=not self.is_desc)
        self.update_state(0, coords)

    def map_on_click(self, event):
        # Handle clicks outside the map
        if not event.inaxes == self.map_ax:
            return
        self.process_click(int(event.xdata + 0.5), int(event.ydata + 0.5))

    def plot(self):
        display(self.figure_box)

        # Plot metric map
        colors = ((0.0, 0.6, 0.0), (.66, 1, 0), (0.9, 0.0, 0.0))
        cmap = mcolors.LinearSegmentedColormap.from_list("cmap", colors)
        self.map_ax.imshow(self.metric_map.metric_map.T, origin="lower", cmap=cmap, aspect="auto", interpolation="None")

        # Init aux plot with the worst metric value
        func = np.nanargmax if self.is_desc else np.nanargmin
        init_x, init_y = np.unravel_index(func(self.metric_map.metric_map), self.metric_map.metric_map.shape)
        self.process_click(init_x, init_y)
