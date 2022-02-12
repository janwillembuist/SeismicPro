from time import time

import matplotlib.pyplot as plt

from .general_utils import MissingModule

# Safe import of modules for interactive plotting
try:
    from ipywidgets import widgets
except ImportError:
    widgets = MissingModule("ipywidgets")

try:
    from IPython.display import display
except ImportError:
    display = MissingModule("IPython.display")


MAX_CLICK_TIME = 0.2


TEXT_LAYOUT = {
    "height": "28px",
    "display": "flex",
    "width": "100%",
    "justify_content": "center",
    "align_items": "center",
}


BUTTON_LAYOUT = {
    "height": "28px",
    "width": "35px",
    "min_width": "35px",
}


TITLE_STYLE = "<style>p{word-wrap:normal; text-align:center; font-size:14px}</style>"
TITLE_TEMPLATE = "{style} <b><p>{title}</p></b>"


class InteractivePlot:
    def __init__(self, title="", plot_fn=None, figsize=(4.5, 4.5), toolbar_position="left"):
        self.plot_fn = plot_fn

        toolbar_visible = (toolbar_position is not None)
        if toolbar_position is None:
            toolbar_position = "left"

        with plt.ioff():
            # Add tight_layout to always correctly show colorbar ticks
            self.fig, self.ax = plt.subplots(figsize=figsize, tight_layout=True)

        self.fig.canvas.header_visible = False
        self.fig.canvas.toolbar_visible = toolbar_visible
        self.fig.canvas.toolbar_position = toolbar_position
        self.fig.canvas.mpl_connect("resize_event", self.on_resize)

        title = TITLE_TEMPLATE.format(style=TITLE_STYLE, title=title)
        self.title = widgets.HTML(value=title, layout=widgets.Layout(**TEXT_LAYOUT))

        self.header = self.create_header()
        self.box = widgets.VBox([self.header, self.fig.canvas])

    def create_header(self):
        placeholder = widgets.HTML(layout=widgets.Layout(**BUTTON_LAYOUT))
        if self.fig.canvas.toolbar_position == "left":
            return widgets.HBox([placeholder, self.title])
        if self.fig.canvas.toolbar_position == "right":
            return widgets.HBox([self.title, placeholder])
        return self.title

    def _resize(self, width):
        width = width / self.fig.canvas._dpi_ratio  # Remove when fixed in ipympl
        width += 4  # Correction for main axes margins
        if self.fig.canvas.toolbar_visible and (self.fig.canvas.toolbar_position in {"left", "right"}):
            width += 44  #  Correction for an optional toolbar and its margins
        self.header.layout.width = f"{int(width)}px"

    def on_resize(self, event):
        self._resize(event.width)

    def set_title(self, title):
        self.title.value = TITLE_TEMPLATE.format(style=TITLE_STYLE, title=title)

    def plot(self, display_box=True):
        self._resize(self.fig.get_figwidth() * self.fig.dpi)  # Init the width of the box
        if self.plot_fn is not None:
            self.plot_fn(ax=self.ax)
        if display_box:
            display(self.box)


class ClickablePlot(InteractivePlot):
    def __init__(self, *args, click_fn=None, allow_unclick=True, unclick_fn=None, init_click_coords=None,
                 marker_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.click_fn = click_fn
        self.unclick_fn = unclick_fn
        marker_params = {} if marker_params is None else marker_params
        self.marker_params = {"color": "black", "marker": "+", **marker_params}
        self.click_time = None
        self.click_scatter = None
        self.init_click_coords = init_click_coords
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        if allow_unclick:
            self.fig.canvas.mpl_connect("key_press_event", self.on_press)

    def plot(self, display_box=True):
        super().plot(display_box)
        if self.init_click_coords is not None:
            self._click(self.init_click_coords)

    def _click(self, coords):
        if self.click_fn is not None:
            coords = self.click_fn(coords)
            if coords is None:  # Skip click processing
                return
        if self.click_scatter is not None:
            self.click_scatter.remove()
        self.click_scatter = self.ax.scatter(*coords, **self.marker_params)
        # TODO: switch to blit
        self.fig.canvas.draw_idle()  # TODO: revert to pure draw when cursor handling is fixed in ipympl

    def on_click(self, event):
        # Discard clicks outside the main axes
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self.click_time = time()

    def on_release(self, event):
        # Discard clicks outside the main axes
        if event.inaxes != self.ax:
            return
        if event.button == 1 and ((time() - self.click_time) < MAX_CLICK_TIME):
            self.click_time = 0
            self._click((event.xdata, event.ydata))

    def on_press(self, event):
        if (event.inaxes != self.ax) or (event.key != "escape"):
            return
        if self.click_scatter is not None:
            if self.unclick_fn is not None:
                self.unclick_fn()
            self.click_scatter.remove()
            self.click_scatter = None
            # TODO: switch to blit
            self.fig.canvas.draw_idle()


class ToggleClickablePlot(ClickablePlot):
    def __init__(self, *args, toggle_fn=None, toggle_icon=None, **kwargs):
        self.button = widgets.Button(icon=toggle_icon, layout=widgets.Layout(**BUTTON_LAYOUT))
        self.button.on_click(toggle_fn)
        super().__init__(*args, **kwargs)

    def create_header(self):
        placeholder = widgets.HTML(layout=widgets.Layout(**BUTTON_LAYOUT))
        if self.fig.canvas.toolbar_position == "right":
            return widgets.HBox([self.button, placeholder])
        return widgets.HBox([self.button, self.title])


class PairedPlot:
    def __init__(self, *args, **kwargs):
        _ = args, kwargs
        self.left = self.construct_left_plot()
        self.right = self.construct_right_plot()
        self.box = widgets.HBox([self.left.box, self.right.box])

    def construct_left_plot(self):
        raise NotImplementedError

    def construct_right_plot(self):
        raise NotImplementedError

    def plot(self):
        self.left.plot(display_box=False)
        self.right.plot(display_box=False)
        display(self.box)
