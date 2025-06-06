import numpy as np
from numpy.typing import ArrayLike

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from collections.abc import Sequence
from typing import Any, Literal

def _create_sequence(obj: Any, length: int, force_iter=False) -> Sequence[Any]:
    if not isinstance(obj, Sequence) or isinstance(obj, str) or force_iter:
        return [obj] * length
    if len(obj) == length:
        return obj
    else:
        raise ValueError(f"Expected object of length [{length}], got [{len(obj)}]")
    

def histerr(x: ArrayLike, err_type: str = "poisson", bins: int | ArrayLike = 10, norm_method: str | None = None, 
            weights: ArrayLike | None = None, scale_factor: float | None = None, step: Literal["pre", "mid", "post"] | None = "post", 
            ax: Axes | None = None, **mpl_kwargs):
    """
    Works like a regular histogram, but additionally handles adding in error bar via ax.fill_between

    x : ArrayLike
        The input data to create a histogram from.
    err_type : str
        The type of error of the histogram. Currently only supported type is Poisson (error is the square root of the count of any given bin)
    bins : int or sequence of scalars or str
        Binning for the histogram, same as bins in np.histogram
    norm_method : str or None
        Determines the normalization method used on the histogram, can be either 'count' (sum of counts in histogram equals 1) or 
        'area' (integral of histogram/area of bins equals 1). Cannot be set at the same time as 'weight'
    weights : ArrayLike or None
        Determines the weight for each entry in the histogram, same as weight in np.histogram
    scale_factor : float or None
        Determines a flat scaling factor to multiply our array by. Cannot be set at the same time as 'norm_method'
    step : str
        Same as passing 'where' parameter to plt.step and 'step' parameter to 'plt.fill_between', basically determines where your histogram
        edges are visually relative to the bins ('pre', 'mid' or 'post'), generally want to keep this as 'post'
    ax : Axes
        Pass in an optional Axes parameter to have the plot apply to that axis rather than creating a new one
    **mpl_kwargs : any
        Additional kwargs that can will be passed to the 'plt.step' function
    """

    if (norm_method is not None) and (scale_factor is not None):
        raise ValueError("Only one or both of 'scale_factor' and 'norm_method' can be none")

    if ax is None:
        ax = plt.gca()
    elif not isinstance(ax, Axes):
        raise TypeError(f"Incorrect type for 'ax' (Expected {type(Axes)}, got {type(ax)})")
    
    orig_hist, bin_edges = np.histogram(x, bins, weights=weights)

    if err_type == "poisson":
        orig_err = np.sqrt(orig_hist, where=(orig_hist >= 0), out=np.zeros(orig_hist.shape))
    else:
        raise NotImplementedError("Only valid err_type is 'poisson'")
    

    if norm_method is None and scale_factor is None:
        hist = orig_hist
        err = orig_err
    elif scale_factor is not None:
        hist = orig_hist * scale_factor
        err = orig_err * scale_factor
    elif norm_method == "area":
        sf = 1 / np.diff(bin_edges) / np.sum(orig_hist)
        hist = orig_hist * sf
        err = orig_err * sf
    elif norm_method == "count":
        sf = 1 / np.sum(orig_hist)
        hist = orig_hist * sf
        err = orig_err * sf
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")

    
    
    step_bins = np.concatenate(([bin_edges[0]], bin_edges, [bin_edges[-1]]))
    step_hist = np.concatenate(([0], hist, [hist[-1], 0]))
    fill_hist = np.concatenate((hist, [hist[-1]]))
    fill_err  = np.concatenate((err, [err[-1]]))

    lines = ax.step(step_bins, step_hist, where=step, **mpl_kwargs)
    ax.fill_between(bin_edges, fill_hist + fill_err, fill_hist - fill_err,
                    alpha=0.3, step=step, color=lines[0].get_color()) # Fill between not customizeable, probably fine

    return ax, (bin_edges, hist, err)


def histerr_comparison(arrays: Sequence[ArrayLike], err_types: Sequence[str] | str = "poisson", bins: int | ArrayLike = 10, 
                       norm_methods: Sequence[str | None] | str | None = None, weights: Sequence[ArrayLike | None] | ArrayLike | None = None, 
                       scale_factors: Sequence[float | None] | float | None = None, 
                       steps: Sequence[Literal["pre", "mid", "post"] | None] | Literal["pre", "mid", "post"] | None = "post", 
                       ax: Axes | None = None, **mpl_kwargs):
    """
    Deals with a plot I seem to make *a lot*, plots a set of histograms together and creates an additional comparison at
    the bottom of the plot between the two.
    
    arrays : sequence of ArrayLike
        Set of arrays from which to build our histograms. The histogram that is compared against will always be the first entry
    err_types : str or sequence of str
        Sets the err_type value for each of the arrays, see 'histerr' for details
    bins : int or sequence of scalars or str (or a sequence of these)
        Sets the bins for each array, see 'histerr' for details
    norm_methods : str or None or sequence of str or None
        Sets the norm_method for each array, see 'histerr' for details
    weights : float or None or sequence of float or None
        Sets the weight for each element in each array. Must be the same shape as arrays.
    steps : str or sequence of str
        Sets the step for each array, see 'histerr' for details
    ax : None or Axes
        Axes to draw the histograms to. If None, axes will be created on the same figure, although a comparison plot will be attached below it.
    **mpl_kwargs : Any
        Additional kwargs that can will be passed to the 'plt.step' functions. Note, if 'colors' or 'labels' is in the kwargs instead of 'color' or 'label', each plot
        will be given a different color/label specified by the list of colors/labels
    """

    # Sanity checks
    for array in arrays:
        if not isinstance(array, Sequence):
            raise ValueError("Arrays must be an array-like sequence of arrays")
    if not isinstance(bins, (int, str)):
        arr = np.asarray(bins)
        if arr.ndim > 1:
            raise ValueError("All bins must be the same")
        
    _weights: Sequence[ArrayLike | None] = _create_sequence(weights, len(arrays))
    for j in range(len(_weights)):
        if _weights[j] is not None and np.shape(arrays[j]) != np.shape(_weights[j]):
            raise ValueError(f"The shape of weights[{j}] ({np.shape(_weights[j])}) is not None and does not match the length of arrays[{j}] ({np.shape(arrays[j])})")
    
    err_types = _create_sequence(err_types, len(arrays))
    norm_methods = _create_sequence(norm_methods, len(arrays))
    scale_factors = _create_sequence(scale_factors, len(arrays))
    steps = _create_sequence(steps, len(arrays))

    # Special kwarg checks
    if (color := mpl_kwargs.pop("color", None)) is not None:
        colors = _create_sequence(color, len(arrays))
    else:
        colors = _create_sequence(mpl_kwargs.pop("colors", None), len(arrays))
    if (label := mpl_kwargs.pop("label", None)) is not None:
        labels = _create_sequence(label, len(arrays))
    else:
        labels = _create_sequence(mpl_kwargs.pop("labels", None), len(arrays))
    
    if ax is None:
        f = plt.gcf()
        ax = plt.gca()
    elif not isinstance(ax, Axes):
        raise TypeError(f"Incorrect type for 'ax' (Expected {type(Axes)}, got {type(ax)})")
    
    # Hardcoded parameters, might want to change, but it's probably fine
    ax2: Axes = make_axes_locatable(ax).append_axes("bottom", size="40%", pad=0.2)

    bin_edges_list = []
    hist_list = []
    err_list = []

    for i in range(len(arrays)):
        if i == 0:
            zorder=2.1 # draw comparison on top
        else:
            zorder=2 # default value

        _, (bin_edges, hist, err) = histerr(arrays[i], err_type=err_types[i], bins=bins, 
                                         norm_method=norm_methods[i], scale_factor=scale_factors[i],
                                         weights=_weights[i], step=steps[i], ax=ax, color=colors[i], 
                                         label=labels[i], zorder=zorder, **mpl_kwargs)
        
        # Apply same color from most recently plotted line
        color = ax.get_lines()[-1].get_color()

        bin_edges_list.append(bin_edges)
        hist_list.append(hist)
        err_list.append(err)
        
        step_bins = np.concatenate(([bin_edges[0]], bin_edges, [bin_edges[-1]]))
        step_hist = np.concatenate(([0], hist, [hist[-1], 0]))
        fill_hist = np.concatenate((hist, [hist[-1]]))
        fill_err  = np.concatenate((err, [err[-1]]))

        if i == 0:
            # Grab our comparison histograms
            step_hist_0 = step_hist
            fill_hist_0 = fill_hist
            ax2.hlines(1, bin_edges[0], bin_edges[-1], "k", lw=2, linestyle="dashed")
            ax2.fill_between(bin_edges, 1 - np.divide(fill_err, fill_hist, where=(fill_hist > 0), out=(np.ones(fill_hist.shape) *  2)),
                                        1 + np.divide(fill_err, fill_hist, where=(fill_hist > 0), out=(np.ones(fill_hist.shape) * -2)),
                                        alpha=0.3, step="post", color=color)
        else:
            ax2.step(step_bins, np.divide(step_hist, step_hist_0, where=(step_hist_0 > 0), out=(np.ones(step_hist.shape) * -1)), where=steps[i], color=color, **mpl_kwargs)
            ax2.fill_between(bin_edges, np.divide((fill_hist - fill_err), fill_hist_0, where=(fill_hist_0 > 0), out=(np.ones(fill_hist.shape) * -1)),
                                        np.divide((fill_hist + fill_err), fill_hist_0, where=(fill_hist_0 > 0), out=(np.ones(fill_hist.shape) * -1)),
                                        alpha=0.3, step="post", color=color)

        # These -1 values just mean "no data" and -1 seems a convienient enough number to shove off of the plot

    return (ax, ax2), (bin_edges_list, hist_list, err_list)