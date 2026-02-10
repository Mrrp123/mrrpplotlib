import numpy as np
from numpy.typing import ArrayLike

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from collections.abc import Sequence
from typing import Any, Literal

def _create_sequence(obj: Any, length: int) -> Sequence[Any]:
    """Creates a sequence of length ``length`` of an arbitrary object"""
    try:
        if not isinstance(obj, str) and len(obj) == length:
            return obj
        elif isinstance(obj, str):
            return [obj] * length
        else:
            raise ValueError(f"Expected object of length [{length}], got [{len(obj)}]")
    except TypeError:
        return [obj] * length
    

def histerr(x: ArrayLike, 
            stat_err: ArrayLike | str = "poisson", 
            syst_err: ArrayLike | None = None, 
            bins: int | ArrayLike = 10, 
            norm_method: str | None = None, 
            weights: ArrayLike | None = None, 
            scale_factor: float | None = None, 
            ax: Axes | None = None, 
            **mpl_kwargs):
    """
    Works like a regular histogram, but additionally handles adding in error bars via by filling above and below the histogram.

    Parameters
    ----------
    x : ArrayLike
        The input data to create a histogram from.
    stat_err : ArrayLike or str, default 'poisson'
        The type of stat error of the histogram, stat errors apply only to the bins and their counts. You can pass in a string (like 'poisson') 
        which will calculate the errors based on sqrt(N) of the bins counts (or sqrt(sum(w^2)) if hist is weighted) or you can pass in the 
        errors directly if need be (Note: make sure your bins are identical, else you may getnonsense results). Stat errs are assumed to be symmetric.
        Note: stat errors apply to the bins, while syst errors apply to weights. Also note: stat errors should be the *UNSCALED* final stat errors. So
        the errors WILL be affected if scale_factor or norm_method is passed in, but are unaffected by weights.
    syst_err : ArrayLike or None, default None
        Systematic errors on each element of the input array. Must either be the same shape as x, or have shape (len(x), 2) for 1down, 1up systematics per
        array entry. Currently, multi-dimensional arrays will not be flatted and probably won't work as expected. Systematics errors are assumed to be non-relative, 
        non-negative values. These are basically intended to be errors on the weight of each element in the array. Note: syst errors should be the *UNSCALED* 
        final errors. So the errors WILL be affected if scale_factor or norm_method is passed in, but are unaffected by weights.
    bins : int or sequence of scalars or str, default 10
        Binning for the histogram, same as bins in np.histogram
    norm_method : str or None, default None
        Determines the normalization method used on the histogram, can be either 'count' (sum of counts in histogram equals 1) or 
        'area' (integral of histogram/area of bins equals 1). Cannot be set at the same time as 'weight'.
    weights : ArrayLike or None, default None
        Determines the weight for each entry in the histogram, same as weight in np.histogram.
    scale_factor : float or None, default None
        Determines a flat scaling factor to multiply our array by. Cannot be set at the same time as 'norm_method'.
    ax : Axes or None, default None
        Pass in an optional Axes parameter to have the plot apply to that axis rather than creating a new one.
    **mpl_kwargs : any
        Additional kwargs that can will be passed to the 'plt.step' function.
    
    Returns
    -------
    ax : Axes
        Axes of the plot that is drawn to.
    bin_edges : array of dtype float
        Bin edges of the histogram ``(length(hist)+1).``
    hist : array
        The values of the histogram.
    err_down : array
        The values of the lower bounds for the error bars for the histogram.
    err_up : array
        The values of the upper bounds for the error bars for the histogram.
    """

    if (norm_method is not None) and (scale_factor is not None):
        raise ValueError("Only one or both of 'scale_factor' and 'norm_method' can be None")

    if (weights is not None) and (scale_factor is not None):
        raise ValueError("Only one or both of 'scale_factor' and 'weights' can be None")

    if ax is None:
        ax = plt.gca()
    elif not isinstance(ax, Axes):
        raise TypeError(f"Incorrect type for 'ax' (Expected {type(Axes)}, got {type(ax)})")
    
    orig_hist, bin_edges = np.histogram(x, bins, weights=weights)

    if syst_err is not None:
        syst_err = np.asarray(syst_err)
        if syst_err.ndim == 2 and syst_err.shape[1] == 2:
            syst_down = syst_err.T[0]
            syst_up = syst_err.T[1]
        elif syst_err.ndim == 2 and syst_err.shape[1] != 2 or syst_err.ndim > 2:
            raise ValueError("Unsupported syst_err array shape")
        elif syst_err.ndim == 1:
            syst_down = syst_up = syst_err
    
        if weights is None:
            weights = np.ones_like(x)
        orig_hist_1down, _ = np.histogram(x, bins, weights=weights-syst_down)
        orig_hist_1up, _   = np.histogram(x, bins, weights=weights+syst_up)

    if not isinstance(stat_err, str):
        stat_err = np.asarray(stat_err)
        if (stat_err.ndim == 1 and stat_err.shape == (len(orig_hist),)):
            if syst_err is None:
                orig_err_down = orig_err_up = stat_err
            else:
                orig_err_down = np.sqrt(stat_err**2 + (orig_hist_1down - orig_hist)**2)
                orig_err_up   = np.sqrt(stat_err**2 + (orig_hist_1up   - orig_hist)**2)
        else:
            raise ValueError(f"Incorrect dimensions for stat_err (expected ({len(orig_hist)},) got {stat_err.shape}")
        
    elif stat_err == "poisson":
        if syst_err is None and weights is None:
            orig_err_down = orig_err_up = np.sqrt(orig_hist, where=(orig_hist >= 0), out=np.zeros(orig_hist.shape))
        elif syst_err is None and weights is not None:
            orig_err_down = orig_err_up = np.sqrt(np.histogram(x, bins, weights=weights**2)[0], where=(orig_hist >= 0), out=np.zeros(orig_hist.shape))
        elif syst_err is not None and weights is None:
            orig_err_down = np.sqrt(orig_hist + (orig_hist_1down - orig_hist)**2, where=(orig_hist >= 0), out=np.zeros(orig_hist.shape))
            orig_err_up   = np.sqrt(orig_hist + (orig_hist_1up   - orig_hist)**2, where=(orig_hist >= 0), out=np.zeros(orig_hist.shape))
        else:
            orig_err_down = np.sqrt(np.histogram(x, bins, weights=weights**2)[0] + (orig_hist_1down - orig_hist)**2, where=(orig_hist >= 0), out=np.zeros(orig_hist.shape))
            orig_err_up   = np.sqrt(np.histogram(x, bins, weights=weights**2)[0] + (orig_hist_1up   - orig_hist)**2, where=(orig_hist >= 0), out=np.zeros(orig_hist.shape))
    else:
        raise NotImplementedError("Only current valid stat_err string value is 'poisson'")
    

    if norm_method is None and scale_factor is None:
        hist = orig_hist
        err_down = orig_err_down
        err_up = orig_err_up
    elif scale_factor is not None:
        hist = orig_hist * scale_factor
        err_down = orig_err_down * scale_factor
        err_up = orig_err_up * scale_factor
    elif norm_method == "area":
        sf = 1 / np.diff(bin_edges) / np.sum(orig_hist)
        hist = orig_hist * sf
        err_down = orig_err_down * sf
        err_up = orig_err_up * sf
    elif norm_method == "count":
        sf = 1 / np.sum(orig_hist)
        hist = orig_hist * sf
        err_down = orig_err_down * sf
        err_up = orig_err_up * sf
    else:
        raise ValueError(f"Unknown normalization method: {norm_method}")

    patch = ax.stairs(hist, bin_edges, baseline=0, fill=False, **mpl_kwargs)
    # Effective ax.fill_between replacement
    ax.stairs(hist + err_up, bin_edges, baseline=(hist - err_down), 
              alpha=0.3, facecolor=patch.get_edgecolor(), edgecolor=patch.get_edgecolor(),
              lw=1, fill=True)
    # Fill between not customizeable, probably fine

    return ax, (bin_edges, hist, err_down, err_up)


def histerr_comparison(arrays: Sequence[ArrayLike] | ArrayLike, 
                       stat_errs: Sequence[ArrayLike | str] | ArrayLike | str = "poisson", 
                       syst_errs: Sequence[ArrayLike | None] | ArrayLike | None = None, 
                       bins: int | ArrayLike = 10, 
                       norm_methods: Sequence[str | None] | str | None = None, 
                       weights: Sequence[ArrayLike | None] | ArrayLike | None = None, 
                       scale_factors: Sequence[float | None] | float | None = None, 
                       ax: Axes | None = None, 
                       **mpl_kwargs):
    """
    Deals with a plot I seem to make *a lot*, plots a set of histograms together and creates an additional ratio comparison at
    the bottom of the plot between the two.
    
    Parameters
    ----------
    arrays : sequence of ArrayLike
        Set of arrays from which to build our histograms. The histogram that is compared against will always be the first entry.
    stat_errs : str or ArrayLike or sequence of str or ArrayLike, default 'poisson'
        Sets the stat_err for each array. See `histerr` for more details.
    syst_errs : None or ArrayLike or sequence of None or ArrayLike, default None
        Sets the syst_err for each array. See `histerr` for more details.
    bins : int or ArrayLike, default 10
        Sets the bins for each array, see `histerr` for more details.
    norm_methods : str or None or sequence of str or None, default None
        Sets the norm_method for each array, see `histerr` for more details.
    weights : float or None or sequence of float or None, default None
        Sets the weight for each element in each array. Must be the same shape as arrays.
    scale_factors : float or None or sequence of float or None, default None
        Sets the scale_factor for each array, see `histerr` for more details.
    ax : None or Axes, default None
        Axes to draw the histograms to. If None, axes will be created on the same figure, although a comparison plot will be attached below it.
    **mpl_kwargs : Any
        Additional kwargs that can will be passed to the 'plt.step' functions. Note, if 'colors' or 'labels' is in the kwargs instead of 'color' or 'label', each plot
        will be given a different color/label specified by the list of colors/labels.

    Returns
    -------
    ax : Axes
        Primary Axes of the plot that is drawn to. This is the main histogram plot.
    ax2 : Axes
        Secondary Axes of the plot that is drawn to. This is the comparison plot that compares the relative errors of our histograms.
    bin_edges_list : list
        List of bin edges for each histogram. 
    hist_list : list
        List of values for each histogram.
    err_down_list : list
        List of the values of the lower bounds for the error bars for each histogram.
    err_up_list : list
        List of the values of the upper bounds for the error bars for each histogram.
    
    """

    # Sanity checks
    for array in arrays:
        try:
            np.asarray(array)
        except TypeError:
            TypeError("Arrays must be an array-like sequence of arrays")
    if not isinstance(bins, (int, str)):
        arr = np.asarray(bins)
        if arr.ndim > 1:
            raise ValueError("All bins must be the same")
    
    if isinstance(bins, int):
        num_bins = bins
    else:
        num_bins = len(bins) - 1 # bins also includes right edge, so subtract 1 from len
        
    _weights: Sequence[ArrayLike | None] = _create_sequence(weights, len(arrays))
    for j in range(len(_weights)):
        if _weights[j] is not None and np.shape(arrays[j]) != np.shape(_weights[j]):
            raise ValueError(f"The shape of weights[{j}] ({np.shape(_weights[j])}) is not None and does not match the length of arrays[{j}] ({np.shape(arrays[j])})")
    
    _syst_errs: Sequence[ArrayLike | None] = _create_sequence(syst_errs, len(arrays))
    for k in range(len(_syst_errs)):
        if _syst_errs[k] is not None and np.size(arrays[k]) != np.shape(_syst_errs[k])[0]:
            raise ValueError(f"The len of syst_errs[{k}] ({np.shape(_syst_errs[k])[0]}) is not None and does not match the flattened length of arrays[{k}] ({np.size(arrays[k])})")
    
    _stat_errs: Sequence[ArrayLike | str] = _create_sequence(stat_errs, len(arrays))
    for l in range(len(_stat_errs)):
        if not isinstance(_stat_errs[l], str) and len(_stat_errs[l]) != num_bins:
            raise ValueError(f"stat_errs[l] is not a string and len of stat_errs[{l}] ({len(_stat_errs[l])}) does not match the number of bins ({num_bins})")

    norm_methods = _create_sequence(norm_methods, len(arrays))
    scale_factors = _create_sequence(scale_factors, len(arrays))

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
    err_down_list = []
    err_up_list = []

    for i in range(len(arrays)):
        if i == 0:
            zorder=2.1 # draw comparison on top
        else:
            zorder=2 # default value

        _, (bin_edges, hist, err_down, err_up) = histerr(arrays[i], stat_err=_stat_errs[i], syst_err=_syst_errs[i], 
                                                         bins=bins, norm_method=norm_methods[i], scale_factor=scale_factors[i], 
                                                         weights=_weights[i], ax=ax, color=colors[i], 
                                                         label=labels[i], zorder=zorder, **mpl_kwargs)
        
        # Apply same color from most recently plotted line
        color = colors[i]

        bin_edges_list.append(bin_edges)
        hist_list.append(hist)
        err_down_list.append(err_down)
        err_up_list.append(err_up)

        if i == 0:
            # Grab our comparison histograms
            hist_0 = hist

            ax2.hlines(1, bin_edges[0], bin_edges[-1], "k", lw=2, linestyle="dashed")
            # Basically ax2.fill_between equivalent
            ax2.stairs(1 + np.divide(err_up, hist, where=(hist > 0), out=np.zeros(hist.shape)), bin_edges,
                      baseline=(1 - np.divide(err_down, hist, where=(hist > 0), out=np.zeros(hist.shape))),
                      facecolor=color, edgecolor=color, alpha=0.3, fill=True, lw=1, zorder=1)
        else:
            ax2.stairs(np.divide(hist, hist_0, where=(hist_0 > 0), out=(np.ones(hist.shape) * -1)), bin_edges, 
                       baseline=-1, color=color, zorder=2, **mpl_kwargs)
            # Basically ax2.fill_between equivalent
            ax2.stairs(np.divide((hist + err_up), hist_0, where=(hist_0 > 0), out=(np.ones(hist.shape) * -1)), bin_edges,
                       baseline=np.divide((hist - err_down), hist_0, where=(hist_0 > 0), out=(np.ones(hist.shape) * -1)),
                       facecolor=color, edgecolor=color, alpha=0.3, fill=True, lw=1, zorder=1)

        # These -1 values just mean "no data" and -1 seems a convienient enough number to shove off of the plot

    return (ax, ax2), (bin_edges_list, hist_list, err_down_list, err_up_list)