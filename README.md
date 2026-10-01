# MrrpPlotLib
This repo contains code for creating 'histerr' plots (Just histogram plots with errorbars). The errorbars are expressed as translucent fills between the upper and lower bounds of the error. I utilize these plots a lot so I made this repo to make my life easier. Feel free to use them yourself however you want. No guarantees on things not being jank.

# How to Install
You can install this package using pip.
```bash
pip install mrrpplotlib
```

# API Docs
### mrrpplotlib.histerr(x, stat_err='poisson', syst_err=None, bins=10, norm_method=None, weights=None, scale_factor=None, ax=None, ignore_neg_weight_err=False, \*\*mpl_kwargs)

Works like a regular histogram, but additionally handles adding in error bars via by filling above and below the histogram.

* **Parameters:**
  * **x** (`ArrayLike`) – The input data to create a histogram from.
  * **stat_err** (`ArrayLike` | `str`) – The type of stat error of the histogram, stat errors apply only to the bins and their counts. You can pass in a string (like ‘poisson’)
    which will calculate the errors based on sqrt(N) of the bins counts (or sqrt(sum(w^2)) if hist is weighted) or you can pass in the
    errors directly if need be (Note: make sure your bin shapes are identical, else you may get nonsense results). Stat errs are assumed to be symmetric.
    Note: stat errors apply to the bins, while syst errors apply to weights. Also note: stat errors should be the *UNSCALED* final stat errors. So
    the errors WILL be affected if scale_factor or norm_method is passed in, but are unaffected by weights.
  * **syst_err** (`ArrayLike` | `None`) – Systematic errors on each element of the input array. Must either be the same shape as x, or have shape (len(x), 2) for 1down, 1up systematics per
    array entry. Currently, multi-dimensional arrays will not be flatted and probably won’t work as expected. Systematics errors are assumed to be non-relative
    values. These are basically intended to be errors on the weight of each element in the array. Note: syst errors should be the *UNSCALED*
    final errors. So the errors WILL be affected if scale_factor or norm_method is passed in, but are unaffected by weights.
  * **bins** (`int` | `ArrayLike`) – Binning for the histogram, same as bins in np.histogram
  * **norm_method** (`str` | `None`) – Determines the normalization method used on the histogram, can be either ‘count’ (sum of counts in histogram equals 1) or
    ‘area’ (integral of histogram/area of bins equals 1). Cannot be set at the same time as ‘weight’.
  * **weights** (`ArrayLike` | `None`) – Determines the weight for each entry in the histogram, same as weight in np.histogram.
  * **scale_factor** (`float` | `None`) – Determines a flat scaling factor to multiply our array by. Cannot be set at the same time as ‘norm_method’.
  * **ax** (`Axes` | `None`) – Pass in an optional Axes parameter to have the plot apply to that axis rather than creating a new one.
  * **ignore_neg_weight_err** (`bool`) – Sets whether or not to ignore negative weights when calculating statistical error. Negative weights will be set to zero for such calculations.
  * **\*\*mpl_kwargs** (*Any*) – Additional kwargs that can will be passed to the ‘plt.step’ function.
* **Returns:**
  * **ax** (*Axes*) – Axes of the plot that is drawn to.
  * **bin_edges** (*ndarray*) – Bin edges of the histogram `(length(hist)+1).`
  * **hist** (*ndarray*) – The values of the histogram.
  * **err_down** (*ndarray*) – The values of the lower bounds for the error bars for the histogram.
  * **err_up** (*ndarray*) – The values of the upper bounds for the error bars for the histogram.

### mrrpplotlib.histerr_comparison(arrays, stat_errs='poisson', syst_errs=None, bins=10, norm_methods=None, weights=None, scale_factors=None, ax=None, ignore_neg_weight_err=False, \*\*mpl_kwargs)

Deals with a plot I seem to make *a lot*, plots a set of histograms together and creates an additional ratio comparison at
the bottom of the plot between the two.

* **Parameters:**
  * **arrays** (`Sequence`[`ArrayLike`] | `ArrayLike`) – Set of arrays from which to build our histograms. The histogram that is compared against will always be the first entry.
  * **stat_errs** (`Sequence`[`ArrayLike` | `str`] | `ArrayLike` | `str`) – Sets the stat_err for each array. See histerr for more details.
  * **syst_errs** (`Sequence`[`ArrayLike` | `None`] | `ArrayLike` | `None`) – Sets the syst_err for each array. See histerr for more details.
  * **bins** (`int` | `ArrayLike`) – Sets the bins for each array. See histerr for more details.
  * **norm_methods** (`Sequence`[`str` | `None`] | `str` | `None`) – Sets the norm_method for each array. See histerr for more details.
  * **weights** (`Sequence`[`ArrayLike` | `None`] | `ArrayLike` | `None`) – Sets the weight for each element in each array. Must be the same shape as arrays.
  * **scale_factors** (`Sequence`[`float` | `None`] | `float` | `None`) – Sets the scale_factor for each array. See histerr for more details.
  * **ax** (`Axes` | `None`) – Axes to draw the histograms to. If None, axes will be created on the same figure, although a comparison plot will be attached below it.
  * **ignore_neg_weight_err** (`bool`) – Whether or not to ignore stat error contributions from negative weights. See histerr for more details.
  * **\*\*mpl_kwargs** (*Any*) – Additional kwargs that can will be passed to the ‘plt.step’ functions. Note, if ‘colors’ or ‘labels’ is in the kwargs instead of ‘color’ or ‘label’, each plot
    will be given a different color/label specified by the list of colors/labels.
* **Returns:**
  * **ax** (*Axes*) – Primary Axes of the plot that is drawn to. This is the main histogram plot.
  * **ax2** (*Axes*) – Secondary Axes of the plot that is drawn to. This is the comparison plot that compares the relative errors of our histograms.
  * **bin_edges_list** (*list*) – List of bin edges for each histogram.
  * **hist_list** (*list*) – List of values for each histogram.
  * **err_down_list** (*list*) – List of the values of the lower bounds for the error bars for each histogram.
  * **err_up_list** (*list*) – List of the values of the upper bounds for the error bars for each histogram.
