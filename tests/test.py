import sys
sys.path.append("..")
from mrrpplotlib import histerr_comparison
import numpy as np
import matplotlib.pyplot as plt


def test_histerr_comparison():

    f, axes = plt.subplots(ncols=3, nrows=1)

    samples = np.random.normal(0, 1, (2, 1000))
    weights = np.random.chisquare(1, samples.shape)
    weights2 = np.ones(1000) * 0.1
    syst_errs = (np.asarray([0.001, 1]*1000).reshape(1000,2), None)

    histerr_comparison(samples, bins=np.arange(-5, 5.1, 0.5), labels=("abc", "def"), weights=weights,
                       syst_errs=syst_errs, lw=2, ax=axes[0], colors=("red", "green"))
    histerr_comparison(samples, bins=np.arange(-5, 5.1, 0.5), labels=("abc", "def"), weights=weights,
                       lw=2, ax=axes[1], colors=("red", "green"))
    histerr_comparison((samples[0], samples[0]), bins=np.arange(-5, 5.1, 0.5), labels=("abc", "def"), 
                       weights=(None, weights2), scale_factors=(0.1, None), lw=2, ax=axes[2], 
                       colors=("red", "blue")) 

test_histerr_comparison()
plt.show()