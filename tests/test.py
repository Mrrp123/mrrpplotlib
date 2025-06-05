import sys
sys.path.append("..")
from mrrpplotlib import histerr_comparison
import numpy as np
import matplotlib.pyplot as plt


def test_histerr_comparison():
    samples = np.random.normal(0, 1, (2, 10000))
    weights = np.random.uniform(1, 1, samples.shape)

    (ax, ax2), (bin_edges_list, hist_list, err_list) = histerr_comparison(samples, bins=np.arange(-5, 5.1, 0.5), labels=("abc", "def"), weights=weights)

test_histerr_comparison()
plt.show()