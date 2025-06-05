import sys
sys.path.append("..")
from mrrpplotlib import histerr_comparison
import numpy as np
import matplotlib.pyplot as plt


samples = np.random.normal(0, 1, (2, 100))

(ax, ax2), (bin_edges_list, hist_list, err_list) = histerr_comparison(samples, labels=("abc", "def"))
plt.show()

print(hist_list, err_list)