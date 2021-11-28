import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

import celeri

EPS = np.finfo(float).eps


def test_plot():
    plt.figure()
    plt.plot(np.random.rand(3), "-r")
    plt.show()


def plot_matrix_abs_log(matrix):
    plt.figure(figsize=(10, 10))
    plt.imshow(np.log10(np.abs(matrix + EPS)), cmap="plasma_r")
    plt.colorbar()
    plt.show()
