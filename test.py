# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:17:36 2019

@author: dlotnyk
"""
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

a = [0, 1, 2, 3]
b = [0, 1, 4, 9]
c = np.asarray(a)
fit = np.polyfit(a, b, 2)
print(fit, type(fit), type(c), fit[0])
