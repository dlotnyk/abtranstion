# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:17:36 2019

@author: dlotnyk
"""
# include needed libraries
import numpy as np
import matplotlib.pyplot as plt


# import test.txt file
def f_a(a, b):
    return zip(a, b)


a = [1, 2, 3, 4]
b = [5, 6, 7, 8]
c = f_a(a, b)
for ii, jj in enumerate(c):
    print(ii, jj)
