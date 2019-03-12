"""
Created on Tue MAR 12 13:48 2019

@autor: dlotnyk
"""

from functools import wraps
import numpy as np
import sqlite3 as sql
import datetime
import inspect
import os
import matplotlib.pyplot as plt
import scipy.signal as sci
from abres import ABData
import unittest
import warnings
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings("ignore", message="Reloaded modules: \
        <chromosome_length>")


class TestAbres(unittest.TestCase):
    '''testing of the ABData class methods'''
    @classmethod
    def setUpClass(cls):
        '''runs once at py calls'''
        ABData.table_name = 'testtb'
        print('test')


if __name__ == '__main__':
    unittest.main()
