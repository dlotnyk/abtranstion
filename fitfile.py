import numpy as np
from scipy.optimize import curve_fit
from logger import log_settings
from typing import Optional, Tuple

app_log = log_settings()


class FitClass:
    """
    Fitting class. used for keep fitting coefs. Instances can be used for HEC and IC
    """
    _fit_raw_qt = None
    _reverse_raw_qt = None
    _idx = None

    def __init__(self, name: str) -> None:
        """
        :param name: HEC or IC
        """
        self.name = name

    def __repr__(self) -> str:
        return "Fit class"

    @property
    def is_hec(self):
        return self.name == "hec"

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, value):
        self._idx = value

    @property
    def fit_raw_qt(self) -> Optional[np.ndarray]:
        """
        Getter and setter for fit coefficients
        """
        return self._fit_raw_qt

    @fit_raw_qt.setter
    def fit_raw_qt(self, value: np.ndarray) -> None:
        """
        set fit coefficients
        Also sets the reverse fit values
        """
        self._fit_raw_qt = value
        self._reverse_raw_qt = np.poly1d(value)
        app_log.debug("Fit params have been updated")

    @property
    def reverse_raw_qt(self) -> Optional[np.poly1d]:
        """
        Gets the reverse fit value
        """
        return self._reverse_raw_qt

    def fit_single_raw_qt(self, Q, T):
        """
        Fits either fork
        :param Q: np array of raw Q
        :param T: np array of raw T
        :return: fitted array and index with the first Q
        """
        poly_degree = 2
        # todo: add method with bounds
        try:
            p_ind = np.argsort(Q)
            fit = self.np_simple_fit(Q, T, p_ind, poly_degree)
            self.fit_raw_qt = fit
            self.idx = p_ind[0]
            return p_ind[0], fit
        except Exception as ex:
            app_log.error(f"Can not fit QT: {ex} ")

    def update_tc(self, dt: float) -> None:
        if self.fit_raw_qt is not None:
            self._fit_raw_qt[-1] += dt
            self._reverse_raw_qt = np.poly1d(self.fit_raw_qt)
            app_log.debug(f"{self.name} is shifted according to Tc Greywall")
        else:
            app_log.error(f"Can not update Tc for {self.name}."
                          f"Raw QT is not fitted yet")

    @staticmethod
    def np_simple_fit(x, y, ind, poly_degree) -> Tuple:
        """
        Simple polyfit with numpy
        :param x: x data
        :param y: y data
        :param ind: array from start to the AB transition
        :param poly_degree: poly degree
        :return: fit as ndarray
        """
        return np.polyfit(x[ind], y[ind], poly_degree)


