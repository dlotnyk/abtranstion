import numpy as np
from scipy.optimize import curve_fit
from logger import log_settings
from typing import Optional

app_log = log_settings()


class FitClass:
    """
    Fitting class. used for keep fitting coefs. Instances can be used for HEC and IC
    """
    _fit_raw_qt = None
    _reverse_raw_qt = None

    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Return Fit class"

    @property
    def fit_raw_qt(self) -> Optional[np.ndarray]:
        return self._fit_raw_qt

    @fit_raw_qt.setter
    def fit_raw_qt(self, value: np.ndarray) -> None:
        self._fit_raw_qt = value
        self._reverse_raw_qt = np.poly1d(value)

    @property
    def reverse_raw_qt(self) -> Optional[np.poly1d]:
        return self._reverse_raw_qt

    @staticmethod
    def fit_single_raw_qt(Q, T):
        """
        Fits either fork
        :param Q: np array of raw Q
        :param T: np array of raw T
        :return: fitted array and index with the first Q
        """
        poly_degree = 2
        try:
            p_ind = np.argsort(Q)
            fit = np.polyfit(Q[p_ind], T[p_ind], poly_degree)
            return p_ind[0], fit
        except Exception as ex:
            app_log.error(f"Can not fit QT: {ex} ")
