from datetime import datetime
import tracemalloc
import numpy as np
import warnings
from ab_calc import ABCalc
from create_local_db import LocalDb
from grid_data import grid_dict
from logger import log_settings
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
from scipy.optimize import curve_fit
from fitfile import FitClass
from plots import PlotFig

app_log = log_settings()


class CreateGrid(ABCalc):
    _fit_dict: Dict[str, Optional[FitClass]] = {"hec": None, "ic": None}

    def __init__(self, db_name: str) -> None:
        """
        :param db_name: database name
        """
        self.db_name = db_name

    def __repr__(self) -> str:
        return "Create Grid"

    def choose_single_data(self):
        """
        Takes one point and creates instance of the AB calc
        """
        num = 1
        app_log.info("Selecting data for {} bar".format(grid_dict[num]["pressure_s"]))
        tc = self.tc_greywall(grid_dict[num]["pressure_s"])
        app_log.info(f"Tc graywall is {tc} mK")
        ab_instance = ABCalc(self.db_name,
                             grid_dict[num]["hec"]["start_time"],
                             grid_dict[num]["hec"]["end_time"])
        ab_instance.open_session()
        arr_hec = ab_instance.select_time(ab_instance.start_time, ab_instance.stop_time)
        arr_dict_hec = ab_instance.obtain_data_dict(arr_hec)
        arr_ic = ab_instance.select_time(grid_dict[num]["ic"]["start_time"],
                                         grid_dict[num]["ic"]["end_time"])
        arr_dict_ic = ab_instance.obtain_data_dict(arr_ic)
        ab_instance.close_session()
        ab_instance.close_engine()
        arr_dict_hec.update({"pressure_s": grid_dict[num]["pressure_s"]})
        arr_dict_ic.update({"pressure_s": grid_dict[num]["pressure_s"]})
        tc = self.tc_greywall(arr_dict_hec.get("pressure_s", None))
        arr_dict_hec.update({"tc": tc})
        arr_dict_ic.update({"tc": tc})
        arr_dict_hec.update({"fork": "hec"})
        arr_dict_ic.update({"fork": "ic"})
        app_log.info("Data successfully created")
        return arr_dict_hec, arr_dict_ic

    @staticmethod
    def update_qic_to_qhec(arr_dict: Dict, dq_hec: float) -> None:
        """
        Shifts Q of IC to align with Q of HEC
        :param arr_dict: array dictionary for IC
        :param dq_hec: delta Q between Qhec and Qic
        """
        if arr_dict.get("q_ic", None) is not None:
            arr_dict["q_ic"] += dq_hec
            app_log.debug("Q of IC is shifted according to Q HEC")

    @staticmethod
    def update_sort_qic(arr_dict: Dict, dq_hec: float) -> None:
        """
        Shifts Q of IC to align with Q of HEC
        :param arr_dict: array dictionary for IC
        :param dq_hec: delta Q between Qhec and Qic
        """
        if arr_dict.get("q_sort", None) is not None:
            arr_dict["q_sort"] += dq_hec
            app_log.debug("Q of IC is shifted according to Q HEC")
        else:
            app_log.warning("IC Data are not sorted")

    def get_temperature_ic(self, arr_dict: Dict):
        """
        Obtain `real` temperature for IC based on temperature fit for HEC
        """
        arr_dict.update({"real_temp":
                            self.revfit_raw_qt_hec(arr_dict.get("q_ic")[0:arr_dict.get("ab_ic_ind")])})
        app_log.debug("Temperature of IC obtained")

    def rev_q(self, arr_dict: Dict) -> None:
        """
        Obtain 1/Qc - 1/Q after all shifts
        """
        if arr_dict.get("qc", None) is not None and arr_dict.get("q_sort", None) is not None:
            arr_dict.update({"reverse_q": 1/arr_dict["qc"] - 1/arr_dict["q_sort"]})
            app_log.debug(f"Reverse Q calculates for {arr_dict['fork']}")
        else:
            app_log.error(f"Reverse Q not found for {arr_dict['fork']}")

    def sqrt_t(self, arr_dict: Dict) -> None:
        """
        Obtain sqrt(1 - T/Tc)
        """
        if arr_dict.get("real_temp", None) is not None:
            arr_dict.update({"sqt": (1 - arr_dict.get("real_temp")/arr_dict.get("tc"))})
            app_log.debug(f"Sqrt of real temp is calculated for {arr_dict['fork']}")
        else:
            app_log.error(f"real temperature is not calculated for {arr_dict['fork']}")

    @staticmethod
    def find_ab_index(arr_hec: Dict, arr_ic: Dict):
        """
        Find and index of AB transition based on derivative.
        Updates dict with indexes and sorted Q up to the AB.
        """
        points_to_skip = 20
        if arr_hec.get("der_hec", None) is not None and arr_ic.get("der_ic", None) is not None:
            # find max derivative
            ind1_extr_hec = np.argmax(np.abs(arr_hec.get("der_hec")))
            ind1_extr_ic = np.argmax(np.abs(arr_ic.get("der_ic")))
            ind1_extr_hec -= points_to_skip
            ind1_extr_ic -= points_to_skip
            # indices of AB transitions
            arr_hec.update({"ab_hec_ind": ind1_extr_hec})
            arr_ic.update({"ab_ic_ind": ind1_extr_ic})
            # sort Q and update dicts
            q_hec_no = arr_hec.get("q_hec")[0:ind1_extr_hec]
            t_hec_no = arr_hec.get("Tmc")[0:ind1_extr_hec]
            p_hec = np.argsort(q_hec_no)
            q_ic_no = arr_ic.get("q_ic")[0:ind1_extr_ic]
            t_ic_no = arr_ic.get("Tmc")[0:ind1_extr_ic]
            p_ic = np.argsort(q_ic_no)
            # update dicts
            # todo: get rid of extra sort
            arr_hec.update({"q_hec_sort": q_hec_no[p_hec]})
            arr_hec.update({"q_sort": q_hec_no[p_hec]})
            arr_hec.update({"Tmc_sort": t_hec_no[p_hec]})
            arr_ic.update({"q_ic_sort": q_ic_no[p_ic]})
            arr_ic.update({"q_sort": q_ic_no[p_ic]})
            arr_ic.update({"Tmc_sort": t_ic_no[p_ic]})
            app_log.debug("AB indices found. Raw Q and Tmc sorted.")
            return ind1_extr_hec, ind1_extr_ic
        else:
            app_log.error("Derivative did not calculated yet")
            return None, None

    @staticmethod
    def get_qt_sorted(arr: Dict) -> Tuple:
        """
        Returns Q, T, and fork name from array dictionary
        """
        if arr.get("fork", None) == "hec" and arr.get("q_hec_sort", None) is not None \
                and arr.get("Tmc_sort", None) is not None:
            fork = "hec"
            q = arr.get("q_hec_sort")
            t = arr.get("Tmc_sort")
        elif arr.get("fork", None) == "ic" and arr.get("q_sort", None) is not None \
                and arr.get("Tmc_sort", None) is not None:
            q = arr.get("q_sort")
            t = arr.get("Tmc_sort")
            fork = "ic"
        else:
            app_log.error("No appropriate fork was found")
            raise AttributeError("No appropriate fork")
        return q, t, fork

    def init_fork_properties(self, fork: str) -> bool:
        """
        Sets FitClass to fit_dict is None
        """
        if self._fit_dict.get(fork, None) is None:
            self._fit_dict.update({fork: FitClass(fork)})
            return True
        else:
            return False

    def get_fit_instance(self, fork: str) -> Optional[FitClass]:
        """
        Returns the FitClass instance located in fit_dict
        """
        return self._fit_dict.get(fork)

    def fit_qt_sorted(self, arr: Dict) -> None:
        """
        Fits Tmc vs Q for either fork
        """
        try:
            q, t, fork = self.get_qt_sorted(arr)
            self.init_fork_properties(fork)
            fit_instance = self.get_fit_instance(fork)
            if fit_instance is not None:
                idx, fit = fit_instance.fit_single_raw_qt(q, t)
                arr.update({"qtfit": fit_instance.reverse_raw_qt(q)})
                if fork == "hec":
                    arr.update({"qc": q[0]})
                app_log.debug(f"Fit of sorted raw QT for {fork} was done")
            else:
                app_log.error("FitClass instance does not exists")
            # todo: continue with it
        except Exception as ex:
            app_log.error(f"Can not fit sorted {arr.get('fork', None)}: {ex}")


    def make_qt_hec(self):
        """
        1. Find_ab_index. Finds AB transitions for HEC and IC. Gets indices of transition.
        Updates with sorted Q and Tmc for both Forks.
        2. Fits HEC Q sorted vs Tmc sorted. Update with hec dict with "qtfit"
        3. Shift according to Tc Greywall. Recalibrate fit of HEC. update with "real_temp"
        4. Shift Q's to align IC to HEC
        5. Fit IC Q sorted vs Tmc sorted. Update ic dict with "qtfit" and "real_temp"
        6. Find 1/Qc - 1/Q
        7. Find sqrt(1 - T/Tc)
        """
        arr_hec, arr_ic = self.choose_single_data()
        ind1_extr_hec, ind1_extr_ic = self.find_ab_index(arr_hec, arr_ic)
        self.fit_qt_sorted(arr_hec)
        dt = arr_hec.get("tc") - arr_hec.get("qtfit")[0]
        fit_hec = self.get_fit_instance("hec")
        fit_hec.update_tc(dt)
        arr_hec["real_temp"] = fit_hec.reverse_raw_qt(arr_hec.get("q_sort"))
        PlotFig(3, arr_hec).simple_grid_plot()
        dq = arr_hec.get("q_sort")[0] - arr_ic.get("q_sort")[0]
        self.update_sort_qic(arr_ic, dq)
        arr_ic.update({"qc": arr_ic.get("q_sort")[0]})
        self.fit_qt_sorted(arr_ic)
        fit_ic = self.get_fit_instance("ic")
        arr_ic.update({"real_temp": fit_ic.reverse_raw_qt(arr_ic.get("q_sort"))})
        PlotFig(4, arr_ic).simple_grid_plot()
        self.rev_q(arr_hec)
        self.rev_q(arr_ic)
        self.sqrt_t(arr_hec)
        self.sqrt_t(arr_ic)
        PlotFig(1, arr_hec).plot_revq_sqt()
        PlotFig(2, arr_ic).plot_revq_sqt()
        app_log.info(f"hec fit params: {fit_hec.fit_raw_qt}")
        app_log.info(f"ic fit params: {fit_ic.fit_raw_qt}")
        return arr_hec, arr_ic


if __name__ == "__main__":
    tracemalloc.start()
    app_log.info("Make Grid app starts..")
    db_name = "ab_data_upd.db"
    instance = CreateGrid(db_name)
    arr_hec, arr_ic = instance.make_qt_hec()
    plt.show()
    app_log.info("Make Grid app stops.")
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics("lineno")
    for stat in stats:
        pass
        # print(stat)
