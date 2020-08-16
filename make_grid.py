from datetime import datetime
import numpy as np
import warnings
from ab_calc import ABCalc
from create_local_db import LocalDb
from grid_data import grid_dict
from logger import log_settings
import matplotlib.pyplot as plt
from typing import Optional, Dict
from scipy.optimize import curve_fit

app_log = log_settings()


class CreateGrid(ABCalc):
    __fit_raw_qt_hec = None
    __revfit_raw_qt_hec = None
    __fit_raw_qt_ic = None
    __revfit_raw_qt_ic = None

    def __init__(self, db_name: str) -> None:
        """
        :param db_name: database name
        """
        self.db_name = db_name

    def __repr__(self) -> str:
        return "Create Grid"

    @property
    def revfit_raw_qt_hec(self) -> Optional[np.poly1d]:
        """
        Getter/setter for rev fit of raw QT
        """
        return self.__revfit_raw_qt_hec

    def set_revfit_raw_qt_hec(self) -> None:
        if self.fit_raw_qt_hec is not None:
            self.__revfit_raw_qt_hec = np.poly1d(self.fit_raw_qt_hec)

    @property
    def revfit_raw_qt_ic(self) -> Optional[np.poly1d]:
        """
        Getter/setter for rev fit of raw QT
        """
        return self.__revfit_raw_qt_ic

    def set_revfit_raw_qt_ic(self) -> None:
        if self.fit_raw_qt_ic is not None:
            self.__revfit_raw_qt_ic = np.poly1d(self.fit_raw_qt_ic)

    @property
    def fit_raw_qt_hec(self) -> Optional[np.ndarray]:
        """
        Getter/setter for rev fit of raw QT
        """
        return self.__fit_raw_qt_hec

    @fit_raw_qt_hec.setter
    def fit_raw_qt_hec(self, value: np.ndarray) -> None:
        self.__fit_raw_qt_hec = value
        self.set_revfit_raw_qt_hec()

    @property
    def fit_raw_qt_ic(self) -> Optional[np.ndarray]:
        """
        Getter/setter for rev fit of raw QT
        """
        return self.__fit_raw_qt_ic

    @fit_raw_qt_ic.setter
    def fit_raw_qt_ic(self, value: np.ndarray) -> None:
        self.__fit_raw_qt_ic = value
        self.set_revfit_raw_qt_ic()

    def update_tc_hec(self, thec: float, tc: float) -> None:
        """
        Updates fit raw qt according to Tc shift for HEC
        :param thec: temperature of HEC
        :param tc: Tc according to Greywall
        """
        if self.fit_raw_qt_hec is not None:
            dt = tc - thec
            self.__fit_raw_qt_hec[-1] += dt
            self.set_revfit_raw_qt_hec()
            app_log.debug("HEC is shifted according to Tc Greywall")
        else:
            app_log.error("Can not update Tc for HEC. Raw QT is not fitted yet")

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

    def choose_single_data(self):
        """
        Takes one point and creates instance of the AB calc
        """
        num = 1
        app_log.info("Selecting data for {} bar".format(grid_dict[num]["pressure_s"]))
        tc = self.tc_greywall(grid_dict[num]["pressure_s"])
        app_log.info(f"Tc graywall is {tc} mK")
        hec_inst = ABCalc(self.db_name,
                          grid_dict[num]["hec"]["start_time"],
                          grid_dict[num]["hec"]["end_time"])
        hec_inst.open_session()
        arr_hec = hec_inst.select_time(hec_inst.start_time, hec_inst.stop_time)
        arr_dict_hec = hec_inst.obtain_data_dict(arr_hec)
        arr_ic = hec_inst.select_time(grid_dict[num]["ic"]["start_time"],
                                      grid_dict[num]["ic"]["end_time"])
        arr_dict_ic = hec_inst.obtain_data_dict(arr_ic)
        hec_inst.close_session()
        hec_inst.close_engine()
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

    def fit_qt_sorted_hec(self, arr_hec: Dict) -> None:
        """
        Fits Tmc vs Q for HEC. Uses Q_sort obtained from find extrema
        """
        try:
            if arr_hec.get("q_hec_sort", None) is not None and arr_hec.get("Tmc_sort", None) is not None:
                q = arr_hec.get("q_hec_sort")
                t = arr_hec.get("Tmc_sort")
                idx, fit = self.fit_single_raw_qt(q, t)
                self.fit_raw_qt_hec = fit
                arr_hec.update({"qtfit":
                                    self.revfit_raw_qt_hec(q)})
                arr_hec.update({"qc": q[0]})
                app_log.debug("Fit of sorted raw QT for HEC is done")
            else:
                app_log.error("AB for HEC is not found yet")
        except Exception as ex:
            app_log.error(f"Can not fit sorted HEC: {ex}")

    def fit_qt_sorted_ic(self, arr_ic: Dict) -> None:
        """
        Fits Tmc vs Q for IC. Uses Q_sort obtained from find extrema
        """
        try:
            if arr_ic.get("q_sort", None) is not None and arr_ic.get("Tmc_sort", None) is not None:
                q = arr_ic.get("q_sort")
                t = arr_ic.get("Tmc_sort")
                idx, fit = self.fit_single_raw_qt(q, t)
                self.fit_raw_qt_ic = fit
                arr_ic.update({"qtfit":
                                    self.revfit_raw_qt_ic(q)})
                app_log.debug("Fit of sorted raw QT for IC is done")
            else:
                app_log.error("AB for IC is not found yet")
        except Exception as ex:
            app_log.error(f"Can not fit sorted IC: {ex}")

    def fit_raw_hec(self, arr_hec: Dict):
        """
        Fits raw data T vs Q for HEC
        Updates main dict. with raw fit data, Qc value
        """
        ind = arr_hec.get("ab_hec_ind", None)
        if ind is not None:
            q_hec = arr_hec.get("q_hec")[0:ind]
            t_hec = arr_hec.get("Tmc")[0:ind]
            id_hec, fit_hec = self.fit_single_raw_qt(q_hec, t_hec)
            self.fit_raw_qt_hec = fit_hec
            arr_hec.update({"qtfit":
                                self.revfit_raw_qt_hec(arr_hec.get("q_hec")[0:ind])})
            arr_hec.update({"qc": q_hec[id_hec]})
            app_log.debug("Fit of raw QT for HEC is done")
        else:
            app_log.error("AB for HEC is not found yet")

    def fit_raw_qt(self, arr_hec: Dict, arr_ic: Dict):
        """
        Polyfit for T vs Q for both forks
        """
        try:
            ind1_extr_hec, ind1_extr_ic = self.find_ab_index(arr_hec, arr_ic)
            # fit HEC
            q_hec = arr_hec.get("q_hec")[0:ind1_extr_hec]
            t_hec = arr_hec.get("Tmc")[0:ind1_extr_hec]
            id_hec, fit_hec = self.fit_single_raw_qt(q_hec, t_hec)
            self.fit_raw_qt_hec = fit_hec
            arr_hec.update({"qtfit":
                                self.revfit_raw_qt_hec(arr_hec.get("q_hec")[0:ind1_extr_hec])})
            arr_hec.update({"qc": q_hec[id_hec]})
            arr_ic.update({"qc": q_hec[id_hec]})
            # fit IC
            q_ic = arr_ic.get("q_ic")[0:ind1_extr_ic]
            t_ic = arr_ic.get("Tmc")[0:ind1_extr_ic]
            id_ic, fit_ic = self.fit_single_raw_qt(q_ic, t_ic)
            self.fit_raw_qt_ic = fit_ic
            # shift according to Tc graywall
            t_1 = self.revfit_raw_qt_hec(q_hec[id_hec])
            tc = arr_hec.get("tc", None)
            self.update_tc_hec(t_1, tc)
            # align Q's at Tc
            dq = q_hec[id_hec] - q_ic[id_ic]
            self.update_qic_to_qhec(arr_ic, dq)
            # fit IC again with sifted Q
            q_ic = arr_ic.get("q_ic")[0:ind1_extr_ic]
            t_ic = arr_ic.get("Tmc")[0:ind1_extr_ic]
            id_ic, fit_ic = self.fit_single_raw_qt(q_ic, t_ic)
            self.fit_raw_qt_ic = fit_ic
            # update dictionaries
            arr_hec.update({"real_temp":
                                self.revfit_raw_qt_hec(arr_hec.get("q_hec")[0:ind1_extr_hec])})
            arr_ic.update({"qtfit":
                               self.revfit_raw_qt_ic(arr_ic.get("q_ic")[0:ind1_extr_ic])})
            self.get_temperature_ic(arr_ic)
            self.rev_q(arr_hec)
        except Exception as ex:
            app_log.error(f"Can not fit raw QT: {ex}")

    def plot_simple_qt(self, fig_num, arr_hec, arr_ic):
        """
        Plot simple T vs Q or otherwise for both forks
        """
        self.fit_raw_qt(arr_hec, arr_ic)
        self.plot_revq_sqt(2, arr_hec)
        try:
            fig1 = plt.figure(fig_num, clear=True)
            ax1 = fig1.add_subplot(111)
            ax1.set_ylabel('T')
            ax1.set_xlabel('Q')
            p = arr_hec.get("pressure_s")
            ax1.set_title(f"Simple plot Q vs T for {p} bar")
            if arr_hec.get("ab_hec_ind", None) is not None and arr_ic.get("ab_ic_ind", None) is not None:
                ax1.scatter(arr_hec.get("q_hec")[0:arr_hec.get("ab_hec_ind")],
                            arr_hec.get("Tmc")[0:arr_hec.get("ab_hec_ind")],
                            color='green', s=0.5, label='HEC')
                ax1.scatter(arr_ic.get("q_ic")[0:arr_ic.get("ab_ic_ind")],
                            arr_ic.get("Tmc")[0:arr_ic.get("ab_ic_ind")],
                            color='blue', s=0.5, label='IC')
                ax1.scatter(arr_hec.get("q_hec")[0:arr_hec.get("ab_hec_ind")],
                            arr_hec.get("real_temp")[0:arr_hec.get("ab_hec_ind")],
                            color='red', s=15, label='fit real HEC')
                ax1.scatter(arr_hec.get("q_hec")[0:arr_hec.get("ab_hec_ind")],
                            arr_hec.get("qtfit")[0:arr_hec.get("ab_hec_ind")],
                            color='black', s=5, label='fit HEC')
                ax1.scatter(arr_ic.get("q_ic")[0:arr_ic.get("ab_ic_ind")],
                            arr_ic.get("qtfit")[0:arr_ic.get("ab_ic_ind")],
                            color='magenta', s=5, label='fit IC')
                ax1.scatter(arr_ic.get("q_ic")[0:arr_ic.get("ab_ic_ind")],
                            arr_ic.get("real_temp")[0:arr_ic.get("ab_ic_ind")],
                            color='cyan', s=5, label='fit real IC')
            ax1.legend()
            plt.grid()
            app_log.info("Simple QT has been plotted")
        except Exception as ex:
            app_log.error(f"Can not plot simple QT: {ex}")

    def plot_revq_sqt(self, fig_num, arr_dict):
        """
        Plots 1/Qc - 1/Q vs sqrt(1 - T/Tc)
        :param fig_num: Figure number
        """
        if arr_dict.get("reverse_q", None) is not None and \
            arr_dict.get("sqt", None) is not None:
            try:
                fig1 = plt.figure(fig_num, clear=True)
                ax1 = fig1.add_subplot(111)
                ax1.set_xlabel(r'1/Q$_c$ - 1/Q')
                ax1.set_ylabel(r'1 - T/T$_c$')
                p = arr_dict.get("pressure_s")
                ax1.set_title(f"Simple plot Q vs T for {p} bar for {arr_dict['fork']}")
                ax1.scatter(arr_dict.get("reverse_q"),
                            arr_dict.get("real_temp"),
                            color='green', s=0.5, label='HEC')
                ax1.legend()
                plt.grid()
                app_log.info(f"Reverse figure plotted for {p} bar")
            except Exception as ex:
                app_log.error(f"Can not plot reverse: {ex}")
        else:
            app_log.error(f"Reverse data do not calculated yet")

    @staticmethod
    def simple_grid_plot(fig_num: int, arr: Dict) -> None:
        try:
            fig1 = plt.figure(fig_num, clear=True)
            ax1 = fig1.add_subplot(111)
            ax1.set_xlabel('Q')
            ax1.set_ylabel(r'T${_MC}$')
            ax1.set_title(r"Simple of sorted Q vs T for {}".format(arr.get("fork")))
            if arr.get("q_sort", None) is not None and \
                    arr.get("Tmc_sort", None) is not None:
                q_sort = arr.get("q_sort")
                t_sort = arr.get("Tmc_sort")
                t_fit = arr.get("qtfit")
                t_real = arr.get("real_temp")
                ax1.scatter(q_sort, t_sort, color='blue', s=0.5, label="raw")
                if arr.get("qtfit", None) is not None:
                    ax1.scatter(q_sort, t_fit, color='red', s=20, label="fit")
                if arr.get("real_temp", None) is not None:
                    ax1.scatter(q_sort, t_real, color='green', s=20, label="real")
                ax1.legend()
                plt.grid()
                app_log.debug("Simple sorted figure plotted")
            else:
                app_log.warning("Not all data are sorted yet")
        except Exception as ex:
            app_log.error(f"Can not do a simple grid plot: {ex}")

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
        self.fit_qt_sorted_hec(arr_hec)
        self.update_tc_hec(arr_hec.get("qtfit")[0], arr_hec.get("tc"))
        arr_hec["real_temp"] = self.revfit_raw_qt_hec(arr_hec.get("q_sort"))
        self.simple_grid_plot(3, arr_hec)
        dq = arr_hec.get("q_sort")[0] - arr_ic.get("q_sort")[0]
        self.update_sort_qic(arr_ic, dq)
        arr_ic.update({"qc": arr_ic.get("q_sort")[0]})
        self.fit_qt_sorted_ic(arr_ic)
        arr_ic.update({"real_temp": self.revfit_raw_qt_hec(arr_ic.get("q_sort"))})
        self.simple_grid_plot(4, arr_ic)
        self.rev_q(arr_hec)
        self.rev_q(arr_ic)
        self.sqrt_t(arr_hec)
        self.sqrt_t(arr_ic)
        self.plot_revq_sqt(1, arr_hec)
        self.plot_revq_sqt(2, arr_ic)
        app_log.info(f"hec fit params: {self.fit_raw_qt_hec}")
        app_log.info(f"ic fit params: {self.fit_raw_qt_ic}")
        return arr_hec, arr_ic


if __name__ == "__main__":
    app_log.info("Make Grid app starts..")
    db_name = "ab_data_upd.db"
    instance = CreateGrid(db_name)
    arr_hec, arr_ic = instance.make_qt_hec()
    plt.show()
    app_log.info("Make Grid app stops.")
