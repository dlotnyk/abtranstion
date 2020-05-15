from datetime import datetime
import numpy as np
import warnings
from ab_calc import ABCalc
from create_local_db import LocalDb
from grid_data import grid_dict
from logger import log_settings
import matplotlib.pyplot as plt
from typing import Optional, Dict

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
        num = 4
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
            app_log.error(f"Can not fit QT: {ex}")

    def rev_q(self, arr_dict: Dict) -> None:
        """
        Obtain 1/Qc - 1/Q after all shifts
        """
        if arr_dict.get("qc", None) is not None:
            q_no = arr_dict.get("q_hec")[0:arr_dict.get("ab_hec_id")]
            p_ind = np.argsort(q_no)
            q = q_no[p_ind]
            t = np.sqrt(1 - self.revfit_raw_qt_hec(q)/arr_dict.get("tc"))
            arr_dict.update({"q_hec_sort": q})
            arr_dict.update({"reverse_q": 1/arr_dict["qc"] - 1/q})
            arr_dict.update({"t_upd": t})
        else:
            app_log.error("Q_c is not found")

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
            arr_hec.update({"q_hec_sort": q_hec_no[p_hec]})
            arr_hec.update({"Tmc_sort": t_hec_no[p_hec]})
            arr_ic.update({"q_ic_sort": q_ic_no[p_ic]})
            arr_ic.update({"Tmc_sort": t_ic_no[p_ic]})
            app_log.debug("AB indices found. Raw Q and Tmc sorted.")
            return ind1_extr_hec, ind1_extr_ic
        else:
            app_log.error("Derivative did not calculated yet")
            return None, None

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
            # ax1.scatter(arr_hec.get("Tmc"), arr_hec.get("q_hec"), color='green', s=0.5, label='HEC')
            # ax1.scatter(arr_ic.get("Tmc"), arr_ic.get("q_ic"), color='blue', s=0.5, label='IC')
            ax1.legend()
            plt.grid()
            plt.show()
            app_log.info("Simple QT has been plotted")
        except Exception as ex:
            app_log.error(f"Can not plot simple QT: {ex}")


if __name__ == "__main__":
    app_log.info("Make Grid app starts..")
    db_name = "ab_data_upd.db"
    instance = CreateGrid(db_name)
    dict1, dict2 = instance.choose_single_data()
    instance.plot_simple_qt(1, dict1, dict2)
    # instance.close_engine()
    app_log.info("Make Grid app stops.")
