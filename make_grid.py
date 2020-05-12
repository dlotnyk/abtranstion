from datetime import datetime
import numpy as np
from ab_calc import ABCalc
from create_local_db import LocalDb
from grid_data import grid_dict
from logger import log_settings
import matplotlib.pyplot as plt

app_log = log_settings()


class CreateGrid(ABCalc):
    __fit_raw_qt_hec = None
    __revfit_raw_qt_hec = None
    __fit_raw_qt_ic = None
    __revfit_raw_qt_ic = None

    def __init__(self, db_name):
        self.db_name = db_name

    def __repr__(self):
        return "Create Grid"

    def choose_single_data(self):
        """
        Takes one point and creates instance of the AB calc
        """
        num = 0
        app_log.info("Selecting data for {} bar".format(grid_dict[num]["pressure_s"]))
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
            return np.poly1d(fit), p_ind[0]
        except Exception as ex:
            app_log.error(f"Can not fit QT: {ex}")

    def fit_raw_qt(self, arr_hec, arr_ic):
        """
        Polyfit for T vs Q for both forks
        """
        points_to_skip = 20
        try:
            ind1_extr_hec = np.argmax(np.abs(arr_hec.get("der_hec")))
            ind1_extr_ic = np.argmax(np.abs(arr_ic.get("der_ic")))
            ind1_extr_hec -= points_to_skip
            ind1_extr_ic -= points_to_skip
            arr_hec.update({"ab_hec_ind": ind1_extr_hec})
            arr_ic.update({"ab_ic_ind": ind1_extr_ic})
            # fit HEC
            q_hec = arr_hec.get("q_hec")[0:ind1_extr_hec]
            t_hec = arr_hec.get("Tmc")[0:ind1_extr_hec]
            rev_hec, id_hec = self.fit_single_raw_qt(q_hec, t_hec)
            t_1 = rev_hec(q_hec[id_hec])
            tc = self.tc_greywall(arr_hec.get("pressure_s", 22.5))
            print(f"T = {t_1}; Tc = {tc}")
            arr_hec.update({"qtfit": rev_hec(arr_hec.get("q_hec")[0:ind1_extr_hec])})
            # fit IC
            q_ic = arr_ic.get("q_ic")[0:ind1_extr_ic]
            t_ic = arr_ic.get("Tmc")[0:ind1_extr_ic]
            rev_ic, id_ic = self.fit_single_raw_qt(q_ic, t_ic)
            t_2 = rev_ic(q_ic[id_ic])
            print(f"Thec = {t_1}; Tic = {t_2}; Tc = {tc}")
            arr_ic.update({"qtfit": rev_ic(arr_ic.get("q_ic")[0:ind1_extr_ic])})
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
            ax1.set_title("Simple plot Q vs T")
            if arr_hec.get("ab_hec_ind", None) is not None and arr_ic.get("ab_ic_ind", None) is not None:
                ax1.scatter(arr_hec.get("q_hec")[0:arr_hec.get("ab_hec_ind")],
                            arr_hec.get("Tmc")[0:arr_hec.get("ab_hec_ind")],
                            color='green', s=0.5, label='HEC')
                ax1.scatter(arr_ic.get("q_ic")[0:arr_ic.get("ab_ic_ind")],
                            arr_ic.get("Tmc")[0:arr_ic.get("ab_ic_ind")],
                            color='blue', s=0.5, label='IC')
                ax1.scatter(arr_hec.get("q_hec")[0:arr_hec.get("ab_hec_ind")],
                            arr_hec.get("qtfit")[0:arr_hec.get("ab_hec_ind")],
                            color='red', s=5, label='fit HEC')
                ax1.scatter(arr_ic.get("q_ic")[0:arr_ic.get("ab_ic_ind")],
                            arr_ic.get("qtfit")[0:arr_ic.get("ab_ic_ind")],
                            color='magenta', s=5, label='fit IC')
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
