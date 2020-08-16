import numpy as np
from datetime import datetime
from typing import Dict
import os
from itertools import count
import matplotlib.pyplot as plt
import scipy.signal as sci
from matplotlib import markers

from logger import log_settings
from create_local_db import LocalDb
from grid_data import grid_dict

app_log = log_settings()
columns = {"id": [0, 0], "date": [1, "1"], "uni_time": [2, 2], "q_hec": [3, 3.5],
           "q_ic": [4, 4.5], "Tmc": [5, 5.5], "pressure": [6, 6.6]}
m_size = 40


class ABCalc(LocalDb):
    """
    Performs select from db and further calculations
    """
    _ids = count(0)

    def __init__(self, db_name, start_time, stop_time):
        self.id = next(self._ids)
        self.init_validator(start_time, stop_time)
        super().__init__(db_name)
        self.start_time = start_time
        self.stop_time = stop_time

    def __repr__(self):
        return "AB Calculate Class. instance: {}".format(self.id)

    @staticmethod
    def init_validator(start_time, stop_time):
        start_measurement = datetime(2018, 11, 27, 23, 59)
        end_measurement = datetime(2019, 12, 23, 1, 0)
        if stop_time < start_time:
            app_log.error("Start time must be less than the Stop time!")
            raise ValueError("Start time is larger than the Stop time")
        if start_time < start_measurement or stop_time < start_measurement:
            app_log.error(f"Wrong time limits. The data starts from: {start_measurement}")
            raise ValueError("Wrong time limits")
        if start_time > end_measurement or stop_time > end_measurement:
            app_log.error(f"Wrong time limits. The data ends : {end_measurement}")
            raise ValueError("Wrong time limits")

    @staticmethod
    def choose_column(col_name: str, array: np.ndarray):
        """
        requests data from selected column
        """
        if col_name in columns:
            res = list()
            if type(columns.get(col_name)[1]) is str:
                for ii in array:
                    res.append(datetime.strptime(ii[columns.get(col_name)[0]],
                                               "%Y-%m-%d %H:%M:%S"))
            else:
                for ii in array:
                    res.append(ii[columns.get(col_name)[0]])
            return np.array(res)
        else:
            app_log.error(f"Wrong column name requested: {col_name}")
            return None

    def choose_columns_list(self, col_list, array):
        res_dict: Dict = dict()
        for item in col_list:
            res_dict.update({item: self.choose_column(item, array)})
        return res_dict

    @staticmethod
    def derivative_calc(data):
        """
        Calculates derivative of data column
        """
        try:
            wind = 31
            poly = 1
            der = sci.savgol_filter(data, window_length=wind,
                                    polyorder=poly, deriv=1)
            return der
        except Exception as ex:
            app_log.error(f"Can not obtain derivative: {ex}")

    def obtain_data_dict(self, array) -> Dict:
        """
        Create a data dictionary
        """
        try:
            arr_dict = self.choose_columns_list(["Tmc", "date", "uni_time",
                                                 "q_hec", "q_ic", "pressure"], array)
            # calculate derivatives and update dict
            arr_dict.update({"der_hec": self.derivative_calc(arr_dict.get("q_hec"))})
            arr_dict.update({"der_ic": self.derivative_calc(arr_dict.get("q_ic"))})
            return arr_dict
        except Exception as ex:
            app_log.error(f"Can not create data dict: {ex}")

    def calc_params(self, array):
        """
        Obtains points of derivative jumps. Returns those points. and the rate
        """
        try:
            arr_dict = self.obtain_data_dict(array)
            # calculate temperarure rate
            idx = np.isfinite(arr_dict.get("uni_time")) & np.isfinite(arr_dict.get("Tmc"))
            fit = np.polyfit(arr_dict.get("uni_time")[idx], arr_dict.get("Tmc")[idx], 1)
            rate = fit[0] * 3600
            fitted_temp = np.poly1d(fit)
            arr_dict.update({"temp_rate": rate})
            arr_dict.update({"fitted_temp": fitted_temp(arr_dict.get("uni_time"))})
            # get pressure
            pr_mean = np.nanmean(arr_dict.get("pressure"))
            arr_dict.update({"mean pressure": pr_mean})
            # first extremuma
            ind1_extr_hec = \
                np.argmax(np.abs(arr_dict.get("der_hec")[0:round(0.4 * len(arr_dict.get("der_hec")))]))
            ind1_extr_ic = \
                np.argmax(np.abs(arr_dict.get("der_ic")[0:round(0.4 * len(arr_dict.get("der_ic")))]))
            arr_dict.update({"first_idx_hec": ind1_extr_hec})
            arr_dict.update({"first_idx_ic": ind1_extr_ic})
            # second extrema
            ind2_extr_hec = \
                np.argmax(np.abs(arr_dict.get("der_hec")[round(0.6 * len(arr_dict.get("der_hec"))):-1]))
            ind2_extr_ic = \
                np.argmax(np.abs(arr_dict.get("der_ic")[round(0.6 * len(arr_dict.get("der_ic"))):-1]))
            ind2_extr_hec += round(0.6 * len(arr_dict.get("der_hec")))
            ind2_extr_ic += round(0.6 * len(arr_dict.get("der_ic")))
            arr_dict.update({"second_idx_hec": ind2_extr_hec})
            arr_dict.update({"second_idx_ic": ind2_extr_ic})
            app_log.debug(f"Parameters calculated successfully")
        except Exception as ex:
            app_log.error(f"Can not calculate parameters: {ex}")
            return None
        else:
            app_log.info("Temperature sweep parameters are:")
            app_log.info(f"Pressure: {pr_mean} bar")
            app_log.info(f"Tc acc to Greywall: {self.tc_greywall(pr_mean)} mK")
            app_log.info(f"The ramp: {rate} mK/hr")
            app_log.info("First extr for HEC: {} mK".format(fitted_temp(arr_dict.get("uni_time")[ind1_extr_hec])))
            app_log.info("First extr for IC: {} mK".format(fitted_temp(arr_dict.get("uni_time")[ind1_extr_ic])))
            app_log.info("Second extr for HEC: {} mK".format(fitted_temp(arr_dict.get("uni_time")[ind2_extr_hec])))
            app_log.info("Second extr for IC: {} mK".format(fitted_temp(arr_dict.get("uni_time")[ind2_extr_ic])))
            app_log.info("Start temperature: {} mK".format(fitted_temp(arr_dict.get("uni_time")[0])))
            app_log.info("End temperature: {} mK".format(fitted_temp(arr_dict.get("uni_time")[-1])))
            return arr_dict

    def simple_plot(self, fig_num: int, array: np.ndarray) -> None:
        """
        Plots selected data
        """
        try:
            dat = self.choose_column("date", array)
            q_hec = self.choose_column("q_hec", array)
            q_ic = self.choose_column("q_ic", array)
            fig1 = plt.figure(fig_num, clear=True)
            ax1 = fig1.add_subplot(111)
            ax1.set_ylabel('Q')
            ax1.set_xlabel('date')
            ax1.set_title("Simple plot Q vs date")
            ax1.scatter(dat, q_hec, color='green', s=0.5, label='HEC')
            ax1.scatter(dat, q_ic, color='blue', s=0.5, label='IC')
            ax1.set_xlim(self.start_time, self.stop_time)
            ax1.legend()
            plt.gcf().autofmt_xdate()
            plt.grid()
            plt.show()
            app_log.debug("Simple figure plotted")
        except Exception as ex:
            app_log.error(f"Can not plot Simple: {ex}")

    def regular_ab_plot(self, fig_num: int, arr_dict: Dict):
        try:
            fig1 = plt.figure(fig_num, clear=True)
            # first quater Q vs date
            ax1 = fig1.add_subplot(221)
            ax1.set_ylabel('Q')
            ax1.set_xlabel('date')
            ax1.set_title("Q vs date for " + str(arr_dict.get("mean pressure")) + " bar")
            ax1.scatter(arr_dict.get("date"), arr_dict.get("q_hec"), color='green', s=0.5, label='HEC')
            ax1.scatter(arr_dict.get("date"), arr_dict.get("q_ic"), color='blue', s=0.5, label='IC')
            ax1.scatter(arr_dict.get("date")[arr_dict.get("first_idx_hec")],
                        arr_dict.get("q_hec")[arr_dict.get("first_idx_hec")], color='green', marker='s', s=m_size)
            ax1.scatter(arr_dict.get("date")[arr_dict.get("second_idx_hec")],
                        arr_dict.get("q_hec")[arr_dict.get("second_idx_hec")], color='green', marker='^', s=m_size)
            ax1.scatter(arr_dict.get("date")[arr_dict.get("first_idx_ic")],
                        arr_dict.get("q_ic")[arr_dict.get("first_idx_ic")], color='blue', marker='s', s=m_size)
            ax1.scatter(arr_dict.get("date")[arr_dict.get("second_idx_ic")],
                        arr_dict.get("q_ic")[arr_dict.get("second_idx_ic")], color='blue', marker='^', s=m_size)
            ax1.set_xlim(self.start_time, self.stop_time)
            ax1.legend()
            plt.gcf().autofmt_xdate()
            plt.grid()
            # second quater Tmc vs Date
            ax2 = fig1.add_subplot(222)
            ax2.set_ylabel(r'$T_{MC}$')
            ax2.set_xlabel('Date')
            ax2.set_title(r'$T_{MC}$ vs time for both forks')
            ax2.scatter(arr_dict.get("date"), arr_dict.get("Tmc"), color='green', s=0.5, label="raw")
            if arr_dict.get("fitted_temp", None) is not None:
                ax2.scatter(arr_dict.get("date"), arr_dict.get("fitted_temp"), color='red', s=0.5, label="fit")
                ax2.scatter(arr_dict.get("date")[arr_dict.get("first_idx_hec")],
                            arr_dict.get("fitted_temp")[arr_dict.get("first_idx_hec")], color='green', marker='s', s=m_size)
                ax2.scatter(arr_dict.get("date")[arr_dict.get("second_idx_hec")],
                            arr_dict.get("fitted_temp")[arr_dict.get("second_idx_hec")], color='green', marker='^', s=m_size)
                ax2.scatter(arr_dict.get("date")[arr_dict.get("first_idx_ic")],
                            arr_dict.get("fitted_temp")[arr_dict.get("first_idx_ic")], color='blue', marker='s', s=m_size)
                ax2.scatter(arr_dict.get("date")[arr_dict.get("second_idx_ic")],
                            arr_dict.get("fitted_temp")[arr_dict.get("second_idx_ic")], color='blue', marker='^', s=m_size)
            ax2.set_xlim(self.start_time, self.stop_time)
            ax2.legend()
            plt.gcf().autofmt_xdate()
            plt.grid()
            # third quater dQ/dT vs Date
            ax3 = fig1.add_subplot(223)
            ax3.set_ylabel('dQ/dt')
            ax3.set_xlabel('date')
            ax3.set_title("derivative vs time")
            ax3.scatter(arr_dict.get("date"), arr_dict.get("der_hec"), color='green', s=0.5, label='HEC')
            ax3.scatter(arr_dict.get("date"), arr_dict.get("der_ic"), color='blue', s=0.5, label='IC')
            ax3.scatter(arr_dict.get("date")[arr_dict.get("first_idx_hec")],
                        arr_dict.get("der_hec")[arr_dict.get("first_idx_hec")], color='green', marker='s', s=m_size)
            ax3.scatter(arr_dict.get("date")[arr_dict.get("second_idx_hec")],
                        arr_dict.get("der_hec")[arr_dict.get("second_idx_hec")], color='green', marker='^', s=m_size)
            ax3.scatter(arr_dict.get("date")[arr_dict.get("first_idx_ic")],
                        arr_dict.get("der_ic")[arr_dict.get("first_idx_ic")], color='blue', marker='s', s=m_size)
            ax3.scatter(arr_dict.get("date")[arr_dict.get("second_idx_ic")],
                        arr_dict.get("der_ic")[arr_dict.get("second_idx_ic")], color='blue', marker='^', s=m_size)
            ax3.set_xlim(self.start_time, self.stop_time)
            # ax3.legend()
            plt.gcf().autofmt_xdate()
            plt.grid()
            # fourth quater. pressure vs Date
            ax4 = fig1.add_subplot(224)
            ax4.set_ylabel('Pressure')
            ax4.set_xlabel('date')
            ax4.set_title('Pressure vs time')
            ax4.scatter(arr_dict.get("date"), arr_dict.get("pressure"), color='green', s=0.5)
            ax4.set_xlim(self.start_time, self.stop_time)
            plt.gcf().autofmt_xdate()
            plt.grid()
            plt.show()
            app_log.info("Figure is plotted")
        except Exception as ex:
            app_log.error(f"Can not plot regular figure {ex}")

    def params_pressure(self, array):
        arr_dict = self.obtain_data_dict(array)
        try:
            ind1_extr_hec = np.argmax(np.abs(arr_dict.get("der_hec")))
            ind1_extr_ic = np.argmax(np.abs(arr_dict.get("der_ic")))
            arr_dict.update({"first_idx_hec": ind1_extr_hec})
            arr_dict.update({"first_idx_ic": ind1_extr_ic})
            arr_dict.update({"second_idx_hec": ind1_extr_hec})
            arr_dict.update({"second_idx_ic": ind1_extr_ic})
        except Exception as ex:
            app_log.error(f"Can not obtain parameters from pressure sweep: {ex}")
        else:
            app_log.info("Pressure sweep parameters are:")
            app_log.info("Pab in HEC {} bar".format(arr_dict.get("pressure")[ind1_extr_hec]))
            app_log.info("Tab in HEC {} mK".format(arr_dict.get("Tmc")[ind1_extr_hec]))
            app_log.info("Pab in IC {} bar".format(arr_dict.get("pressure")[ind1_extr_ic]))
            app_log.info("Tab in IC {} mK".format(arr_dict.get("Tmc")[ind1_extr_ic]))
            return arr_dict

    @staticmethod
    def tc_greywall(p):
        """
        Obtains Tc according to Greywall
        """
        a0 = 0.929383750000000
        a1 = 0.138671880000000
        a2 = -0.006930218500000
        a3 = 0.000256851690000
        a4 = -0.000005724864400
        a5 = 5.30E-08
        return a0 + a1*p + a2*p**2 + a3*p**3 + a4*p**4 + a5*p**5

    def calculate_temperature_sweep(self):
        app_log.info("Temperature sweep is calculating...")
        self.open_session()
        arr = self.select_time(self.start_time, self.stop_time)
        arr_dict = self.calc_params(arr)
        self.regular_ab_plot(1, arr_dict)
        self.close_session()
        self.close_engine()

    def calculate_pressure_sweep(self):
        app_log.info("Pressure sweep is calculating...")
        self.open_session()
        arr = self.select_time(self.start_time, self.stop_time)
        arr_dict = self.params_pressure(arr)
        self.regular_ab_plot(1, arr_dict)
        self.close_session()
        self.close_engine()


if __name__ == "__main__":
    app_log.info("Calculation app starts.")
    db_name = "ab_data_upd.db"
    start = datetime(2019, 2, 21, 21, 0)
    stop = datetime(2019, 2, 22, 4, 30)
    instance = ABCalc(db_name, start, stop)
    instance.calculate_temperature_sweep()
    # instance.calculate_pressure_sweep()
    app_log.info("Calculation app ends")

