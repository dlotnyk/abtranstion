import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import scipy.signal as sci

from logger import log_settings
from create_local_db import LocalDb

app_log = log_settings()
columns = {"id": [0, 0], "date": [1, "1"], "uni_time": [2, 2], "q_hec": [3, 3.5],
           "q_ic": [4, 4.5], "Tmc": [5, 5.5], "pressure": [6, 6.6]}


class ABCalc(LocalDb):
    """
    Performs select from db and further calculations
    """
    def __init__(self, db_name, start_time, stop_time):
        super().__init__(db_name)
        self.start_time = start_time
        self.stop_time = stop_time

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
            app_log.info("Simple figure plotted")
        except Exception as ex:
            app_log.error(f"Can not plot Simple: {ex}")


if __name__ == "__main__":
    app_log.info("Calculation app starts.")
    db_name = "ab_data_upd.db"
    start = datetime(2018, 11, 28, 10, 0)
    stop = datetime(2018, 11, 28, 10, 10)
    instance = ABCalc(db_name, start, stop)
    instance.open_session()
    arr = instance.select_time(start, stop)
    instance.simple_plot(1, arr)
    instance.close_session()
    instance.close_engine()
    app_log.info("Calculation app ends")

