import matplotlib.pyplot as plt
from logger import log_settings

app_log = log_settings()


class PlotFig:
    def __init__(self, fig_num=None, arr_dict=None):
        self.fig_num = fig_num
        self.arr_dict = arr_dict

    def plot_revq_sqt(self):
        """
        Plots 1/Qc - 1/Q vs sqrt(1 - T/Tc)
        """
        if self.arr_dict.get("reverse_q", None) is not None and \
                self.arr_dict.get("sqt", None) is not None:
            try:
                fig1 = plt.figure(self.fig_num, clear=True)
                ax1 = fig1.add_subplot(111)
                ax1.set_xlabel(r'1/Q$_c$ - 1/Q')
                ax1.set_ylabel(r'1 - T/T$_c$')
                p = self.arr_dict.get("pressure_s")
                ax1.set_title(f"Simple plot Q vs T for {p} bar for {self.arr_dict['fork']}")
                ax1.scatter(self.arr_dict.get("reverse_q"),
                            self.arr_dict.get("real_temp"),
                            color='green', s=0.5, label='HEC')
                ax1.legend()
                plt.grid()
                app_log.info(f"Reverse figure plotted for {p} bar")
            except Exception as ex:
                app_log.error(f"Can not plot reverse: {ex}")
        else:
            app_log.error(f"Reverse data do not calculated yet")

    def simple_grid_plot(self) -> None:
        try:
            fig1 = plt.figure(self.fig_num, clear=True)
            ax1 = fig1.add_subplot(111)
            ax1.set_xlabel('Q')
            ax1.set_ylabel(r'T${_MC}$')
            ax1.set_title(r"Simple of sorted Q vs T for {}".format(self.arr_dict.get("fork")))
            if self.arr_dict.get("q_sort", None) is not None and \
                    self.arr_dict.get("Tmc_sort", None) is not None:
                q_sort = self.arr_dict.get("q_sort")
                t_sort = self.arr_dict.get("Tmc_sort")
                t_fit = self.arr_dict.get("qtfit")
                t_real = self.arr_dict.get("real_temp")
                ax1.scatter(q_sort, t_sort, color='blue', s=0.5, label="raw")
                if self.arr_dict.get("qtfit", None) is not None:
                    ax1.scatter(q_sort, t_fit, color='red', s=20, label="fit")
                if self.arr_dict.get("real_temp", None) is not None:
                    ax1.scatter(q_sort, t_real, color='green', s=20, label="real")
                ax1.legend()
                plt.grid()
                app_log.debug("Simple sorted figure plotted")
            else:
                app_log.warning("Not all data are sorted yet")
        except Exception as ex:
            app_log.error(f"Can not do a simple grid plot: {ex}")

