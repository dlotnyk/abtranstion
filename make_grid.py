from datetime import datetime
import numpy as np
from ab_calc import ABCalc
from create_local_db import LocalDb
from grid_data import grid_dict
from logger import log_settings

app_log = log_settings()


class CreateGrid(ABCalc):

    def __init__(self, db_name):
        self.db_name = db_name

    def __repr__(self):
        return "Create Grid"

    def some(self):
        """
        Takes one point and creates instance of the AB calc
        """
        num = 0
        hec_inst = ABCalc(self.db_name,
                          grid_dict[num]["hec"]["start_time"],
                          grid_dict[num]["hec"]["end_time"])
        ic_inst = ABCalc(self.db_name,
                          grid_dict[num]["ic"]["start_time"],
                          grid_dict[num]["ic"]["end_time"])
        hec_inst.open_session()
        arr_hec = hec_inst.select_time(hec_inst.start_time, hec_inst.stop_time)
        arr_dict_hec = hec_inst.obtain_data_dict(arr_hec)
        hec_inst.close_session()
        ic_inst.open_session()
        arr_ic = ic_inst.select_time(ic_inst.start_time, ic_inst.stop_time)
        arr_dict_ic = ic_inst.obtain_data_dict(arr_ic)
        ic_inst.close_session()
        ic_inst.close_engine()
        hec_inst.close_engine()


if __name__ == "__main__":
    app_log.info("Make Grid app starts..")
    db_name = "ab_data_upd.db"
    instance = CreateGrid(db_name)
    instance.some()
    # instance.close_engine()
    app_log.info("Make Grid app stops.")
