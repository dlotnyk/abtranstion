from datetime import datetime

grid_dict = ({"pressure_s": 22.5, "hec": {"start_time": datetime(2019, 2, 21, 21, 0),
                                        "end_time": datetime(2019, 2, 22, 4, 30)},
              "ic": {"start_time": datetime(2019, 2, 21, 21, 30),
                     "end_time": datetime(2019, 2, 22, 4, 30)}},
             {"pressure_s": 22.25, "hec": {"start_time": datetime(2019, 2, 22, 19, 40),
                                         "end_time": datetime(2019, 2, 23, 2, 0)},
              "ic": {"start_time": datetime(2019, 2, 22, 20, 12),
                     "end_time": datetime(2019, 2, 23, 2, 0)}},
             {"pressure_s": 22.0, "hec": {"start_time": datetime(2019, 2, 23, 21, 5),
                                        "end_time": datetime(2019, 2, 24, 2, 0)},
              "ic": {"start_time": datetime(2019, 2, 23, 21, 25),
                     "end_time": datetime(2019, 2, 24, 2, 0)}},
             {"pressure_s": 21.75, "hec": {"start_time": datetime(2019, 4, 19, 21, 45),
                                         "end_time": datetime(2019, 4, 20, 9, 0)},
              "ic": {"start_time": datetime(2019, 4, 19, 22, 30),
                     "end_time": datetime(2019, 4, 20, 11, 30)}},
             {"pressure_s": 21.6, "hec": {"start_time": datetime(2019, 4, 6, 19, 15),
                                        "end_time": datetime(2019, 4, 7, 2, 0)},
              "ic": {"start_time": datetime(2019, 4, 6, 20, 0),
                     "end_time": datetime(2019, 4, 7, 10, 0)}},
             {"pressure_s": 21.4, "hec": {"start_time": datetime(2019, 4, 4, 17, 45),
                                        "end_time": datetime(2019, 4, 4, 22, 0)},
              "ic": {"start_time": datetime(2019, 4, 4, 18, 30),
                     "end_time": datetime(2019, 4, 5, 2, 0)}}
             )


if __name__ == "__main__":
    print(grid_dict[0]["hec"]["start_time"])

