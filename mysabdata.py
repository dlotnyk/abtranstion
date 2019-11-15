import mysql.connector as msql
import matplotlib.pyplot as plt
from abres import ABData
from abres import time_this
from abres import my_logger
from mysql.connector import errorcode
import sys
import datetime
import numpy as np
sys.path.insert(0, 'd:\\dima\\proj\\ab_trans')
from configa import config
# another class for mysql connection


class MysABdata(ABData):
    def __init__(self, conf, data_dir):
        #       self.conf=conf
        #       self.connect_loc(conf)
        ABData.__init__(self, conf, data_dir)

#    @my_logger
    @time_this
    def connect_loc(self, conf):
        '''reconfigure acc to mysql'''
        assert type(conf) is dict, "Input parameter should be dict!!!"
        self.autoinc = 'AUTO_INCREMENT'
        self.db_name = conf['database']
        self.cnx = None
        try:
            self.cnx = msql.connect(**conf)
            self.cursor = self.cnx.cursor()
            print('myCONNECTED CLOUD!!!!')
        except msql.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)
            if self.cnx:
                self.cnx.close()

    @my_logger
    @time_this
    def import_fun(self, path_press):
        '''import needed data from dat files
        time/date [0] [1]; unitime [2]; Q [6]; Tmc [13]
        update an sql table'''
        path = self.work_dir+self.hec_file
        path1 = self.work_dir+self.ic_file
        path2 = self.work_dir+path_press[0]
        with open(path, 'r') as f, open(path1, 'r') as f1, \
                open(path2, 'r') as f2:
            next(f)  # skip first
            next(f1)
            next(f2)
            d2 = next(f2).split()
            press = float(d2[3])
            for line, line1 in zip(f, f1):
                d = line.split()
                d1 = line1.split()
                mm, dd, yy, hh, mins, ss = self._gettime(d[0], d[1])
                mm1, dd1, yy1, hh1, mins1, ss1 = self._gettime(d2[0], d2[1])
                a = datetime.datetime(
                    int('20'+yy), int(mm), int(dd), int(hh), int(mins),
                    int(ss), 0)
                a1 = datetime.datetime(
                    int('20'+yy1), int(mm1), int(dd1), int(hh1), int(mins1),
                    int(ss1), 0)
                arr = np.array((d[2], d[13], d[6], d1[6], d[7], d1[7], press,
                                0, d[10], d[15], d[16], d[17], d[18]),
                                dtype=float)
                if a1 < a:  # synchronize pressure and Q data in time
                    try:
                        d2 = next(f2).split()
                        mm1, dd1, yy1, hh1, mins1, ss1 = self._gettime(
                            d2[0], d2[1])
                        press = float(d2[3])
                    except StopIteration:
                        press = press
                self.__update_table(self.table_name, a, arr)
            self.cnx.commit()

    def __update_table(self, tb_name, date, arr1):
        '''set a new values to the table'''
#        assert type(date) is str and type(arr1) is np.ndarray, 
# "wrong types in update table"
        arr = ['NULL' if np.isnan(jj) else str(jj) for jj in arr1]
        query = ("INSERT INTO {0} "
                 "(`date`, `utime`, `Tmc`, `Q1`, `Q2`, `F1`, `F2`, "
                 "`Pressure`, `Cmc`, `DriveV`, `PulseA`, `SmallP`, `BigP`, "
                 "`WaitT`) "
                 "VALUES ('{1}', {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, "
                 "{10}, {11}, {12}, {13}, {14}) ".format(tb_name, date, 
                     arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], 
                     arr[7], arr[8], arr[9], arr[10], arr[11], arr[12]))
        self.cursor.execute(query)

    @my_logger
    @time_this
    def _forselect(self, tb_name, t1, t2):
        assert type(t1) is datetime.datetime and type(
            t2) is datetime.datetime, "t1 and t2 should be datetime"
        assert t2 > t1, "t2 should be > t1"
        query = ("SELECT * FROM {} WHERE `date` >= '{}' AND `date` <= '{}' "
        "ORDER BY `date` ASC".format(tb_name, str(t1), str(t2)))
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        res = self._removeNull(data)
        return res

    def _removeNull(self, res):
        '''remove nulls and convert it to pyhonic nan's'''
        kerneldt = np.dtype({'names': ['date', 'utime', 'Tmc', 'Q1', 'Q2', 
            'F1', 'F2', 'Pressure','Cmc', 'DriveV', 'PulseA', 'SmallP', 
            'BigP', 'WaitT'],'formats': ['U20', np.float32, np.float32,
                np.float32, np.float32,np.float32, np.float32, np.float32,
                np.float32, np.float32, np.float32,np.float32, np.float32, 
                np.float32, np.float32]})
        dat = np.zeros(np.shape(res)[0], dtype=kerneldt)
        assert len(dat) > 0, "no data were taken from SELECT"
        for ind, x in enumerate(res):
            for y in x:
                if y is None:
                    y = np.nan
            dat[ind] = x
        return dat

    @time_this
    def calc_pressure(self, dQ1, dQ2, temp, pres):
        '''calculate Pab and Tab based on derivative
        rate in [mK/hr]'''
        ind1Q1 = np.argmax(np.abs(dQ1))
        ind1Q2 = np.argmax(np.abs(dQ2))
        p1 = pres[ind1Q1]
        t1 = temp[ind1Q1]
        p2 = pres[ind1Q2]
        t2 = temp[ind1Q2]
        print("AB Pressure in HEC is ", p1)
        print("AB Pressure in IC is ", p2)
        print("Tab in HEC ", t1)
        print("Tab in IC ", t2)
        return 0, ind1Q1, ind1Q2, ind1Q1, ind1Q2


# -------------------main----------------
if __name__ == '__main__':
    data_dir = "d:\\dima\\proj\\ab_trans\\data\\"
    t1 = datetime.datetime(2019, 6, 24, 2, 0, 0, 0)
    t2 = datetime.datetime(2019, 6, 24, 12, 0, 0, 0)
    B = MysABdata(config, data_dir)
    B.table_name = 'data'
    B.qttime = B.dataset(q1=3, q2=4, temp=2, pressure=7, time=1)
#    B.dir_scan()
    time, Q2, Q1, temp, pres, date, dQ2, dQ1 = B.select_interval(
        B.table_name, t1, t2)
    p1 = np.nanmean(pres)
    rate, ind1Q1, ind2Q1, ind1Q2, ind2Q2 = B.calc_params(dQ1, dQ2, temp,
                                                         time, pres)

#    rate, ind1Q1, ind2Q1, ind1Q2, ind2Q2 = B.calc_pressure(dQ1, dQ2, temp,
#                                                           pres)
# --------------------------------------------------------------------s

    # plotting
    fig1 = plt.figure(1, clear=True)
    ax1 = fig1.add_subplot(221)
    ax1.set_ylabel('Q')
    ax1.set_xlabel('date')
    ax1.set_title("Q vs time for "+str(p1)+" bar")
    ax1.scatter(date, Q1, color='green', s=1, label='HEC')
    ax1.scatter(date, Q2, color='blue', s=1, label='IC')
    ax1.scatter(date[ind1Q1], Q1[ind1Q1], color='red', s=10)
    ax1.scatter(date[ind2Q1], Q1[ind2Q1], color='red', s=10)
    ax1.scatter(date[ind1Q2], Q2[ind1Q2], color='red', s=10)
    ax1.scatter(date[ind2Q2], Q2[ind2Q2], color='red', s=10)
    ax1.set_xlim(t1, t2)
    ax1.legend()
    plt.gcf().autofmt_xdate()
    plt.grid()
    ax2 = fig1.add_subplot(222)
    ax2.set_ylabel(r'$T_{MC}$')
    ax2.set_xlabel('Date')
    ax2.set_title(r'$T_{MC}$ vs time for both forks')
    ax2.scatter(date, temp, color='green', s=0.5)
    ax2.set_xlim(t1, t2)
    plt.gcf().autofmt_xdate()
    plt.grid()
    ax3 = fig1.add_subplot(223)
    ax3.set_ylabel('dQ/dt')
    ax3.set_xlabel('date')
    ax3.set_title("derivative vs time")
    ax3.scatter(date, dQ1, color='green', s=0.5)
    ax3.scatter(date, dQ2, color='blue', s=0.5)
    ax3.scatter(date[ind1Q1], dQ1[ind1Q1], color='red', s=10)
    ax3.scatter(date[ind2Q1], dQ1[ind2Q1], color='red', s=10)
    ax3.scatter(date[ind1Q2], dQ2[ind1Q2], color='red', s=10)
    ax3.scatter(date[ind2Q2], dQ2[ind2Q2], color='red', s=10)
    ax3.set_xlim(t1, t2)
    plt.gcf().autofmt_xdate()
    plt.grid()
    ax4 = fig1.add_subplot(224)
    ax4.set_ylabel('Pressure')
    ax4.set_xlabel('date')
    ax4.set_title('Pressure vs time')
    ax4.scatter(date, pres, color='green', s=0.5)
    ax4.set_xlim(t1, t2)
    plt.gcf().autofmt_xdate()
    plt.grid()

    plt.show()
    #------------- delete
    del time, Q1, Q2, temp, pres, date, dQ1, dQ2
    B.close_f()
