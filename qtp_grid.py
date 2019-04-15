import mysql.connector as msql
import matplotlib.pyplot as plt
from mysabdata import MysABdata
from abres import time_this
from abres import my_logger
from mysql.connector import errorcode
import sys
import datetime
import numpy as np
sys.path.insert(0, 'd:\\dima\\proj\\ab_trans')
from configa import config


class QT_Grid(MysABdata):
    '''create a T vs Q grid for different pressures'''
    def __init__(self, conf, data_dir):
        MysABdata.__init__(self, conf, data_dir)

    @my_logger
    @time_this
    def import_grid(self):
        '''import dates for grid from .dat file'''
        path = 'grid.dat'
        data = np.genfromtxt(path, unpack=True, skip_header=1)
        kerneldt = np.dtype({'names': ['pressure', 'HECstart', 'HECstop', 
            'ICstart', 'ICstop'],
            'formats': [np.float32, 'U20', 'U20', 'U20', 'U20', 'U20']})
        arr = np.zeros((int(np.shape(data)[1]/2)), dtype=kerneldt)
        for ind, val in enumerate(arr):
            a1 = 2*ind
            a2 = 2*ind + 1
            val[0] = data[0][a1]
            val[1] = datetime.datetime(int(data[1][a1]), int(data[2][a1]),
                    int(data[3][a1]),
                    int(data[4][a1]), int(data[5][a1]), 0)
            val[2] = datetime.datetime(int(data[6][a1]), int(data[7][a1]),
                    int(data[8][a1]),
                    int(data[9][a1]), int(data[10][a1]), 0)
            val[3] = datetime.datetime(int(data[1][a2]), int(data[2][a2]),
                    int(data[3][a2]),
                    int(data[4][a2]), int(data[5][a2]), 0)
            val[4] = datetime.datetime(int(data[6][a2]), int(data[7][a2]),
                    int(data[8][a2]),
                    int(data[9][a2]), int(data[10][a2]), 0)
        return arr        

    def dt_parse(self, p_arr):
        '''convert string to datetime'''
        return datetime.datetime.strptime(p_arr, '%Y-%m-%d %H:%M:%S')

    @my_logger
    @time_this
    def QtoT(self, arr):
        '''start making a grid'''
        d1 = self.dt_parse(arr[0][1])
        d2 = self.dt_parse(arr[0][2])
        d3 = self.dt_parse(arr[0][3])
        d4 = self.dt_parse(arr[0][4])
        tc = self.PtoTc(arr[0][0])
        time, _, Q1, temp, _, date, _, dQ1 = MysABdata.select_interval(
            self, self.table_name, d1, d2)
        numm = np.argmax(np.abs(dQ1))
        time2, Q2, _, _, _, date, dQ2, _ = MysABdata.select_interval(
                self, self.table_name, d3, d4)
        self.revQtoT_hec(time[0:numm], Q1[0:numm], temp[0:numm], tc)
        fig1 = plt.figure(1, clear=True)
        ax1 = fig1.add_subplot(211)
        ax1.set_ylabel('Q')
        ax1.set_xlabel('date')
        ax1.set_title("Q vs time for ")
        ax1.scatter(time, Q1, color='green', s=1, label='HEC')
        ax1.scatter(time2, Q2, color='blue', s=1, label='IC')
        ax1.scatter(time[numm], Q1[numm], color='red', s=5)
#        ax1.set_xlim(d1, d2)
        ax1.legend()
#        plt.gcf().autofmt_xdate()
        plt.grid()
        ax2 = fig1.add_subplot(212)
        ax2.scatter(temp, Q1, color='green', s=1, label='HEC')
        plt.grid()

    @my_logger
    @time_this
    def revQtoT_hec(self, time, Q, T, Tc):
        '''1/Q to T for HEC
        1. T vs time linear
        2. Q to T based on T(time)'''
        numf = 5
        Qc = 1/Q[0]
        Q1 = [np.abs(Qc - 1/ii) for ii in Q]
        fit_t = np.polyfit(time, T, 1)
        fit_fnt = np.poly1d(fit_t)
        T2 = fit_fnt(time)
        dt = T2[0] - Tc
        T1 = [ii-dt for ii in T2]
        fit = np.polyfit(Q1, T1, numf)
        fit_fn = np.poly1d(fit)
        fig1 = plt.figure(2, clear=True)
        ax1 = fig1.add_subplot(111)
        ax1.set_xlabel(r'Q$^{-1}$')
        ax1.set_ylabel('T [mK]')
        ax1.set_title("Q vs time for ")
        ax1.scatter(Q1, T1, color='green', s=1, label='HEC')
        ax1.plot(Q1, fit_fn(Q1), color='red')
        ax1.legend()
        plt.grid()
        return fit, Q1

    @my_logger
    @my_logger
    @time_this
    def revQtoT_ic(self, Q, Qh, fit):
        '''1/Q to T for IC chamber'''
        fit_fn = np.poly1d(fit)
        dq = Q[0] - Qh
        Qc = 1/(Q[0] - dq)
        Q1 = [np.abs(Qc - 1/(ii-dq)) for ii in Q]
        T = fit_fn(Q1)
        return T

    @time_this
    def PtoTc(self, p):
        '''find Tc at pressure p'''
        a0 = 0.929383750000000
        a1 = 0.138671880000000
        a2 = -0.006930218500000
        a3 = 0.000256851690000
        a4 = -0.000005724864400
        a5 = 5.30E-08
        return a0 + a1*p + a2*p**2 + a3*p**3 + a4*p**4 + a5*p**5


# -------------------main----------------
if __name__ == '__main__':
    data_dir = "d:\\dima\\proj\\ab_trans\\data\\"
    t1 = datetime.datetime(2019, 4, 7, 15, 0, 0)
    t2 = datetime.datetime(2019, 4, 10, 9, 0, 0, 0)
    B = QT_Grid(config, data_dir)
    B.table_name = 'data'
    B.qttime = B.dataset(q1=3, q2=4, temp=2, pressure=7, time=1)
    arr = B.import_grid()
    B.QtoT(arr)
    plt.show()
    B.close_f()
