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
    flag_plot = True
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

    def dt_time(self, date):
        '''converts datetime into seconds'''
        t0 = date[0].timestamp()
        sec = [ii.timestamp()-t0 for ii in date]
        return(sec)

    def ttc(self, T, Tc):
        '''convert T to T/Tc'''
        T1 = [ii/Tc for ii in T]
        return T1

    @my_logger
    @time_this
    def QtoT(self, arr, num_ar):
        '''start making a grid'''
       # num_ar = 0
        d1 = self.dt_parse(arr[num_ar][1])
        d2 = self.dt_parse(arr[num_ar][2])
        d3 = self.dt_parse(arr[num_ar][3])
        d4 = self.dt_parse(arr[num_ar][4])
        tc = self.PtoTc(arr[num_ar][0])
      
        _, _, Q1, temp, _, date, _, dQ1 = MysABdata.select_interval(
            self, self.table_name, d1, d2)
        time = self.dt_time(date)
        numm = np.argmax(np.abs(dQ1))
        numm -= 50
        fit, rQ1, Qhec, Thec, sqT1 = self.revQtoT_hec(time[0:numm], Q1[0:numm],
                temp[0:numm], tc)
        _, Q2, _, _, _, date2, dQ2, _ = MysABdata.select_interval(
                self, self.table_name, d3, d4)
        num2 = np.argmax(np.abs(dQ2))
        num2 -= 50
        Tic, rQ2, sqT2 = self.revQtoT_ic(Q2[0:num2], Qhec, fit, tc)
        print(arr[num_ar][0], np.mean(Thec[-25:-20]), np.mean(Tic[-25:-20]))
        fig1 = plt.figure(1, clear=True)
        ax1 = fig1.add_subplot(211)
        ax1.set_ylabel('Q')
        ax1.set_xlabel('date')
        ax1.set_title("Q vs time for ")
        ax1.scatter(date, Q1, color='green', s=1, label='HEC')
        ax1.scatter(date2, Q2, color='blue', s=1, label='IC')
        ax1.scatter(date[numm], Q1[numm], color='red', s=5)
        ax1.scatter(date2[num2], Q2[num2], color='red', s=5)
        ax1.set_xlim(d1, d4)
        ax1.legend()
#        plt.gcf().autofmt_xdate()
        plt.grid()
        ax2 = fig1.add_subplot(212)
        ax2.scatter(date[0:len(Thec)], Thec, color='green', s=1, label='HEC')
        ax2.scatter(date2[0:num2], Tic, color='blue', s=1, label='IC')
        ax2.set_xlim(d1, d4)
        plt.grid()
        fig2 = plt.figure(3, clear=True)
        ax2 = fig2.add_subplot(111)
        ax2.set_ylabel(r'Q(T$_c$)$^{-1}$ - Q(T)$^{-1}$')
        ax2.set_xlabel(r'$\sqrt{1 - T/T_c}$')
        ax2.set_title(r"Q$^{-1}$ vs T$_{loc} for$ "+str(arr[num_ar][0]) + 
                " bar")
        ax2.plot(sqT1, rQ1, color='green', linewidth=5, label='HEC')
        ax2.plot(sqT2, rQ2, color='blue', linewidth=1, label='IC')
        ax2.legend()
#        ax2.set_yscale('log')
        plt.grid()
        return arr[num_ar][0], 1/Qhec, np.mean(Q2[0:10])

    @my_logger
    @time_this
    def revQtoT_hec(self, time, Q, T, Tc):
        '''1/Q to T for HEC
        1. T vs time linear
        2. Q to T based on T(time)
        fit_nz - fit params; qq=1/q; tt = sqrt...'''
        numf = 3
        Qc = 1/np.mean(Q[0:10])
        Q1 = sorted(self.QtorevQ(Q, Qc)) # 1/Q
        fit_t = np.polyfit(time, T, 1) # T(t)
        fit_fnt = np.poly1d(fit_t)
        T2 = fit_fnt(time)
        dt = np.mean(T2[0:10]) - Tc # shift to Tc
        T1 = [ii-dt for ii in T2]
        lt = self.TtosqT(T1, Tc)    #sqrt(1 - T/tc) 
        qq, tt = self.QTnoZer(Q1, lt)
        w = np.ones(len(qq))
        w[0:20] = 5
        w[-10:-1] = 5
        fit_nz = np.polyfit(qq, tt, numf, w=w)
        rev_f = np.poly1d(fit_nz)
        Thec = self.SqTtoT(tt, Tc)
        if self.flag_plot:
            fig1 = plt.figure(2, clear=True)
            ax1 = fig1.add_subplot(111)
            ax1.set_xlabel(r'Q(T$_c$)$^{-1}$ - Q(T)$^{-1}$')
            ax1.set_ylabel('(1 - T/T$_c$)$^{1/2}$')
            ax1.set_title("Q --> T for HEC")
            ax1.scatter(qq, tt, color='green', s=1, label='HEC')
    #        ax1.scatter(Q2, lt, color='blue', s=1, label='HEC')
            ax1.plot(qq, rev_f(qq), color='red')
    #        ax1.set_xscale('log')
            ax1.legend()
            plt.grid()
        return fit_nz, qq, Qc, Thec, rev_f(qq) 

    @my_logger
    @time_this
    def revQtoT_ic(self, Q, Qh, fit, Tc):
        '''1/Q to T for IC chamber'''
        dq = np.mean(Q[0:10]) - 1/Qh
        rev_f = np.poly1d(fit)
        Q1 = sorted([ii-dq for ii in Q])
        Qc = 1/np.mean(Q1[0:10])
        T, qq, tt = self.recombineT(fit, Q1, Tc)
        tt1 = self.TtosqT(T, Tc)
        if self.flag_plot:
            fig1 = plt.figure(6, clear=True)
            ax1 = fig1.add_subplot(111)
            ax1.set_xlabel(r'Q(T$_c$)$^{-1}$ - Q(T)$^{-1}$')
            ax1.set_ylabel('(1 - T/T$_c$)$^{1/2}$')
            ax1.set_title("Q --> T for IC")
            ax1.scatter(qq, tt1, color='green', s=1, label='HEC')
    #        ax1.plot(qq, rev_f(qq), color='red')
            ax1.legend()
            plt.grid()
        return T, qq, tt

    @time_this
    @my_logger
    def PtoTc(self, p):
        '''find Tc at pressure p'''
        a0 = 0.929383750000000
        a1 = 0.138671880000000
        a2 = -0.006930218500000
        a3 = 0.000256851690000
        a4 = -0.000005724864400
        a5 = 5.30E-08
        return a0 + a1*p + a2*p**2 + a3*p**3 + a4*p**4 + a5*p**5

    @time_this
    @my_logger
    def TtosqT(self, T, Tc):
        '''convert T into (1 - T/Tc)^1/2'''
        return [(1-ii/Tc) for ii in T]

    @time_this
    @my_logger
    def QtorevQ(self, Q, Qc):
        '''Q into 1/Qc - 1/Q'''
        return [Qc - 1/ii for ii in Q]

    @time_this
    @my_logger
    def SqTtoT(self, tt, Tc):
        '''sqrt(1-t/tc) to T'''
        return [Tc*(1-ii**2) for ii in tt]

    @time_this
    @my_logger
    def QTnoZer(self, Q, T):
        '''1/Q and (1-T) with 1/Q > 0 and make sqrt(T)'''
        T1 = []
        Q1 = []
        for ii, jj in zip(Q, T):
            if ii >= 0:
                Q1.append(ii)
                T1.append(np.sqrt(jj))
        return Q1, T1

    @time_this
    @my_logger
    def recombineT(self, fit, Q, Tc):
        '''recombine from fit sqrt(1-T/Tc) vs polyfit(1/Qc - 1/Q)'''
        rev_f = np.poly1d(fit)
        Qc1 = np.mean(Q[:10])
#        print(Tc, Qc, Qc1)
        y = [np.abs((1/Qc1 - 1/ii)) for ii in Q]
        T = [Tc*(1-rev_f(ii)**2) for ii in y]
        tt = [rev_f(ii) for ii in y]
        if self.flag_plot:
            fig1 = plt.figure(5, clear=True)
            ax1 = fig1.add_subplot(111)
            ax1.set_xlabel(r'Q(T$_c$)$^{-1}$ - Q(T)$^{-1}$')
            ax1.set_ylabel('(1 - T/T$_c$)$^{1/2}$')
            ax1.set_title(r'Reconstruction part Q$^{-1}$ -> $\sqrt{1-T}$')
            ax1.scatter(range(0, len(Q)), T, color='blue', s=1, label='IC')
            ax1.legend()
            plt.grid()
        return T, y, tt 


# -------------------main----------------
if __name__ == '__main__':
    data_dir = "d:\\dima\\proj\\ab_trans\\data\\"
    t1 = datetime.datetime(2019, 4, 7, 15, 0, 0)
    t2 = datetime.datetime(2019, 4, 10, 9, 0, 0, 0)
    B = QT_Grid(config, data_dir)
    B.table_name = 'data'
    B.flag_plot = False
    P = []
    Q1 = []
    Q2 = []
    B.qttime = B.dataset(q1=3, q2=4, temp=2, pressure=7, time=1)
    arr = B.import_grid()
    for ii in range(0, 5):
        p, Qc1, Qc2 = B.QtoT(arr, ii)
        P.append(p)
        Q1.append(Qc1)
        Q2.append(Qc2)
        print(p, Qc1, Qc2)
    fig1 = plt.figure(10, clear=True)
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel(r'P [bar]')
    ax1.set_ylabel(r'Q$_c$')
    ax1.set_title(r'Q$_c$ vs P dependence (main)')
    ax1.scatter(P, Q2, color='blue', s=1, label='IC')
    ax1.scatter(P, Q1, color='green', s=1, label='HEC')
    ax1.legend()
    plt.grid()
    plt.show()
    B.close_f()
