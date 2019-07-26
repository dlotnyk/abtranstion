import mysql.connector as msql
import scipy.signal as sci
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm
from mysabdata import MysABdata
from calibration import QT_calibration
from abres import time_this
from abres import my_logger
from mysql.connector import errorcode
import sys
import datetime
import numpy as np
sys.path.insert(0, 'd:\\dima\\proj\\ab_trans')
from configa import config


class QT_Grid(QT_calibration):
    '''create a T vs Q grid for different pressures'''
    flag_plot = True
    tab_qt = 'tab_qt'
    tab_params = 'tab_params'
    grid_pressures = (22.5, 22.25, 22, 21.75, 21.6, 21.4)

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
        pr = arr[num_ar][0]
        Qc1 = 1/Qhec
#        Qc2 = np.mean(Q2[0:10])
        Qc2 = np.nanmin(Q2)
        return pr, (Qc1, Qc2), zip(Q1[0:len(Thec)], Thec, rQ1, sqT1),\
                zip(Q2[0:num2], Tic, rQ2, sqT2), fit

    @my_logger
    @time_this
    def revQtoT_hec(self, time, Q, T, Tc):
        '''1/Q to T for HEC
        1. T vs time linear
        2. Q to T based on T(time)
        fit_nz - fit params; qq=1/q; tt = sqrt...'''
        numf = 3
#        Qc = 1/np.mean(Q[0:10])
        Qc = 1/np.nanmin(Q)
        Q1 = sorted(self.QtorevQ(Q, Qc)) # 1/Q
        fit_t = np.polyfit(time, T, 1) # T(t)
        fit_fnt = np.poly1d(fit_t)
        T2 = fit_fnt(time)
        dt = np.mean(T2[0:10]) - Tc # shift to Tc
        T1 = [ii-dt for ii in T2]
        lt = self.TtosqT(T1, Tc)    #sqrt(1 - T/tc) 
        qq, tt = self.QTnoZer(Q1, lt)
        w = np.ones(len(qq))
        midp = int(len(qq)*0.75)
        w[10:20] = 5
        #w[-10:-1] = 5
        w[midp:midp+10] = 5
        fit_nz = np.polyfit(qq, tt, numf, w=w)
        rev_f = np.poly1d(fit_nz)
        Thec = self.SqTtoT(tt, Tc)
        #derivative
        wind = 11
        poly = 2
        dQ1 = sci.savgol_filter(qq, wind, poly, deriv=2)
        #plot
        if self.flag_plot:
            fig1 = plt.figure(2, clear=True)
            ax1 = fig1.add_subplot(211)
            ax1.set_xlabel(r'Q(T$_c$)$^{-1}$ - Q(T)$^{-1}$')
            ax1.set_ylabel('(1 - T/T$_c$)$^{1/2}$')
            ax1.set_title("Q --> T for HEC")
            ax1.scatter(qq, tt, color='green', s=1, label='HEC')
    #        ax1.scatter(Q2, lt, color='blue', s=1, label='HEC')
            ax1.plot(qq, rev_f(qq), color='red')
    #        ax1.set_xscale('log')
            ax1.legend()
            plt.grid()
            ax2 = fig1.add_subplot(212)
            ax2.scatter(tt, dQ1, s=5)
            ax2.set_ylim(np.amin(dQ1), np.amax(dQ1))
            plt.grid()
        return fit_nz, qq, Qc, Thec, rev_f(qq) 

    @my_logger
    @time_this
    def revQtoT_ic(self, Q, Qh, fit, Tc):
        '''1/Q to T for IC chamber'''
#        dq = np.mean(Q[0:10]) - 1/Qh
        dq = np.nanmin(Q) - 1/Qh
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
            if (jj >= 0) and (ii >= 0):
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

    @time_this
    @my_logger
    def create_tableQT(self, tb_name1):
        '''create table with q, t, reverse Q and sqrt(1-t/tc)'''
        table_HEC = ("CREATE TABLE `{}` ("
             "`index` int AUTO_INCREMENT COMMENT 'index', "
             "`revQ` DOUBLE NOT NULL COMMENT '1/Qc - 1/Q(T)', "
             "`sqT` DOUBLE NOT NULL COMMENT 'sqrt(1-t/tc)', "
             "`Q` DOUBLE NULL COMMENT 'Quality factor', "
             "`T` DOUBLE NULL COMMENT 'Local temperature [mK]', "
             "`P` DOUBLE NOT NULL COMMENT 'Pressure [bar]', "
             "`FN` int NOT NULL COMMENT 'Fork Number', "
             "PRIMARY KEY (`index`), "
             "KEY `ii` (`sqT`)"
             ") ENGINE=InnoDB".format(tb_name1))
        alter_hec = ("ALTER TABLE {} AUTO_INCREMENT=1".format(tb_name1))
        try:
            print("Creating table for {}: ".format(tb_name1), end='')
            self.cursor.execute(table_HEC)
            self.cnx.commit()
            self.cursor.execute(alter_hec)
            self.cnx.commit()
        except msql.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("HEC table already exists.")
            else:
                print(err.msg)
        else:
            print("OK")

    @time_this
    @my_logger
    def create_tableparams(self, tb_name1):
        '''create table of 3rd order fit parameters '''
        table_HEC = ("CREATE TABLE `{}` ("
             "`index` int AUTO_INCREMENT COMMENT 'index', "
             "`P` DOUBLE NOT NULL COMMENT 'Pressure [bar]', "
             "`a0` DOUBLE NOT NULL COMMENT 'a0', "
             "`a1` DOUBLE NOT NULL COMMENT 'a1', "
             "`a2` DOUBLE NOT NULL COMMENT 'a2', "
             "`a3` DOUBLE NOT NULL COMMENT 'a3', "
             "PRIMARY KEY (`P`), "
             "KEY `ii` (`index`)"
             ") ENGINE=InnoDB".format(tb_name1))
        alter_hec = ("ALTER TABLE {} AUTO_INCREMENT=1".format(tb_name1))
        try:
            print("Creating table for {}: ".format(tb_name1), end='')
            self.cursor.execute(table_HEC)
            self.cnx.commit()
            self.cursor.execute(alter_hec)
            self.cnx.commit()
        except msql.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                print("HEC table already exists.")
            else:
                print(err.msg)
        else:
            print("OK")

    def __sq_qt(self, q, t, rq, sqt, p, fn):
        '''insert valueus to sql tables calculated in QtoT  
        pr - pressure; zqt1 - zip(q, t, rq, sqT)
        fit - fit params'''
        query1 = ("INSERT INTO {0} "
                 "(`revQ`, `sqT`, `Q`, `T`, `P`, `FN`) "
                 "VALUES ('{1}', {2}, {3}, {4}, {5}, {6}) "
                 .format(self.tab_qt, rq, sqt, q, t, p, fn))
        self.cursor.execute(query1)

    def __sq_params(self, p, fit):
        '''insert valueus to sql tables calculated in QtoT  
        p - pressure; 
        fit - fit params'''
        query1 = ("INSERT INTO {0} "
                 "(`P`, `a0`, `a1`, `a2`, `a3`) "
                 "VALUES ('{1}', {2}, {3}, {4}, {5}) "
                 .format(self.tab_params, p, fit[0], fit[1], fit[2], fit[3]))
        self.cursor.execute(query1)

    @my_logger
    @time_this
    def insert_qt_params(self, pr, zqt1, zqt2, fit):
        '''insert valueus to sql tables calculated in QtoT  
        pr - pressure; zqt1 - zip(q, t, rq, sqT)
        fit - fit params'''
        for ii in zqt1:
            self.__sq_qt(ii[0], ii[1], ii[2], ii[3], pr, 1)
        self.cnx.commit()
        print('Pressure {}; fork1'.format(pr))
        self.__sq_params(pr, fit)
        self.cnx.commit()
        print('Pressure {}; fit'.format(pr))
        for ii in zqt2:
            self.__sq_qt(ii[0], ii[1], ii[2], ii[3], pr, 2)
        self.cnx.commit()
        print('Pressure {}; fork2'.format(pr))
        
    @my_logger
    @time_this
    def setdata_qt_params(self, arr):
        '''drop if any, create and insert data for qt and fit parameters'''
        self.drop_table(self.tab_qt)
        self.drop_table(self.tab_params)
        self.create_tableQT(self.tab_qt)
        self.create_tableparams(self.tab_params)
        arr_s = np.shape(arr)[0]
        for ii in range(0, arr_s):
            p, Qct, qt1, qt2, fit = self.QtoT(arr, ii)
            self.insert_qt_params(p, qt1, qt2, fit)

    @my_logger
    @time_this
    def select_qt(self, p, fn):
        '''select 1/q and sqrt(T) for pressure and desired fork
        fn = 1 - HEC; fn = 2 - IC'''
        pl = p-0.01
        ph = p+0.01 
        query = ("SELECT `revQ`, `sqT` FROM {} WHERE `P` >= '{}' AND `P` <= "
                "'{}' AND `FN` = '{}' "
        "ORDER BY `sqT` ASC".format(self.tab_qt, str(pl), str(ph), str(fn)))
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        rQ = tuple([ii[0] for ii in data])
        sqT = tuple([ii[1] for ii in data])
        return rQ, sqT
     
    @my_logger
    @time_this
    def select_params(self, p):
        '''select parameters for a pressure p'''
        pl = p-0.01
        ph = p+0.01
        query = ("SELECT * FROM {} WHERE `P` >= '{}' AND `P` <= '{}'"
                .format(self.tab_params, str(pl), str(ph)))
#        print(query)
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        fit = (data[0][2], data[0][3], data[0][4], data[0][5])
        return np.asarray(fit)

    @my_logger
    @time_this
    def map3D(self):
        '''plot a 3D map'''
        sT = np.linspace(0, 0.01, 100)
        p = np.asarray(self.grid_pressures)
        sizeP = np.shape(p)[0]
        sizeT = np.shape(sT)[0]
        Z = np.ones((sizeT, sizeP))
        for id1, p1 in enumerate(p):
            fit1 = self.select_params(p1)
            r_fit = np.poly1d(fit1)
            for id2, x in enumerate(sT):
                Z[id2][id1] = r_fit(x)
        sT, p = np.meshgrid(p, sT)
        fig = plt.figure(13, clear=True)
        ax1 = fig.gca(projection='3d')
#        surf = ax1.plot_surface(p, sT, Z, cmap=cm.coolwarm)
        wa = ax1.plot_wireframe(p, sT, Z)
        ax1.set_xlabel(r'$\sqrt{1-T/T_c}$')
        ax1.set_ylabel('P [bar]')
        ax1.set_zlabel(r'Q$^{-1}$(T$_c$) - Q$^{-1}$(T)')


# -------------------main----------------
if __name__ == '__main__':
    data_dir = "d:\\dima\\proj\\ab_trans\\data\\"
    t1 = datetime.datetime(2019, 4, 7, 15, 0, 0)
    t2 = datetime.datetime(2019, 4, 10, 9, 0, 0, 0)
    B = QT_Grid(config, data_dir)
    B.table_name = 'data'
    B.flag_plot = False 
    P = []
    Q1c = []
    Q2c = []
    B.qttime = B.dataset(q1=3, q2=4, temp=2, pressure=7, time=1)
    rQ, sqT = B.select_qt(B.grid_pressures[0], 1)
    fit = B.select_params(B.grid_pressures[0])
#    print(rQ[0], sqT[0], fit)
    arr = B.import_grid()
    p, Qct, qt1, qt2, fit1 = B.QtoT(arr, 0)
    B.imp_QT(arr, 0)
    print('main {}'.format(p))
#    print(np.poly1d(fit), np.poly1d(fit1))
#    B.setdata_qt_params(arr)
    #B.map3D()

#    fig2 = plt.figure(11, clear=True)
#    ax2 = fig2.add_subplot(111)
#    ax2.set_xlabel('T [mk]')
#    ax2.set_ylabel('Q')
#    ax2.set_prop_cycle(color=['red', 'green', 'blue', 'black', 'cyan',
#        'magenta', 'orange'])
#    for ii in B.grid_pressures:
#        rq1, sqT1 = B.select_qt(ii, 1)
#        rq2, sqT2 = B.select_qt(ii, 2)
#        ax2.scatter(sqT1, rq1, s=10)
#        ax2.scatter(sqT2, rq2, s=3)
#    plt.grid() 
    plt.show()
    B.close_f()
