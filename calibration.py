import numpy as np
import scipy.signal as scisig
import matplotlib.pyplot as plt
import datetime
from mysabdata import MysABdata
from abres import time_this
from abres import my_logger


class QT_calibration(MysABdata):
    '''make transformation Q into T'''

    def Qto1Q(self, Q, Qc):
        '''convert Q to 1/Qc - 1/Q'''
        revQ = [1/Qc-1/ii for ii in Q]
        return revQ

    def ttc(self, T, Tc):
        '''convert T to T/Tc'''
        T1 = [ii/Tc for ii in T]
        return T1

    def PtoTc(self, p):
        '''find Tc at pressure p'''
        a0 = 0.929383750000000
        a1 = 0.138671880000000
        a2 = -0.006930218500000
        a3 = 0.000256851690000
        a4 = -0.000005724864400
        a5 = 5.30E-08
        return a0 + a1*p + a2*p**2 + a3*p**3 + a4*p**4 + a5*p**5

    def dt_parse(self, p_arr):
        '''convert string to datetime'''
        return datetime.datetime.strptime(p_arr, '%Y-%m-%d %H:%M:%S')

    def dt_time(self, date):
        '''converts datetime into seconds'''
        t0 = date[0].timestamp()
        sec = [ii.timestamp()-t0 for ii in date]
        return(sec)

    def imp_QT(self, arr, num_ar):
        '''import needed Q, T, time arrays'''
        d1 = self.dt_parse(arr[num_ar][1])
        d2 = self.dt_parse(arr[num_ar][2])
        d3 = self.dt_parse(arr[num_ar][3])
        d4 = self.dt_parse(arr[num_ar][4])
        tc = self.PtoTc(arr[num_ar][0])
        _, _, Q1o, temp, _, date, _, dQ1 = MysABdata.select_interval(
            self, self.table_name, d1, d2)
        time = self.dt_time(date)
        _, Q2o, _, temp2, _, date2, dQ2, _ = MysABdata.select_interval(
                self, self.table_name, d3, d4)
        numm = np.argmax(np.abs(dQ1))
        num2 = np.argmax(np.abs(dQ2))
        numm -= 20
        num2 -= 20
        Q1 = sorted(Q1o)
        Q2 = sorted(Q2o)
        Qc1 = np.amin(Q1)
        Qc2 = np.amin(Q2)
        dQc = Qc2-Qc1
        Q2 = [ii-dQc for ii in Q2] # eqaulize Q's for 2 forks
        Qrev1 = self.Qto1Q(Q1, Qc1)
        Qrev2 = self.Qto1Q(Q2, Qc1)
        beg = 0
        fitQ1, Tf1 = self.poly_QinT(Qrev1[beg:numm], temp[beg:numm], 5, tc)
        _, Tf2 = self.poly_QinT(Qrev2[0:num2], temp2[0:num2], 5, tc)
        rFitQ = np.poly1d(fitQ1)
        Tf2_real = rFitQ(Qrev2[0:num2])
        print('HEC Tc={}; Tab={}'.format(Tf1[1], Tf1[-2]))
        print('IC Tc={}; Tab={}'.format(Tf2[1], Tf2[-2]))
        print('IC_real Tc={}; Tab={}'.format(Tf2_real[1], Tf2_real[-2]))
        #filQ1 = scisig.medfilt(temp[0:numm], 51)
        fig1 = plt.figure(51, clear=True)
        ax1 = fig1.add_subplot(211)
        ax1.set_xlabel('1/Q')
        ax1.set_ylabel('T')
        ax1.set_title("QT_calib Q vs T")
        ax1.scatter(Qrev1[0:numm], temp[0:numm], color='green', s=1, 
                label='HEC')
        ax1.scatter(Qrev2[0:num2], temp2[0:num2], color='blue', s=1, 
                label='IC')
        ax1.plot(Qrev2[0:num2], Tf2_real, color='orange', label='fitIC2')
        ax1.plot(Qrev1[beg:numm], Tf1, color='red', label='fitHEC')
        ax1.plot(Qrev2[0:num2], Tf2, color='magenta', label='fitIC')
        ax1.legend()
        plt.grid()
        ax2 = fig1.add_subplot(212)
        ax2.set_xlabel('T')
        ax2.set_ylabel('Q')
        #ax1.set_title("QT_calib Q vs T")
        ax2.scatter(Q1[0:numm], temp[0:numm], color='green', s=1, 
                label='HEC')
        ax2.scatter(Q2[0:num2], temp2[0:num2], color='blue', s=1, 
                label='IC')
        #ax1.scatter(date[numm], Qrev1[numm], color='red', s=5)
        #ax1.scatter(date2[num2], Qrev2[num2], color='red', s=5)
        #ax1.set_xlim(d1, d4)
        ax2.legend()
        plt.grid()

    def poly_QinT(self, revQ1, T1, poly, tc):
        '''polyfit of 1/Q vs T by poly order
        and shift temperature to tc'''
        #mid = int(len(revQ)/2)
        revQ = revQ1
        #T = T1[0:-1:10]
        #w = np.ones(len(revQ))
        w = self.mov_wei(len(revQ))
        T = scisig.medfilt(T1, 51)
        #w[mid-20:mid+20] = 5
        #w[-30:-10] = 5
        #w[0:30] = 5
        fitQ = np.polyfit(revQ, T, poly, w=w)
        fit1d = np.poly1d(fitQ)
        Tf = fit1d(revQ1)
        dT = tc-np.mean(Tf[0:5])
        fitQ[-1] += dT
        fit2 = tuple(fitQ)
        fit2d = np.poly1d(fit2)
        Tf2 = fit2d(revQ1)
        return fitQ, Tf2

    def mov_wei(self, num):
        '''returns array changed weights from 11 to 1 for num elems'''
        b = 11
        k = -10/num
        arr = np.ones(num)
        for inx, val in enumerate(arr):
            arr[inx] = int(k*inx+b)
        return arr    
