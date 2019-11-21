# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:17:13 2019

@author: dlotnyk
"""
from __future__ import print_function
from functools import wraps
import numpy as np
import sqlite3 as sql
import datetime
import inspect
import os
import matplotlib.pyplot as plt
import scipy.signal as sci
import collections
# --------------------------------------------------------------------


def time_this(original_function):
    '''Measures the processing time. Decorator'''
    @wraps(original_function)
    def new_function(*args, **kwargs):
        import time
        before = time.time()
        x = original_function(*args, **kwargs)
        after = time.time()
        print("Elapsed Time of fun {0} = {1}".format(
            original_function.__name__, after-before))
        return x
    return new_function


def my_logger(orig_func):
    '''Decorate function to write into log on the level ERROR'''
    import logging
    logging.getLogger('').handlers = []
    logging.basicConfig(filename='work.log'.format(
        orig_func.__name__), level=logging.INFO)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        frm = str(inspect.stack()[1][4][0])
        dt = datetime.datetime.now()
        dt_str = str(dt)
        vrema = dt_str.split('.')[0]
        logging.info(
            ' {} Ran with args:{}, {}, and kwargs: {}'.format(vrema, 
                frm[0:-1], args, kwargs))
        return orig_func(*args, **kwargs)
    return wrapper
# -------------------------------------------------------------
# -------------------------------------------------------------


class ABData():
    '''read from fies, create a sqlite database from date, universal time, Q, 
    Tmc and pressure'''
    # class attributes
    table_name = 'my_t'
    hec_file = "HEC_combine.dat"
    ic_file = "IC_combine.dat"
    work_dir = ""
    dataset = collections.namedtuple('dataset', 'q1 q2 temp pressure time')
    qttime = dataset(q1=2, q2=3, temp=4, pressure=5, time=1)
    # class methods

    def __init__(self, conf, data_dir):
        assert type(conf) is str or type(data_dir) is str, \
                "both input are strings"
        self.conf = conf
        self.data_dir = data_dir
        self.connect_loc(conf)
# -------------------------------------------------------------

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
# -------------------------------------------------------------
    @my_logger
    @time_this
    def connect_loc(self, conf):
        '''open connection to sql lite database'''
        assert type(conf) is str, "input parameter must be str"
        self.autoinc = ''
        self.cnx = sql.connect(conf)
        self.cursor = self.cnx.cursor()
        print('Connected!')
# -------------------------------------------------------------
    @my_logger
    @time_this
    def close_f(self):
        '''close connection'''
        self.cursor.close()
        self.cnx.close()
        print('Disconnected!')
# -------------------------------------------------------------
    @my_logger
    @time_this
    def import_fun(self, path_press):
        '''import needed data from dat files
        time/date [0] [1]; unitime [2]; Q [6]; Tmc [13]
        update an sql table'''
        assert type(path_press) is list, "must be a list"
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
                a = datetime.datetime(\
                    int('20'+yy), int(mm), int(dd), int(hh), int(mins), \
                    int(ss), 0)
                a1 = datetime.datetime(\
                    int('20'+yy1), int(mm1), int(dd1), int(hh1), int(mins1),  
                    int(ss1), 0)
                if a1 < a:
                    try:
                        d2 = next(f2).split()
                        mm1, dd1, yy1, hh1, mins1, ss1 = self._gettime(\
                            d2[0], d2[1])
                        press = float(d2[3])
                    except StopIteration:
                        press = press
                self.__update_table(self.table_name, (a, int(d[2]), \
                        float(d[6]), float(d1[6]), float(d[13]), press))
            self.cnx.commit()
# -------------------------------------------------------------
    @my_logger
    @time_this
    def find_files(self):
        '''find dat files in directory'''
        for root, dirs, files in os.walk(self.work_dir):
            ll = {os.path.join(root, file)
                  for file in files if file.endswith(".dat")}
        assert len(ll) > 0, "no files in the directory"
        return ll
# -------------------------------------------------------------

    def _gettime(self, date, time):
        '''get time in datetime from strings date and time'''
        mm, dd, yy = date.split('/')
        hh, mins, ss = time.split(':')
        return mm, dd, yy, hh, mins, ss
# -------------------------------------------------------------
    @my_logger
    @time_this
    def create_table(self, db_name):
        '''Create table in local database'''
        query = ("CREATE TABLE IF NOT EXISTS {} ("
                 #               "id integer PRIMARY KEY, "
                 "date text NOT NULL, "
                 "uni_time integer NOT NULL, "
                 "Q1 real, "
                 "Q2 real, "
                 "Tmc real, "
                 "pressure real);".format(db_name))
        self.cursor.execute(query)
# -------------------------------------------------------------
    @my_logger
    @time_this
    def drop_table(self, tb_name):
        '''drop db_name table'''
        query = ("DROP TABLE IF EXISTS {} ".format(tb_name))
        print('Drop table: OK')
        self.cursor.execute(query)
# -------------------------------------------------------------

    def __update_table(self, tb_name, values):
        '''set a new values to the table'''
        query = "INSERT INTO %s (date, uni_time, Q1, Q2, Tmc, pressure) \
        VALUES (?, ?, ?, ?, ?, ?) " % tb_name
        self.cursor.execute(query, values)
# -------------------------------------------------------------
    @my_logger
    @time_this
    def select_vals(self, tb_name):
        '''select all values'''
        query = "SELECT * FROM %s WHERE 1  ORDER BY date ASC" % tb_name
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        res = self._removeNull(data)
#        print(data)
        return res
# -------------------------------------------------------------

    def _removeNull(self, res):
        '''remove nulls and convert it to pyhonic nan's'''
        assert type(res) is list, "must be a list"
        assert np.shape(res)[0] > 0, "no data provided"
        kerneldt = np.dtype({'names': ['time', 'uni_t', 'Q1', 'Q2', 'Tmc', \
            'pressure'], 'formats': [
                            'U20', np.float32, np.float32, np.float32, 
                            np.float32, np.float32]})
        dat = np.zeros(np.shape(res)[0], dtype=kerneldt)
        assert len(dat) > 0, "no data were taken from SELECT"
        for ind, x in enumerate(res):
            for y in x:
                if y is None:
                    y = np.nan
            dat[ind] = x
        assert type(dat) is np.ndarray, "wrong output"    
        return dat
    # -------------------------------------------------------------
    @my_logger
    @time_this
    def find_hec(self, files):
        '''find all HEC and IC files. combine them into single HEC and IC. 
        find a path with pressure'''
        hec_f = self.work_dir+self.hec_file
        ic_f = self.work_dir+self.ic_file
        path_press = sorted([k.split('\\')[-1]
                             for k in files if k.split('\\')[-1][0:3] == 
                                     'pre'])
        hec = sorted([jj.split('\\')[-1] for jj in files 
                if jj.split('\\')[-1][0:3] == 'HEC'
                      and jj.split('\\')[-1] != self.hec_file])
        ic = sorted([ii.split('\\')[-1] for ii in files 
                if ii.split('\\')[-1][0:2] == 'IC'
                     and ii.split('\\')[-1] != self.ic_file])
        with open(hec_f, 'w') as outfile:
            for f in hec:
                with open(self.work_dir+f, 'r') as infile:
                    next(infile)
                    for line in infile:
                        outfile.write(line)

        with open(ic_f, 'w') as outfile1:
            for f1 in ic:
                with open(self.work_dir+f1, 'r') as infile1:
                    next(infile1)
                    for line1 in infile1:
                        outfile1.write(line1)
        return path_press
# -------------------------------------------------------------
    @my_logger
    @time_this
    def dir_scan(self):
        '''scan through he data directory. find all and sort
        self.work_dir is changed here.'''
        l_com = [dirs for root, dirs, files in os.walk(self.data_dir)]
        l_dir = sorted(l_com[0])
        for ii in l_dir:
            print(ii)
            self.work_dir = self.data_dir+ii+"\\"
            files = self.find_files()
            path_p = self.find_hec(files)
            self.import_fun(path_p)
# -------------------------------------------------------------
    @my_logger
    @time_this
    def _forselect(self, tb_name, t1, t2):
        assert type(t1) is datetime.datetime and type(
            t2) is datetime.datetime, "t1 and t2 should be datetime"
        assert t2 > t1, "t2 should be > t1"
        query = "SELECT * FROM {} WHERE date >= Datetime('{}') AND date \
                <= Datetime('{}') ORDER BY date ASC".format(
            tb_name, str(t1), str(t2))
        self.cursor.execute(query)
        data = self.cursor.fetchall()
        res = self._removeNull(data)
        return res
# -------------------------------------------------------------
    @my_logger
    @time_this
    def select_interval(self, tb_name, t1, t2):
        '''select data between two timestamps t1 and t2 in datetime format'''
        assert type(t1) is datetime.datetime and type(
            t2) is datetime.datetime, "t1 and t2 should be datetime"
        assert t2 > t1, "t2 should be > t1"
        res = self._forselect(tb_name, t1, t2)
        print("mark!")
        wind = 31
        poly = 1
        date = tuple([datetime.datetime.strptime(
            ii[0], '%Y-%m-%d %H:%M:%S') for ii in res])
        Q1 = tuple([ii[self.qttime.q1] for ii in res])
        Q2 = tuple([ii[self.qttime.q2] for ii in res])
        temp = tuple([ii[self.qttime.temp] for ii in res])
        pres = tuple([ii[self.qttime.pressure] for ii in res])
        time = tuple([ii[self.qttime.time] for ii in res])
        dQ1 = sci.savgol_filter(Q1, wind, poly, deriv=1)
        dQ2 = sci.savgol_filter(Q2, wind, poly, deriv=1)
        time = time-time[0]
        return time, Q1, Q2, temp, pres, date, dQ1, dQ2
# -------------------------------------------------------------
#    @my_logger
    @time_this
    def calc_params(self, dQ1, dQ2, temp1, time1, pres):
        '''calculate Tc and Tab Tba based on derivative
        rate in [mK/hr]'''
        temp = np.array(temp1)
        time = np.array(time1)
        idx = np.isfinite(temp) & np.isfinite(time)
        fit = np.polyfit(time[idx], temp[idx], 1)
        rate = fit[0]*3600
        ind1Q1 = np.argmax(np.abs(dQ1[0:round(0.4*len(dQ1))]))
        ind2Q1 = np.argmax(np.abs(dQ1[round(0.6*len(dQ1)):-1]))
        ind1Q2 = np.argmax(np.abs(dQ2[0:round(0.4*len(dQ2))]))
        ind2Q2 = np.argmax(np.abs(dQ2[round(0.6*len(dQ2)):-1]))
        p1 = np.mean(pres)
        ind2Q1 = ind2Q1+round(0.6*len(dQ1)) 
        ind2Q2 = ind2Q2+round(0.6*len(dQ2))
        print("Pressure is ", p1)
        print("The Ramp is {} mK/hr".format(rate))
        print("First der for HEC ", temp[ind1Q1])
        print("First der for IC ", temp[ind1Q2])
        print("Second der for HEC ", temp[ind2Q1])
        print("Second der for IC ", temp[ind2Q2])
        print("Start temperature is ", temp[0])
        print("Ending temperature is ", temp[-1])
        return rate, ind1Q1, ind2Q1, ind1Q2, ind2Q2



# -------------main--------------------------------------------
if __name__ == '__main__':
    '''28.11 to 11.01'''

    conf_loc = "d:\\dima\\proj\\ab_trans\\ab_data.db"
    data_dir = "d:\\dima\\proj\\ab_trans\\data\\"
    t1 = datetime.datetime(2019, 2, 12, 10, 0, 0, 0)
    t2 = datetime.datetime(2019, 2, 12, 16, 40, 0, 0)
    A = ABData(conf_loc, data_dir)
#    A.dir_scan()
    time, Q1, Q2, temp, pres, date, dQ1, dQ2 = A.select_interval(
        A.table_name, t1, t2)

    p1 = np.mean(pres)
    rate, ind1Q1, ind2Q1, ind1Q2, ind2Q2 = A.calc_params(dQ1, dQ2, temp,
            time, pres)
# ----------Print and plot-------------------------------------
    fig1 = plt.figure(1, clear=True)
    ax1 = fig1.add_subplot(221)
    ax1.set_ylabel('Q')
    ax1.set_xlabel('date')
    ax1.set_title("Q vs time for "+str(p1)+" bar")
    ax1.scatter(date, Q1, color='green', s=0.5, label='HEC')
    ax1.scatter(date, Q2, color='blue', s=0.5, label='IC')
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

    A.close_f()
