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
def time_this(original_function):  
    '''Measures the processing time. Decorator'''
    @wraps(original_function)                      
    def new_function(*args,**kwargs):    
        import time       
        before = time.time()                     
        x = original_function(*args,**kwargs)                
        after = time.time()                      
        print ("Elapsed Time of fun {0} = {1}".format(original_function.__name__,after-before))      
        return x                                             
    return new_function  
def my_logger(orig_func):
    '''Decorate function to write into log on the level ERROR'''
    import logging
    logging.getLogger('').handlers = []
    logging.basicConfig(filename='work.log'.format(orig_func.__name__), level=logging.INFO)
    @wraps(orig_func)
    def wrapper(*args,**kwargs):
        frm = str(inspect.stack()[1][0])
        ff1=str(frm.split(' code ')[1])
        ff2=str(ff1.split('>')[0])
        dt=datetime.datetime.now()
        dt_str=str(dt)
        vrema=dt_str.split('.')[0]
        logging.info(
                ' {} Ran with args: {}, and kwargs: {} \n'.format(vrema, args,kwargs))
        return orig_func(*args, **kwargs) 
    return wrapper 
    
class abdata():
    '''read from fies, create a sql database from date, universal time, Q, Tmc and pressure'''
    def __init__(self,conf):
        self.conf=conf
        self.connect_loc(conf)
        
#-------------------------------------------------------------
    def __repr__(self):
        return "{}, db: {}".format(self.__class__.__name__,self.conf)
    @my_logger  
    @time_this 
    def connect_loc(self,conf):   
        '''open connection to sql lite database'''
        assert type(conf) is str, "input parameter must be str"
        self.autoinc=''
        self.cnx=sql.connect(conf)
        self.cursor = self.cnx.cursor()
        print('Connected!')
#-------------------------------------------------------------
    @my_logger  
    @time_this
    def close_f(self):
        '''close connection'''
        self.cursor.close()
        self.cnx.close()
        print('Disconnected!')
#-------------------------------------------------------------
    @my_logger
    @time_this
    def import_fun(self):
        '''import needed data from dat files
        time/date [0] [1]; unitime [2]; Q [6]; Tmc [13]
        update an sql table'''
        path="c:\\Users\\JMP\\Documents\\Thermal Conductivity\\Backup\\2019JAN\\20190102\\HEC2p2mK.dat"
        path1="c:\\Users\\JMP\\Documents\\Thermal Conductivity\\Backup\\2019JAN\\20190102\\IC2p2mK.dat"
        path2="c:\\Users\\JMP\\Documents\\Thermal Conductivity\\Backup\\2019JAN\\20190102\\pressure_log20190102.dat"
        ps=path.split('\\')
        ls=len(ps[-1])
#        print(ps[-1])
#        print(path[0:-ls])
#        data=np.genfromtxt(path, unpack=True, skip_header=1, usecols = (4, 5, 2, 6, 13, 7))
#        data=np.genfromtxt(path, unpack=True, skip_header=1)
#        print(data[0][0],data[1][0],data[2][0])
        with open(path,'r') as f, open(path1,'r') as f1, open(path2,'r') as f2:
            next(f) # skip first
            next(f1)
            next(f2)
            d2=next(f2).split()
            press=float(d2[3])
            for line, line1 in zip(f,f1):
                d=line.split()
                d1=line1.split()
                mm,dd,yy,hh,mins,ss=self.__gettime(d[0],d[1])
                mm1,dd1,yy1,hh1,mins1,ss1=self.__gettime(d2[0],d2[1])
                a=datetime.datetime(int('20'+yy),int(mm),int(dd),int(hh),int(mins),int(ss),0)
                a1=datetime.datetime(int('20'+yy1),int(mm1),int(dd1),int(hh1),int(mins1),int(ss1),0)
                if a1<a:
                    try:
                        d2=next(f2).split()
                        mm1,dd1,yy1,hh1,mins1,ss1=self.__gettime(d2[0],d2[1])
                        press=float(d2[3])
                    except StopIteration:
                        press=press
                self.__update_table('my_t',(a,int(d[2]),float(d[6]),float(d1[6]),float(d[13]),press))
            self.cnx.commit()
            
#-------------------------------------------------------------    
    @my_logger
    @time_this
    def find_files(self):
        '''find dat files in directory'''
        dir1="c:\\Users\\JMP\\Documents\\Thermal Conductivity\\Backup\\2019JAN\\20190102\\"
        for root,dirs,files in os.walk(dir1):
            ll=[os.path.join(root,file) for file in files if file.endswith(".dat")]
        return ll
#-------------------------------------------------------------    
    
    def __gettime(self,date,time):
        '''get time in datetime from strings date and time'''
        mm,dd,yy=date.split('/')
        hh,mins,ss=time.split(':')
        return mm,dd,yy,hh,mins,ss
#-------------------------------------------------------------    
    @my_logger
    @time_this
    def create_table(self,db_name):   
        '''Create table in local database'''
        query=("CREATE TABLE IF NOT EXISTS {} ("
#               "id integer PRIMARY KEY, "
               "date text NOT NULL, "
               "uni_time integer NOT NULL, "
               "Q1 real NOT NULL, "
               "Q2 real NOT NULL, "
               "Tmc real, "
               "pressure real NOT NULL);".format(db_name))
        self.cursor.execute(query)
#-------------------------------------------------------------    
    @my_logger
    @time_this
    def drop_table(self,db_name):  
        '''drop db_name table'''
        query=("DROP TABLE {}".format(db_name))
        self.cursor.execute(query)
#-------------------------------------------------------------    
    def __update_table(self,db_name,values):
        '''set a new values to the table'''
#        print(len(values))
        query="INSERT INTO %s (date, uni_time, Q1, Q2, Tmc, pressure) VALUES (?, ?, ?, ?, ?, ?) " % db_name
#        print(query)
        self.cursor.execute(query,values)
#-------------------------------------------------------------            
    def select_vals(self,tb_name):
        '''select all values'''
        query="SELECT * FROM %s WHERE 1" % tb_name
        self.cursor.execute(query)
        data=self.cursor.fetchall()
        res=self._removeNull(data)
#        print(data)
        return res
#-------------------------------------------------------------    
    def _removeNull(self,res):
        '''remove nulls and convert it to pyhonic nan's'''
#        res1=np.transpose(res)
        kerneldt=np.dtype({'names':['time','uni_t','Q1','Q2','Tmc','pressure'],'formats':['U20',float,float,float,float,float]})
        dat=np.zeros(np.shape(res)[0],dtype=kerneldt)
        assert len(dat) > 0, "no data were taken from SELECT"
        for ind,x in enumerate(res):
            for y in x:
                if y is None:
                    y=np.nan
            dat[ind]=x
        return dat       
#-------------main--------------------------------------------            
conf_loc='ab_data.db'
A=abdata(conf_loc)
#files=A.find_files()
A.drop_table('my_t')
A.create_table('my_t')
A.import_fun()
data1=A.select_vals('my_t')
A.close_f()