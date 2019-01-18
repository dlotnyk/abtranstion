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
        frm = str(inspect.stack()[1][4][0])
        dt=datetime.datetime.now()
        dt_str=str(dt)
        vrema=dt_str.split('.')[0]
        logging.info(
                ' {} Ran with args:{}, {}, and kwargs: {}'.format(vrema,frm[0:-1],args,kwargs))
        return orig_func(*args, **kwargs) 
    return wrapper 
#-------------------------------------------------------------
#-------------------------------------------------------------    
class abdata():
    '''read from fies, create a sql database from date, universal time, Q, Tmc and pressure'''
    
#    test_path="c:\\dima\\proj\\ab_trans\\20181201\\"
    hec_file="HEC_combine.dat"
    ic_file="IC_combine.dat"
    work_dir=""
    def __init__(self,conf,data_dir):
        self.conf=conf
        self.data_dir=data_dir
        self.connect_loc(conf)
#-------------------------------------------------------------
    def __repr__(self):
        return "{}, db: {}".format(self.__class__.__name__,self.conf)
#-------------------------------------------------------------    
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
    def import_fun(self,path_press):
        '''import needed data from dat files
        time/date [0] [1]; unitime [2]; Q [6]; Tmc [13]
        update an sql table'''
        path = self.work_dir+self.hec_file
        path1=self.work_dir+self.ic_file
        path2=self.work_dir+path_press[0]
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
        for root,dirs,files in os.walk(self.work_dir):
            ll={os.path.join(root,file) for file in files if file.endswith(".dat")}
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
               "Q1 real, "
               "Q2 real, "
               "Tmc real, "
               "pressure real);".format(db_name))
        self.cursor.execute(query)
#-------------------------------------------------------------    
    @my_logger
    @time_this
    def drop_table(self,tb_name):  
        '''drop db_name table'''
        query=("DROP TABLE {}".format(tb_name))
        self.cursor.execute(query)
#-------------------------------------------------------------    
    def __update_table(self,tb_name,values):
        '''set a new values to the table'''
        query="INSERT INTO %s (date, uni_time, Q1, Q2, Tmc, pressure) VALUES (?, ?, ?, ?, ?, ?) " % tb_name
        self.cursor.execute(query,values)
#-------------------------------------------------------------            
    @my_logger
    @time_this
    def select_vals(self,tb_name):
        '''select all values'''
        query="SELECT * FROM %s WHERE 1  ORDER BY date ASC" % tb_name
        self.cursor.execute(query)
        data=self.cursor.fetchall()
        res=self._removeNull(data)
#        print(data)
        return res
#-------------------------------------------------------------    
    def _removeNull(self,res):
        '''remove nulls and convert it to pyhonic nan's'''
#        res1=np.transpose(res)
        kerneldt=np.dtype({'names':['time','uni_t','Q1','Q2','Tmc','pressure'],'formats':['U20',np.float32,np.float32,np.float32,np.float32,np.float32]})
        dat=np.zeros(np.shape(res)[0],dtype=kerneldt)
        assert len(dat) > 0, "no data were taken from SELECT"
        for ind,x in enumerate(res):
            for y in x:
                if y is None:
                    y=np.nan
            dat[ind]=x
        return dat       
    #-------------------------------------------------------------            
    @my_logger
    @time_this
    def find_hec(self,files):
        '''find all HEC and IC files. combine them into single HEC and IC. find a path with pressure'''
        hec_f=self.work_dir+self.hec_file
        ic_f=self.work_dir+self.ic_file
        path_press=sorted([k.split('\\')[-1] for k in files if k.split('\\')[-1][0:3]=='pre'])
        hec=sorted([jj.split('\\')[-1] for jj in files if jj.split('\\')[-1][0:3]=='HEC'
                    and jj.split('\\')[-1] != self.hec_file])
        ic=sorted([ii.split('\\')[-1] for ii in files if ii.split('\\')[-1][0:2]=='IC'
                   and ii.split('\\')[-1] != self.ic_file])
        with open (hec_f, 'w') as outfile:
            for f in hec:
                with open(self.work_dir+f,'r') as infile:
                    next(infile)
                    for line in infile:
                        outfile.write(line)
                        
        with open (ic_f, 'w') as outfile1:
            for f1 in ic:
                with open(self.work_dir+f1,'r') as infile1:
                    next(infile1)
                    for line1 in infile1:
                        outfile1.write(line1)
        return path_press
#-------------------------------------------------------------   
    @my_logger
    @time_this
    def dir_scan(self):
        '''scan through he data directory. find all and sort
        self.work_dir is changed here.'''
        l_com = [dirs for root,dirs,files in os.walk(self.data_dir)]
        l_dir = sorted(l_com[0])
        for ii in l_dir:
            print(ii)
            self.work_dir=self.data_dir+ii+"\\"
            files=self.find_files()
            path_p=self.find_hec(files)
            self.import_fun(path_p)

#-------------main--------------------------------------------   
if __name__ == '__main__':       
    conf_loc='ab_data.db'
    data_dir="c:\\dima\\proj\\ab_trans\\data\\"
    A=abdata(conf_loc,data_dir)
#    A.drop_table('my_t')
    A.create_table('my_t')
    A.dir_scan()
    data1=A.select_vals('my_t')
    A.close_f()