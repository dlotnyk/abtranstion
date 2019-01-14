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


conf_loc='ab_data.db'
A=abdata(conf_loc)
A.close_f()