import mysql.connector as msql
from abres import abdata
from abres import time_this
from abres import my_logger
from mysql.connector import errorcode
import sys
sys.path.insert(0,'c:\\dima\\proj\\ab_trans')
from configa import config
# another class for mysql connection
class mysabdata(abdata):
    def __init__(self, conf, data_dir):
#       self.conf=conf
#       self.connect_loc(conf)
        abdata.__init__(self,conf, data_dir)
#--------------------------------------------------
    @my_logger
    @time_this
    def connect_loc(self,conf):
        '''reconfigure acc to mysql'''
        assert type(conf) is dict, "Input parameter should be dict!!!"
        self.autoinc='AUTO_INCREMENT'
        self.db_name=conf['database']
        self.cnx = None
        print(conf)
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

#-------------------main---------------- 
if __name__ == '__main__':    
    data_dir="c:\\dima\\proj\\ab_trans\\data\\"
    
    B=mysabdata(config, data_dir)
    B.close_f()
