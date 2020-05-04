"""
Created on Mon Jan 14 13:17:13 2019

@author: dlotnyk
"""
import numpy as np
import sqlalchemy as db
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from typing import List, Tuple

from logger import log_settings
from db_tables import PrTable, DataTable, BufferTable, Base, buf_table_name, pr_table_name, raw_table_name
app_log = log_settings()
# Base = declarative_base()
n_hec_combine = "hs_combine.dat"
n_ic_combine = "is_combine.dat"


class LocalDb:
    table_name = raw_table_name
    rawdata_tb = None

    def __init__(self, db_name: str, data_dir: str) -> None:
        """
        Creates db from data dir folder
        :param db_name: database name
        :param data_dir: Directory with all data
        """
        try:
            self.db_name = db_name
            self.data_dir = data_dir
            parent_path = os.path.dirname(os.getcwd())
            os.chdir(parent_path)
            self.db_path = os.path.join(parent_path, self.db_name)
            connector = "sqlite:///" + self.db_path
            self.db_engine = db.create_engine(connector)
            self.session = None
            app_log.info(f"Engine creates: `{self.db_name}`")
        except Exception as ex:
            app_log.error(f"Can not create engine: `{ex}`")

    def create_table(self):
        """
        Creates a table with table name
        """
        metadata = db.MetaData()
        self.rawdata_tb = db.Table(self.table_name, metadata,
                                   db.Column("id", db.Integer, primary_key=True, autoincrement=True),
                                   db.Column("date", db.String, unique=True, index=True),
                                   db.Column("uni_time"),
                                   db.Column("Q_hec", db.Float),
                                   db.Column("Q_ic", db.Float),
                                   db.Column("Tmc", db.Float),
                                   db.Column("pressure", db.Float)
                                   )
        try:
            Base.metadata.create_all(self.db_engine)
        except Exception as ex:
            app_log.error(f"Can not create table: `{ex}`")
        else:
            app_log.info(f"Table `{self.table_name}` was successfully created")

    def create_pressure_table(self):
        """
        Creates a table with table name
        """
        metadata = db.MetaData()
        self.pressure_tb = db.Table(pr_table_name, metadata,
                                    db.Column("id", db.Integer, primary_key=True, autoincrement=True),
                                    db.Column("date", db.String),
                                    db.Column("uni_time", db.Integer, unique=True, index=True),
                                    db.Column("pressure", db.Float)
                                    )
        try:
            Base.metadata.create_all(self.db_engine)
        except Exception as ex:
            app_log.error(f"Can not create pressure table: `{ex}`")
        else:
            app_log.info(f"Table `{pr_table_name}` was successfully created")

    def create_buffer_table(self):
        """
        Creates a table with table name
        """
        metadata = db.MetaData()
        self.buf_tb = db.Table(buf_table_name, metadata,
                               db.Column("id", db.Integer, primary_key=True, autoincrement=True),
                               db.Column("date", db.String),
                               db.Column("uni_time", db.Integer, unique=True, index=True),
                               db.Column("Q_hec", db.Float),
                               db.Column("Q_ic", db.Float),
                               db.Column("Tmc", db.Float),
                               )
        try:
            Base.metadata.create_all(self.db_engine)
        except Exception as ex:
            app_log.error(f"Can not create pressure table: `{ex}`")
        else:
            app_log.info(f"Table `{buf_table_name}` was successfully created")

    def drop_table(self, obj):
        """
        Drops table
        """
        try:
            obj.__table__.drop(self.db_engine)
            app_log.info(f"Table `{obj.__tablename__}` successfully drops")
        except db.exc.OperationalError:
            app_log.warning(f"Can not drop `{obj.__tablename__}` table. It does not exists")
        except Exception as ex:
            app_log.error(f"Can not drop `{obj.__tablename__}`: {ex}")

    def open_session(self):
        """
        Opens the local db
        """
        try:
            sess = sessionmaker(bind=self.db_engine)
            self.session = sess()
            app_log.info(f"Session creates for: `{self.db_name}`")
        except Exception as ex:
            app_log.error(f"Can not create session: {ex}")

    def close_session(self):
        """
        Close connection to db
        """
        try:
            if self.session is not None:
                self.session.close()
                app_log.info(f"Session `{self.db_name}` closed")
        except Exception as ex:
            app_log.error(f"Can not close session: {ex}")

    def close_engine(self):
        """
        Close the db engine
        """
        try:
            self.db_engine.dispose()
            app_log.info("db Engine disposed")
        except Exception as ex:
            app_log.error(f"Engine NOT disposed: {ex}")

    def dir_scan(self):
        """
        scan through he data directory. find all and sort
        self.work_dir is changed here.
        :raise: Raises an error if scan fails
        """
        l_com = [dirs for root, dirs, files in os.walk(self.data_dir)]
        l_dir = sorted(l_com[0])
        try:
            for ii in l_dir:
                app_log.info(f"Check dir: `{ii}`")
                work_dir = self.data_dir+ii+"\\"
                if not self.is_folder_checked(work_dir):
                    self.create_pressure_table()
                    hec_files, ic_files, p_list = self.find_files(work_dir)
                    if self.files_validator(work_dir, hec_files, ic_files, p_list):
                        self.drop_table(PrTable)
                        continue
                    else:
                        hec_combine = self.combine_files(work_dir, hec_files)
                        ic_combine = self.combine_files(work_dir, ic_files, is_hec=False)
                        p_path = os.path.join(work_dir, p_list[0])
                        self.parse_pressure(p_path)
                        self.parse_insert(hec_combine, ic_combine)
                        self.insert_into_main()
                        self.drop_table(PrTable)
                        self.drop_table(BufferTable)
                else:
                    app_log.info("Directory already parsed")
        except Exception as ex:
            app_log.error(f"Error while dir scan: {ex}")
            self.close_engine()
            raise

    @staticmethod
    def is_folder_checked(work_dir):
        """
        Checked if folder was parsed before:
        :param work_dir: path to folder to check
        :return: True of False
        """
        files_list = [files for root, dirs, files in os.walk(work_dir)]
        if n_ic_combine in files_list[0] and n_hec_combine in files_list[0]:
            return True
        else:
            return False

    @staticmethod
    def find_files(work_dir: str) -> Tuple[List, List, List]:
        """
        find dat files in directory, ignore sweep
        :param work_dir: path to working directory
        :return: tuple of list for HEC, IC and pressure files
        """
        hec_list: List = list()
        ic_list: List = list()
        p_list: List = list()
        for root, dirs, files in os.walk(work_dir):
            hec_list = [os.path.join(root, file)
                  for file in files if file.endswith(".dat") and "sweep" not in file.lower()
                         and "hec" in file.lower()
                        ]
            ic_list = [os.path.join(root, file)
                        for file in files if file.endswith(".dat") and "sweep" not in file.lower()
                        and "ic" in file.lower()
                       ]
            p_list = [os.path.join(root, file)
                       for file in files if file.endswith(".dat") and "sweep" not in file.lower()
                       and "pres" in file.lower()
                       ]
        return sorted(hec_list), sorted(ic_list), sorted(p_list)

    @staticmethod
    def files_validator(work_dir: str,
                        hec_list: List,
                        ic_list: List,
                        p_list: List) -> bool:
        """
        Checks if list are not empty.
        :param work_dir: path to working directory
        :param hec_list: HEC files list
        :param ic_list: IC files list
        :param p_list: pressure files list
        :return: True if something wrong
        """
        if len(hec_list) == 0 or len(ic_list) == 0 or len(p_list) == 0:
            app_log.warning(f"Folder `{work_dir}` does not have required files")
            return True
        else:
            return False

    @staticmethod
    def combine_files(work_dir: str, file_list: List, is_hec=True) -> str:
        """
        combines files in directory into one.
        :param work_dir: path to working directory
        :param file_list: List of files in directory
        :param is_hec: True if list of HEC files.
        :return: path to combined file
        """
        if is_hec:
            file_name = work_dir + n_hec_combine
        else:
            file_name = work_dir + n_ic_combine
        with open(file_name, "w") as output_file:
            for file in file_list:
                with open(file, "r") as inner:
                    next(inner)   # skip first line
                    for line in inner:
                        output_file.write(line)
        return file_name

    def add_table_entry(self, entry):
        """
        Add an entry to table
        :param entry: entry to INSERT
        Handles UNIQUE exception
        """
        try:
            self.session.add(entry)
        except db.exc.IntegrityError:
            app_log.warning(f"Can not INSERT entry: {entry}")

    def insert_into_main(self):
        """
        Insert data into the main rawtable
        """
        try:
            app_log.info(f"Start insert to `{DataTable.__tablename__}`...")
            buf_rec = self.session.query(BufferTable).order_by(BufferTable.uni_time).all()
            pres_rec = self.session.query(PrTable).order_by(PrTable.uni_time).all()
            counter = 0
            pressure = pres_rec[counter].pressure
            for item in buf_rec:
                try:
                    if item.uni_time > pres_rec[counter].uni_time:
                        counter += 1
                        pressure = pres_rec[counter].pressure
                except IndexError:
                    pressure = pressure
                data_db = DataTable(date=item.date,
                                    uni_time=item.uni_time,
                                    Q_hec=item.Q_hec,
                                    Q_ic=item.Q_ic,
                                    Tmc=item.Tmc,
                                    pressure=pressure
                                    )
                self.add_table_entry(data_db)
        except Exception as ex:
            app_log.error(f"Can not insert into main table: {ex}")
        else:
            self.session.commit()
            app_log.info(f"Data committed to `{DataTable.__tablename__}`")

    def parse_insert(self, path_hec: str, path_ic: str):
        """
        Parse files and update the bufffer db table.
        :param path_hec: path to hec file
        :param path_ic: path to ic file
        """
        try:
            app_log.info(f"Start inserting to `{BufferTable.__tablename__}`...")
            with open(path_hec, "r") as f_hec, open(path_ic, "r") as f_ic:
                for line_hec, line_ic in zip(f_hec, f_ic):
                    d_hec = line_hec.split()
                    d_ic = line_ic.split()
                    dstr = self.get_date(d_hec[0], d_hec[1])
                    dtime = self.str_to_datetime(dstr)
                    data_db = BufferTable(date=dstr,
                                          uni_time=int(dtime.timestamp()),
                                          Q_hec=float(d_hec[6]),
                                          Q_ic=float(d_ic[6]),
                                          Tmc=float(d_hec[13])
                                          )
                    self.add_table_entry(data_db)
        except Exception as ex:
            app_log.error(f"Fails inserting: {ex}")
        else:
            self.session.commit()
            app_log.info(f"Data committed to `{BufferTable.__tablename__}`")

    def parse_pressure(self, path_p: str):
        """
        Parse files and update the db table.
        :param path_p: path to pressure file
        """
        try:
            app_log.info(f"Start inserting to `{PrTable.__tablename__}`...")
            with open(path_p, "r") as f_p:
                next(f_p)
                for line_p in f_p:
                    d_pres = line_p.split()
                    dstr = self.get_date(d_pres[0], d_pres[1])
                    dtime = self.str_to_datetime(dstr)
                    data_db = PrTable(date=dstr,
                                      uni_time=int(dtime.timestamp()),
                                      pressure=float(d_pres[3]))
                    self.add_table_entry(data_db)
        except Exception as ex:
            app_log.error(f"Fails inserting: {ex}")
        else:
            self.session.commit()
            app_log.info(f"Data committed to `{PrTable.__tablename__}`")

    @staticmethod
    def get_date(dats: str, tims: str) -> str:
        """
        Transforms Labview date time into datetime like string
        :param dats: labview date
        :param tims: Labview time
        :return: datetime like string
        """
        mm, dd, yy = dats.split("/")
        hh, mins, ss = tims.split(":")
        return "20" + yy + "-" + mm + "-" + dd + " " + hh + ":" + mins + ":" + ss

    def find_closest_time(self, dtime: int) -> float:
        """
        Find closest timestamp between data and pressure tables
        :param dtime: time in integer
        :return: pressure value closest to dtime timestamp
        """
        rec = self.session.query(PrTable).order_by(db.func.abs(dtime - PrTable.uni_time)).first()
        return rec.pressure

    def select_time(self, start: datetime, stop: datetime) -> np.ndarray:
        """
        Makes select between start and stop and returns an numpy array
        :param start: start time of range as datetime object
        :param stop: stop time of range as datetime object
        :return: fetched data as numpy array
        """
        try:
            app_log.info("Selecting requested data...")
            count = self.session.query(DataTable).filter(DataTable.date >= start).\
                      filter(DataTable.date <= stop).count()
            self.check_response(count)
            kerneldt = np.dtype({'names': ['id', 'time', 'uni_t', 'Q1', 'Q2', 'Tmc', \
                                           'pressure'], 'formats': [np.int,
                'U20', np.float32, np.float32, np.float32,
                np.float32, np.float32]})
            array = np.zeros(count, dtype=kerneldt)
            rec = self.session.query(DataTable).filter(DataTable.date >= start).\
                      filter(DataTable.date <= stop).order_by(DataTable.date)
            for idx, item in enumerate(rec):
                item = self.check_null(item)
                array[idx] = (item.id, item.date, item.uni_time,
                              item.Q_hec, item.Q_ic, item.Tmc, item.pressure)
            app_log.info("Response array generated")
            return array
        except ValueError:
            app_log.warning("There are no data in selected date range!")
        except Exception as ex:
            app_log.error(f"Error selecting data from table: {ex}")

    @staticmethod
    def check_response(count: int) -> None:
        """
        Checks count of select fetch.
        :raise: Value error if zero data
        """
        if count == 0:
            raise ValueError

    @staticmethod
    def str_to_datetime(dats):
        return datetime.strptime(dats, "%Y-%m-%d %H:%M:%S")

    @staticmethod
    def check_null(value: DataTable) -> DataTable:
        """
        Checks fetched data for None, replaces with numpy.nan
        :param value: instanse of DataTable i.e. the fetched data from select
        """
        if value.Q_hec is None:
            value.Q_hec = np.nan
        if value.Q_ic is None:
            value.Q_ic = np.nan
        if value.Tmc is None:
            value.Tmc = np.nan
        if value.pressure is None:
            value.pressure = np.nan
        return value


if __name__ == "__main__":
    app_log.info("Create db app starts.")
    db_name = "ab_data2.db"
    data_dir = "k:\\data\\lab\\abdata\\"
    start = datetime(2019, 1, 25, 10, 0)
    stop = datetime(2019, 1, 25, 10, 30)
    loc = LocalDb(db_name, data_dir)
    loc.drop_table(BufferTable)
    loc.drop_table(PrTable)
    # loc.drop_table(DataTable)
    # loc.create_table()
    loc.create_buffer_table()
    loc.open_session()
    loc.dir_scan()
    # arr = loc.select_time(start, stop)
    loc.close_session()
    loc.close_engine()
    app_log.info("Create db app ends")
