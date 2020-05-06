from abc import ABC, abstractmethod
from logger import log_settings
import os
from typing import Optional

params = {"db_name", "data_dir", "data_table", "pressure_table", "buffer_table"}
app_log = log_settings()


class BuildDb(ABC):
    """
    Buider interface for create a main database table
    Minimal number of elements
    """

    @property
    @abstractmethod
    def product(self) -> None:
        pass

    @abstractmethod
    def open_session(self) -> None:
        pass

    @abstractmethod
    def close_session(self) -> None:
        pass

    @abstractmethod
    def close_engine(self) -> None:
        pass

    def drop_table(self, table) -> None:
        pass

    def create_table(self) -> None:
        pass

    def create_buffer_table(self) -> None:
        pass

    def directory_scan(self) -> None:
        pass

    @property
    @abstractmethod
    def pressure_table(self):
        pass

    @property
    @abstractmethod
    def buffer_table(self):
        pass

    @property
    @abstractmethod
    def data_table(self):
        pass


class ConcreteDbBuilder(BuildDb):
    """
    Provide specific implementation
    """
    def __init__(self, obj, config) -> None:
        """
        :param obj: Db class form where it is buid
        """
        self.config_validator(config)
        self.config = config
        self.obj = obj
        self.reset()

    @staticmethod
    def config_validator(config):
        """
        Checks configuration for db creqtion
        """
        if params != config.keys():
            app_log.error(f"Configuration does not have all required parameters"
                          "\n"
                          f"you input: {config.keys()}\n"
                          f"must have: {params}")
            raise IndexError("Configuration do not have required parameters")
        if config.get("data_dir") is not None:
            if not os.path.isdir(config.get("data_dir")):
                app_log.error("Data directory does not exists")
                raise FileNotFoundError("Directory does not exists")

    def reset(self):
        """
        Creates an instance of the obj class i.e. LocalDb
        """
        self._product = self.obj(self.config.get("db_name"), self.config.get("data_dir"))

    @property
    def pressure_table(self):
        return self.config.get("pressure_table")

    @property
    def buffer_table(self):
        return self.config.get("buffer_table")

    @property
    def data_table(self):
        return self.config.get("data_table")

    @property
    def product(self):
        product = self._product
        self.reset()
        return product

    def open_session(self) -> None:
        self._product.open_session()

    def close_session(self) -> None:
        self._product.close_session()

    def close_engine(self) -> None:
        self._product.close_engine()

    def drop_table(self, table) -> None:
        self._product.drop_table(table)

    def create_table(self) -> None:
        self._product.create_table()

    def create_buffer_table(self) -> None:
        self._product.create_buffer_table()

    def directory_scan(self):
        self._product.dir_scan()


class DirectorDb:
    """
    Creates specific configurations of building
    """
    def __init__(self) -> None:
        self._builder = None

    @property
    def builder(self) -> Optional[BuildDb]:
        return self._builder

    @builder.setter
    def builder(self, builder: BuildDb):
        self._builder = builder

    def build_full_db(self) -> None:
        app_log.info("Full builder starts")
        self.builder.drop_table(self.builder.buffer_table)
        self.builder.drop_table(self.builder.pressure_table)
        self.builder.drop_table(self.builder.data_table)
        self.builder.create_table()
        self.builder.create_buffer_table()
        self.builder.open_session()
        self.builder.directory_scan()
        self.builder.close_session()
        self.builder.close_engine()
        app_log.info("Full builder ends")

    def build_minimal(self) -> None:
        app_log.info("Minimal builder starts")
        self.builder.open_session()
        self.builder.close_session()
        self.builder.close_engine()
        app_log.info("Minimal builder ends")


