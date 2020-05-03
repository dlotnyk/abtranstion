import logging
from logging.handlers import RotatingFileHandler


class Logger():
    logfile = "app.log"
    logname = "ForksFT"


def log_settings():
    #  Logger definitions
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - line: %(lineno)d - %(message)s')
    logFile = "app.log"
    my_handler = RotatingFileHandler(logFile, mode="a", maxBytes=2*1024*1024, backupCount=2, encoding=None, delay=False)
    my_handler.setFormatter(log_formatter)
    my_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    app_log = logging.getLogger("AbDatabase")
    app_log.setLevel(logging.INFO)
    if len(app_log.handlers) < 2:
        app_log.addHandler(my_handler)
        app_log.addHandler(console_handler)
    return app_log