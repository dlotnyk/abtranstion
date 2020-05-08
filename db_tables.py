import sqlalchemy as db
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

pr_table_name = "pr_table"
raw_table_name = "rawdata"
buf_table_name = "buffer_table"


class PrTable(Base):
    """
    Pressure table
    """
    __tablename__ = pr_table_name
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    date = db.Column(db.String)
    uni_time = db.Column(db.Integer, unique=True, index=True)
    pressure = db.Column(db.Float)

    def __init__(self, date, uni_time, pressure):
        self.date = date
        self.uni_time = uni_time
        self.pressure = pressure

    def __repr__(self):
        return "Pressure Table"


class DataTable(Base):
    """
    ORM table for Raw Data
    """
    __tablename__ = raw_table_name
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    date = db.Column(db.String, unique=True, index=True)
    uni_time = db.Column(db.Integer)
    Q_hec = db.Column(db.Float)
    Q_ic = db.Column(db.Float)
    Tmc = db.Column(db.Float)
    pressure = db.Column(db.Float)

    def __init__(self, date, uni_time, Q_hec, Q_ic, Tmc, pressure):
        self.date = date
        self.uni_time = uni_time
        self.Q_hec = Q_hec
        self.Q_ic = Q_ic
        self.Tmc = Tmc
        self.pressure = pressure

    def __repr__(self):
        return "Data Table"


class BufferTable(Base):
    """
    ORM table for Buffer Data
    """
    __tablename__ = buf_table_name
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    date = db.Column(db.String)
    uni_time = db.Column(db.Integer, unique=True, index=True)
    Q_hec = db.Column(db.Float)
    Q_ic = db.Column(db.Float)
    Tmc = db.Column(db.Float)

    def __init__(self, date, uni_time, Q_hec, Q_ic, Tmc):
        self.date = date
        self.uni_time = uni_time
        self.Q_hec = Q_hec
        self.Q_ic = Q_ic
        self.Tmc = Tmc

    def __repr__(self):
        return "Buffer Table"


if __name__ == "__main__":
    print(repr(BufferTable))
    print(BufferTable.__dict__)
