from db.Connection import base
from sqlalchemy import Column, String, Numeric
from sqlalchemy.orm import backref, relationship

class StockHistory(base):
    __tablename__ = 'stock_history'
    code = Column(String, primary_key=True)
    type = Column(String, primary_key=True)
    date = Column(String, primary_key=True)
    capacity = Column(Numeric)
    turnover = Column(Numeric)
    open = Column(Numeric)
    high = Column(Numeric)
    low = Column(Numeric)
    close = Column(Numeric)
    rsi = Column(Numeric)
    williams = Column(Numeric)
    macd = Column(Numeric)
    macdsignal = Column(Numeric)
    macdhist = Column(Numeric)
    upperband = Column(Numeric)
    middleband = Column(Numeric)
    lowerband = Column(Numeric)
    
    def as_simple_dict(self):
        return {
            'date': self.date,
            'capacity': float(self.capacity),
            'turnover': float(self.turnover),
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'close': float(self.close)
        }
        
    def as_all_dict(self):
        return {
            'date': self.date,
            'capacity': float(self.capacity),
            'turnover': float(self.turnover),
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'rsi': float(self.rsi),
            'williams': float(self.williams),
            'macd': float(self.macd),
            'macdsignal': float(self.macdsignal),
            'macdhist': float(self.macdhist),
            'upperband': float(self.upperband),
            'middleband': float(self.middleband),
            'lowerband': float(self.lowerband)
        }
