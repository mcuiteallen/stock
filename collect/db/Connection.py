from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

class Connection:
    """Connection class"""
    session = None
    base = None
    
    def __init__(self):
        engine = create_engine('postgresql://stock:password@172.22.12.211:5432/stock')
        self.session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
        self.base = declarative_base()
        self.base.query = self.session.query_property()
        
    def getSession(self):
        return self.session
        
    def getBase(self):
        return self.base
        
connection = Connection()
session = connection.getSession()
base = connection.getBase()
