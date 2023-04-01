# Import necessary libraries
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# Define the MySQL connection string
USER = 'root'
PASSWORD = ''
HOST = 'localhost'
PORT = '3306'
DATABASE = 'data_science'

connection_string = f'mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}'

# Create an engine that connects to your MySQL database
engine = create_engine(connection_string)

# Create a declarative base object to define the table
Base = declarative_base()

# Define the table


class Client(Base):
    __tablename__ = 'clients'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    age = Column(Integer)


# Create the table in the database
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

# Operations on the table
session.add(Client(name='John', age=25))
session.commit()
