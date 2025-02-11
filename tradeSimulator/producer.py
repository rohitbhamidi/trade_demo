from tradeSimulator.db_handler import DBHandler
from tradeSimulator.config import Config

class DBProducer:
    def __init__(self, db_url):
        self.db_handler = DBHandler(db_url)

    def produce_batch(self, trades):
        self.db_handler.insert_trades(trades)

    def close(self):
        pass

def get_producer(mode):
    if mode == "db":
        return DBProducer(Config.get_singlestore_db_url())
    else:
        raise ValueError("Unsupported mode")
