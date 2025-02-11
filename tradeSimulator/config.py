import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Config:
    @staticmethod
    def get_log_level():
        return os.getenv("LOG_LEVEL", "INFO")

    @staticmethod
    def get_singlestore_db_url():
        return os.getenv("SINGLESTORE_DB_URL")

    @staticmethod
    def get_db_pool_size():
        return int(os.getenv("DB_POOL_SIZE", "10"))

    @staticmethod
    def get_throughput():
        return int(os.getenv("THROUGHPUT", "10"))

    @staticmethod
    def get_mode():
        return os.getenv("MODE", "db")

    @staticmethod
    def get_num_threads():
        return int(os.getenv("NUM_THREADS", "8"))

    @staticmethod
    def get_local_csv_path():
        return os.getenv("LOCAL_CSV_PATH", "./trades_data.csv")

    @staticmethod
    def get_log_interval():
        return int(os.getenv("LOG_INTERVAL", "5"))

    @staticmethod
    def get_batch_size():
        return int(os.getenv("BATCH_SIZE", "10"))
