import logging
import pandas as pd
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tradeSimulator.config import Config
from tradeSimulator.logger_config import setup_logging
from tradeSimulator.producer import get_producer
from tradeSimulator.utils import RateLimiter
from tenacity import retry, wait_exponential, stop_after_attempt
import random

logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    df = pd.read_csv(Config.get_local_csv_path())
    required_cols = [
        "localTS", "localDate", "ticker", "conditions", "correction", "exchange",
        "id", "participant_timestamp", "price", "sequence_number", "sip_timestamp",
        "size", "tape", "trf_id", "trf_timestamp"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
            logger.warning(f"Column {col} not found in CSV. Created dummy column with default values.")
    df['conditions'] = df['conditions'].fillna('')
    df['conditions'] = df['conditions'].astype(str)
    return df

def simulate_trades(throughput: int, mode: str, batch_size: int, num_threads: int):
    df = load_data()
    producer = get_producer(mode)
    total_trades = 0
    rate_limiter = RateLimiter(throughput)
    last_log_time = time.time()

    def send_batch(batch_trades):
        producer.produce_batch(batch_trades)
        return len(batch_trades)

    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            while True:
                batch = df.sample(n=batch_size, replace=True)
                current_dt = datetime.now()
                current_ts_str = current_dt.strftime("%Y-%m-%d %H:%M:%S")
                current_date_str = current_dt.strftime("%Y-%m-%d")
                current_ts_ns = int(current_dt.timestamp() * 1_000_000_000)

                batch['localTS'] = current_ts_str
                batch['localDate'] = current_date_str
                batch['participant_timestamp'] = current_ts_ns
                batch['sip_timestamp'] = current_ts_ns
                batch['trf_timestamp'] = current_ts_ns

                trades_list = batch.to_dict(orient='records')

                with rate_limiter:
                    future = executor.submit(send_batch, trades_list)
                    futures.append(future)

                now = time.time()
                if now - last_log_time > Config.get_log_interval():
                    logger.info(f"Sent {total_trades} trades so far at ~{throughput} trades per second.")
                    last_log_time = now

                done_futures = [f for f in futures if f.done()]
                for f in done_futures:
                    res = f.result()
                    total_trades += res
                    futures.remove(f)
    except KeyboardInterrupt:
        logger.info("Stopping simulation due to keyboard interrupt.")
    finally:
        producer.close()
        logger.info(f"Simulation ended. Total trades sent: {total_trades}")

def main():
    setup_logging()
    logger.info("Starting trade simulation...")
    simulate_trades(
        throughput=10,
        mode="db",
        batch_size=10,
        num_threads=Config.get_num_threads()
    )
    logger.info("Trade simulation completed.")

if __name__ == '__main__':
    main()
