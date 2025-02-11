import logging
import sys
from tradeSimulator.config import Config

def setup_logging():
    log_level = Config.get_log_level().upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout
    )