# Trade Demo

This project simulates real-time stock trades and provides an analytics dashboard. It includes:

- A multi-threaded trade simulator that inserts synthetic trade data into SingleStoreDB.
- A Dash-based web UI for live market cap visualization and query input.
- OpenAI GPT-4o integration for natural language interaction with pre-defined SQL functions.

## Requirements

- Python 3.9+
- SingleStoreDB with a `live_trades` table
- OpenAI API key
- Required Python packages (see `requirements.txt`)

## Project Structure

```
trade_demo/
│
├── demo.py                         # Main Dash app and OpenAI logic
├── trades_data.csv                 # Input data for the simulator
├── .env                            # Configuration for DB, OpenAI, and simulator
├── assets/style.css                # Frontend CSS
│
└── tradeSimulator/
    ├── simulator.py                # Main loop for sending trades
    ├── config.py                   # Reads .env config values
    ├── db_handler.py               # Handles batched inserts to SingleStore
    ├── producer.py                 # Provides DB producer
    ├── utils.py                    # Rate limiter
    └── logger_config.py            # Logging setup
```

## Database Schema

The system expects a `live_trades` table with the following schema:

```sql
CREATE TABLE live_trades (
    trade_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    localTS DATETIME,
    localDate DATE,
    ticker VARCHAR(16),
    conditions TEXT,
    correction TINYINT,
    exchange TINYINT,
    id BIGINT,
    participant_timestamp BIGINT,
    price DOUBLE,
    sequence_number BIGINT,
    sip_timestamp BIGINT,
    size BIGINT,
    tape TINYINT,
    trf_id BIGINT,
    trf_timestamp BIGINT
);
```

Note:
- `trade_id` is auto-incremented and not included in the simulator insert statement.
- Timestamps are in nanoseconds where applicable.

## Setup

1. Clone the repository.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file:

   ```env
   LOG_LEVEL=INFO

   # Database
   SINGLESTORE_DB_URL=mysql+pymysql://<user>:<password>@<host>:<port>/<db>?ssl=true

   # OpenAI
   OPENAI_API_KEY=sk-...

   # Simulator
   THROUGHPUT=10
   MODE=db
   NUM_THREADS=8
   BATCH_SIZE=10
   LOCAL_CSV_PATH=./trades_data.csv
   LOG_INTERVAL=5
   ```

## Running the System

1. Start the simulator:

   ```bash
   python -m tradeSimulator.simulator
   ```

2. Run the Dash web app:

   ```bash
   python demo.py
   ```

3. Open [http://localhost:8051](http://localhost:8051) in a browser.

## Available Queries

The frontend supports questions like:

- “What is the market cap for AAPL?”
- “What are the top 5 sectors by market capitalization?”
- “Total volume for AAPL on 2018-01-02”
- “Average volume per transaction from 2018-01-01 to 2018-01-05”

## Notes

- Trade data is simulated using the `trades_data.csv` file.
- Market cap is calculated as `price * 1,000,000,000`.
- Sector mapping is hard-coded for select tickers.
- OpenAI tool calls map to specific SQL functions defined in `demo.py`.