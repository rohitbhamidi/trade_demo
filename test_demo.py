import os
import json
import logging
from decimal import Decimal
from getpass import getpass

from dotenv import load_dotenv
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import singlestoredb as s2  # kept for compatibility if needed

from openai import OpenAI

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from flask import Flask
import plotly.graph_objs as go

# =============================================================================
# Configuration and Logging
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Retrieve OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = getpass("Enter OpenAI API key: ")
client = OpenAI(api_key=api_key)

# Retrieve database connection URL
connection_url = os.getenv("SINGLESTORE_DB_URL")
if not connection_url:
    raise ValueError("SINGLESTORE_DB_URL not set in .env")
engine = create_engine(connection_url)
# Note: We no longer use a global connection; instead, each query opens a new connection.

# =============================================================================
# Database Helper Function
# =============================================================================

def execute_query(query, params=None):
    """
    Execute a SQL query using a new connection (ensuring that any previous
    transaction state does not carry over). If an error occurs, it is logged.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            # Fetch all rows before closing the connection.
            return result.fetchall()
    except SQLAlchemyError as e:
        logger.error("Database query error: %s", e)
        raise

# =============================================================================
# Query Functions (Using the Provided Schema)
# =============================================================================

def get_total_volume(date, ticker):
    """
    Retrieve the total trading volume for a given ticker on a specified date.
    """
    query = """
        SELECT ROUND(SUM(size), 0) AS "Total Volume"
        FROM live_trades
        WHERE localDate = :date
          AND ticker = :ticker
    """
    rows = execute_query(query, {"date": date, "ticker": ticker})
    row = rows[0] if rows else None
    total_volume = row[0] if row else None
    if isinstance(total_volume, Decimal):
        total_volume = float(total_volume)
    return total_volume

def get_market_cap(ticker):
    """
    Retrieve the latest price for the given ticker and calculate a simulated market capitalization.
    """
    query = """
        SELECT price
        FROM live_trades
        WHERE ticker = :ticker
        ORDER BY localTS DESC
        LIMIT 1
    """
    rows = execute_query(query, {"ticker": ticker})
    row = rows[0] if rows else None
    if row is None or row[0] is None:
        return []
    latest_price = float(row[0])
    market_cap = latest_price * 1000000000  # Simulated market cap
    return [market_cap]

def get_top_sectors():
    """
    Retrieve the top 5 sectors by aggregated market capitalization.
    Uses a hard-coded ticker-to-sector mapping.
    """
    mapping = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "TSLA": "Automotive",
        "NVDA": "Semiconductors",
        "AMZN": "Consumer Discretionary"
    }
    query = "SELECT DISTINCT ticker FROM live_trades"
    rows = execute_query(query)
    tickers = [row[0] for row in rows]
    
    sectors = {}
    for t in tickers:
        caps = get_market_cap(t)
        cap = caps[0] if caps else 0
        sector = mapping.get(t, "Unknown")
        sectors[sector] = sectors.get(sector, 0) + cap

    sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:5]
    top_sectors = [{'sector': sec, 'market_cap': cap} for sec, cap in sorted_sectors]
    return top_sectors

def get_average_volume_per_transaction(start_date, end_date):
    """
    Retrieve the average volume per transaction over a specified date range.
    """
    query = """
        SELECT AVG(size) AS "Average Volume per Transaction"
        FROM live_trades
        WHERE localDate BETWEEN :start_date AND :end_date
    """
    rows = execute_query(query, {"start_date": start_date, "end_date": end_date})
    avg_volume = rows[0][0] if rows else None
    if isinstance(avg_volume, Decimal):
        avg_volume = float(avg_volume)
    return avg_volume

# --- New Functions for Additional Financial Insights ---

def get_max_trade_size(date, ticker):
    """
    Retrieve the maximum trade size for a given ticker on a specified date.
    """
    query = """
        SELECT MAX(size) AS "Max Trade Size"
        FROM live_trades
        WHERE localDate = :date
          AND ticker = :ticker
    """
    rows = execute_query(query, {"date": date, "ticker": ticker})
    row = rows[0] if rows else None
    return row[0] if row else None

def get_average_trade_price(date, ticker):
    """
    Retrieve the average trade price for a given ticker on a specified date.
    """
    query = """
        SELECT AVG(price) AS "Average Trade Price"
        FROM live_trades
        WHERE localDate = :date
          AND ticker = :ticker
    """
    rows = execute_query(query, {"date": date, "ticker": ticker})
    row = rows[0] if rows else None
    avg_price = row[0] if row else None
    if isinstance(avg_price, Decimal):
        avg_price = float(avg_price)
    return avg_price

def get_trade_count(date, ticker):
    """
    Retrieve the total number of trades for a given ticker on a specified date.
    """
    query = """
        SELECT COUNT(*) AS "Trade Count"
        FROM live_trades
        WHERE localDate = :date
          AND ticker = :ticker
    """
    rows = execute_query(query, {"date": date, "ticker": ticker})
    row = rows[0] if rows else None
    return row[0] if row else 0

# =============================================================================
# Conversation Function (Chat with OpenAI)
# =============================================================================

def run_conversation(question):
    """
    Send a question and available functions to the OpenAI chat model and process any
    function calls returned by the model.
    """
    messages = [{"role": "user", "content": question}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_total_volume",
                "description": (
                    "Retrieve the total trading volume for a given stock ticker on a specified date.\n"
                    "Parameters:\n"
                    " - date (str): The date (YYYY-MM-DD) for which to retrieve the volume.\n"
                    " - ticker (str): The stock ticker symbol."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "Date in the format 'YYYY-MM-DD'"},
                        "ticker": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["date", "ticker"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_market_cap",
                "description": (
                    "Retrieve the simulated market capitalization for a given ticker using its latest price.\n"
                    "Parameters:\n"
                    " - ticker (str): The stock ticker symbol."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["ticker"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_top_sectors",
                "description": (
                    "Retrieve the top 5 sectors by total market capitalization. Returns a list of sectors and their aggregated market caps."
                )
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_average_volume_per_transaction",
                "description": (
                    "Retrieve the average volume per transaction over a specified date range.\n"
                    "Parameters:\n"
                    " - start_date (str): Start date in 'YYYY-MM-DD' format.\n"
                    " - end_date (str): End date in 'YYYY-MM-DD' format."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                        "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                    },
                    "required": ["start_date", "end_date"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_max_trade_size",
                "description": (
                    "Retrieve the maximum trade size for a given stock ticker on a specified date.\n"
                    "Parameters:\n"
                    " - date (str): The date (YYYY-MM-DD).\n"
                    " - ticker (str): The stock ticker symbol."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "Date in the format 'YYYY-MM-DD'"},
                        "ticker": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["date", "ticker"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_average_trade_price",
                "description": (
                    "Retrieve the average trade price for a given stock ticker on a specified date.\n"
                    "Parameters:\n"
                    " - date (str): The date (YYYY-MM-DD).\n"
                    " - ticker (str): The stock ticker symbol."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "Date in the format 'YYYY-MM-DD'"},
                        "ticker": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["date", "ticker"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_trade_count",
                "description": (
                    "Retrieve the total number of trades for a given stock ticker on a specified date.\n"
                    "Parameters:\n"
                    " - date (str): The date (YYYY-MM-DD).\n"
                    " - ticker (str): The stock ticker symbol."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "Date in the format 'YYYY-MM-DD'"},
                        "ticker": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["date", "ticker"]
                }
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"  # explicitly set to auto
        )
    except Exception as e:
        logger.error("Error during OpenAI chat completion: %s", e)
        raise

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        available_functions = {
            "get_total_volume": get_total_volume,
            "get_market_cap": get_market_cap,
            "get_top_sectors": get_top_sectors,
            "get_average_volume_per_transaction": get_average_volume_per_transaction,
            "get_max_trade_size": get_max_trade_size,
            "get_average_trade_price": get_average_trade_price,
            "get_trade_count": get_trade_count,
        }
        messages.append(response_message)  # add model's initial reply

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions.get(function_name)
            if not function_to_call:
                logger.error("Requested function '%s' is not available.", function_name)
                continue

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logger.error("Error decoding function arguments: %s", e)
                continue

            try:
                if function_name == "get_total_volume":
                    result_value = function_to_call(
                        date=function_args.get("date"), ticker=function_args.get("ticker")
                    )
                elif function_name == "get_market_cap":
                    result_value = function_to_call(
                        ticker=function_args.get("ticker")
                    )
                elif function_name == "get_top_sectors":
                    result_value = function_to_call()
                elif function_name == "get_average_volume_per_transaction":
                    result_value = function_to_call(
                        start_date=function_args.get("start_date"),
                        end_date=function_args.get("end_date")
                    )
                elif function_name == "get_max_trade_size":
                    result_value = function_to_call(
                        date=function_args.get("date"),
                        ticker=function_args.get("ticker")
                    )
                elif function_name == "get_average_trade_price":
                    result_value = function_to_call(
                        date=function_args.get("date"),
                        ticker=function_args.get("ticker")
                    )
                elif function_name == "get_trade_count":
                    result_value = function_to_call(
                        date=function_args.get("date"),
                        ticker=function_args.get("ticker")
                    )
                else:
                    result_value = None
            except Exception as e:
                logger.error("Error executing function '%s': %s", function_name, e)
                result_value = f"Error executing function: {e}"

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(result_value),
            })

        try:
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            return second_response
        except Exception as e:
            logger.error("Error during second OpenAI chat completion: %s", e)
            raise

    return response

# =============================================================================
# Dash Application with a Modern, Responsive Layout
# =============================================================================

server = Flask(__name__)
# Use a Bootswatch Darkly theme for a modern dark look
external_stylesheets = [
    "https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/darkly/bootstrap.min.css"
]
app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/',
    external_stylesheets=external_stylesheets
)

app.layout = html.Div([
    # Navigation Bar
    html.Nav(className="navbar navbar-expand-lg navbar-dark bg-primary", children=[
        html.Div("GenAI Driven Stock Data Analysis", className="navbar-brand")
    ]),
    # Main container
    html.Div(className="container mt-4", children=[
        html.Div(className="row", children=[
            # Left Column: Real-Time Graph
            html.Div(className="col-lg-8 mb-4", children=[
                html.Div(className="card shadow", children=[
                    html.Div(className="card-header", children=[
                        html.H5("Real-Time Stock Data", className="card-title mb-0")
                    ]),
                    html.Div(className="card-body", children=[
                        dcc.Graph(
                            id='live-update-graph',
                            style={'height': '60vh'}
                        )
                    ])
                ])
            ]),
            # Right Column: Chat/Query Input
            html.Div(className="col-lg-4", children=[
                html.Div(className="card shadow", children=[
                    html.Div(className="card-header", children=[
                        html.H5("Ask a Financial Question", className="card-title mb-0")
                    ]),
                    html.Div(className="card-body", children=[
                        dcc.Input(
                            id='user-input',
                            type='text',
                            placeholder='Enter your question here...',
                            className='form-control mb-2'
                        ),
                        html.Button(
                            'Submit',
                            id='submit-val',
                            n_clicks=0,
                            className='btn btn-primary mb-3'
                        ),
                        html.Div(id='response-area', className='mt-2')
                    ])
                ])
            ])
        ])
    ]),
    # Interval for live graph updates
    dcc.Interval(id='interval-component', interval=1 * 1000, n_intervals=0)
])

# -----------------------------------------------------------------------------
# Callback for Live Updating the Stock Graph
# -----------------------------------------------------------------------------

@app.callback(
    Output('live-update-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph_live(n):
    query = """
        SELECT localTS as time, price
        FROM live_trades
        WHERE localTS IS NOT NULL
        ORDER BY localTS DESC
        LIMIT 100
    """
    try:
        # Use the engine directly; pd.read_sql_query accepts an Engine.
        df = pd.read_sql_query(query, engine)
    except Exception as e:
        logger.error("Error querying live trades: %s", e)
        df = pd.DataFrame({'time': [], 'price': []})
    
    if df.empty:
        df = pd.DataFrame({'time': [], 'price': []})
    
    # Determine a safe y-axis range
    if not df.empty:
        y_min = df['price'].min() - 5
        y_max = df['price'].max() + 5
    else:
        y_min, y_max = 0, 100

    figure = {
        'data': [
            go.Scatter(
                x=df['time'],
                y=df['price'],
                name='Stock Data',
                mode='lines+markers',
                line={'color': 'cyan'},
                fill='tozeroy'
            )
        ],
        'layout': {
            'title': 'Real-Time Stock Data',
            'xaxis': {'title': 'Time', 'color': '#ffffff'},
            'yaxis': {'title': 'Price', 'color': '#ffffff', 'range': [y_min, y_max]},
            'plot_bgcolor': '#2c3e50',
            'paper_bgcolor': '#2c3e50',
            'font': {'color': '#ffffff'},
        }
    }
    return figure

# -----------------------------------------------------------------------------
# Callback for Processing User Questions
# -----------------------------------------------------------------------------

@app.callback(
    Output('response-area', 'children'),
    [Input('submit-val', 'n_clicks')],
    [State('user-input', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0 and value:
        try:
            response = run_conversation(value)
            if response and response.choices:
                response_message = response.choices[0].message.content
                return html.Div(
                    className='response-container',
                    children=[html.P(response_message, className="mb-0")],
                    style={'color': '#ffffff'}
                )
            else:
                return html.Div([
                    html.P("Query not recognized. Please try again.", style={'color': '#ffffff'})
                ])
        except Exception as e:
            logger.error("Error processing conversation: %s", e)
            return html.Div([
                html.P(f"An error occurred: {str(e)}", style={'color': '#FF4136'})
            ])
    # Initial instructions with sample queries based on the actual data
    return html.Div([
        html.P("Enter a financial question and press submit.", style={'color': '#ffffff'}),
        html.Br(),
        html.P("Sample 1: \"What was the total volume for AAPL on 2025-02-10?\"", style={'color': '#ffffff'}),
        html.Br(),
        html.P("Sample 2: \"What is the market cap for TSLA?\"", style={'color': '#ffffff'}),
        html.Br(),
        html.P("Sample 3: \"What is the average volume per transaction from 2025-02-10 to 2025-02-10?\"", style={'color': '#ffffff'}),
        html.Br(),
        html.P("Sample 4: \"What was the maximum trade size for NVDA on 2025-02-10?\"", style={'color': '#ffffff'}),
        html.Br(),
        html.P("Sample 5: \"What was the average trade price for MSFT on 2025-02-10?\"", style={'color': '#ffffff'}),
        html.Br(),
        html.P("Sample 6: \"How many trades were executed for AMZN on 2025-02-10?\"", style={'color': '#ffffff'}),
    ])

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    server.run(debug=True, port=8051)
