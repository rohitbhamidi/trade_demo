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
connection = engine.connect()

# =============================================================================
# Database Helper Function
# =============================================================================

def execute_query(query, params=None):
    """
    Execute a SQL query using a given connection. If an error occurs, attempt a rollback
    and log the error.
    """
    try:
        result = connection.execute(text(query), params or {})
        return result
    except SQLAlchemyError as e:
        try:
            connection.rollback()
        except Exception as rollback_error:
            logger.error("Error during rollback: %s", rollback_error)
        logger.error("Database query error: %s", e)
        raise

# =============================================================================
# Query Functions
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
    result = execute_query(query, {"date": date, "ticker": ticker})
    row = result.fetchone()
    total_volume = row[0] if row else None
    if isinstance(total_volume, Decimal):
        total_volume = float(total_volume)
    return total_volume

def get_market_cap(ticker):
    """
    Retrieve the latest price for the given ticker from live_trades and calculate
    the simulated market capitalization.
    """
    query = """
        SELECT price
        FROM live_trades
        WHERE ticker = :ticker
        ORDER BY localTS DESC
        LIMIT 1
    """
    result = execute_query(query, {"ticker": ticker})
    row = result.fetchone()
    if row is None or row[0] is None:
        return []
    latest_price = float(row[0])
    market_cap = latest_price * 1000000000
    return [market_cap]

def get_top_sectors():
    """
    Retrieve the top 5 sectors by aggregated market capitalization.
    A hard-coded ticker-to-sector mapping is used since the live_trades table does not
    include sector information.
    """
    mapping = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOG": "Technology",
        "AMZN": "Consumer",
        "TSLA": "Automotive"
    }
    query = "SELECT DISTINCT ticker FROM live_trades"
    result = execute_query(query)
    tickers = [row[0] for row in result.fetchall()]
    
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
    result = execute_query(query, {"start_date": start_date, "end_date": end_date})
    avg_volume = result.fetchone()[0]
    if isinstance(avg_volume, Decimal):
        avg_volume = float(avg_volume)
    return avg_volume

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
                    "Executes a SQL query to retrieve the total trading volume for a specific stock ticker on a specified date.\n"
                    "Parameters:\n"
                    " - date (str): The date for which to retrieve the total volume (YYYY-MM-DD).\n"
                    " - ticker (str): The stock ticker symbol.\n"
                    "Returns:\n"
                    " - total_volume (float): The total trading volume."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {
                            "type": "string",
                            "description": "Date in the format 'YYYY-MM-DD'"
                        },
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        }
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
                    "Retrieves the simulated market capitalization for a given ticker by obtaining its latest price.\n"
                    "Parameters:\n"
                    " - ticker (str): The stock ticker symbol.\n"
                    "Returns:\n"
                    " - market_cap (list): A list containing the simulated market cap."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        }
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
                    "Retrieves the top 5 sectors by total market capitalization.\n"
                    "Returns:\n"
                    " - top_sectors (list): A list of dictionaries with sector names and market caps."
                )
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_average_volume_per_transaction",
                "description": (
                    "Retrieves the average volume per transaction for a given date range.\n"
                    "Parameters:\n"
                    " - start_date (str): Start date in 'YYYY-MM-DD' format.\n"
                    " - end_date (str): End date in 'YYYY-MM-DD' format.\n"
                    "Returns:\n"
                    " - avg_volume_per_transaction (float): The average volume per transaction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)"
                        }
                    },
                    "required": ["start_date", "end_date"]
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
# Dash Application
# =============================================================================

server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/',
    external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
)

app.layout = html.Div(className='app', children=[
    html.Div(className='background-shapes'),
    html.H1("GenAI Driven Stock Data Analysis", className='title'),
    html.Div(
        id='graph-container',
        className='card',
        style={
            'width': '40%',
            'position': 'absolute',
            'left': '5%',
            'top': '50%',
            'transform': 'translateY(-50%)'
        },
        children=[
            dcc.Graph(id='live-update-graph', style={'height': '80vh'})
        ]
    ),
    html.Div(
        className='right-container',
        style={
            'width': '40%',
            'position': 'absolute',
            'right': '5%',
            'top': '50%',
            'transform': 'translateY(-50%)'
        },
        children=[
            dcc.Input(
                id='user-input',
                type='text',
                placeholder='Enter your question here...',
                className='input'
            ),
            html.Button('Submit', id='submit-val', n_clicks=0, className='button'),
            html.Div(id='response-area', className='response-area')
        ]
    ),
    dcc.Interval(id='interval-component', interval=1 * 1000, n_intervals=0)  # Update every second
])

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
        df = pd.read_sql_query(query, connection)
    except Exception as e:
        logger.error("Error querying live trades: %s", e)
        df = pd.DataFrame({'time': [], 'price': []})
    
    if df.empty:
        df = pd.DataFrame({'time': [], 'price': []})
    
    # Safely determine y-axis range
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
                line={'color': 'orange'},
                fill='tozeroy'
            )
        ],
        'layout': {
            'title': 'Real-Time Stock Data',
            'xaxis': {'title': 'Time', 'color': 'white'},
            'yaxis': {'title': 'Price', 'color': 'white', 'range': [y_min, y_max]},
            'plot_bgcolor': 'black',
            'paper_bgcolor': 'black',
            'font': {'color': 'white'},
        }
    }
    return figure

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
                    children=[html.P(response_message)],
                    style={'color': '#000000'}
                )
            else:
                return html.Div([
                    html.P("Query not recognized. Please try again.", style={'color': '#000000'})
                ])
        except Exception as e:
            logger.error("Error processing conversation: %s", e)
            return html.Div([
                html.P(f"An error occurred: {str(e)}", style={'color': '#FF4136'})
            ])
    return html.Div([
        html.P("Enter a question and press submit.", style={'color': '#000000'}),
        html.Br(),
        html.P("Sample 1: \"What was the total volume for AAPL on May 31, 2024?\"", style={'color': '#000000'}),
        html.Br(),
        html.P("Sample 2: \"What is the market cap for AAPL?\"", style={'color': '#000000'}),
        html.Br(),
        html.P("Sample 3: \"What are the top 5 sectors by market capitalization?\"", style={'color': '#000000'}),
        html.Br(),
        html.P("Sample 4: \"What is the average volume per transaction from May 1, 2024 to May 31, 2024?\"", style={'color': '#000000'}),
    ], style={'color': '#000000'})

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    server.run(debug=True, port=8051)
