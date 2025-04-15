import os
import pandas as pd
import requests
import datetime
import time

from data.cache import get_cache
from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
)

# Global cache instance
_cache = get_cache()


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API."""
    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # If not in cache or no data in range, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    price_response = PriceResponse(**response.json())
    prices = price_response.prices

    if not prices:
        return []

    # Cache the results as dicts
    _cache.set_prices(ticker, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Check cache first
    if cached_data := _cache.get_financial_metrics(ticker):
        # Filter cached data by date and limit
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

    # Parse response with Pydantic model
    metrics_response = FinancialMetricsResponse(**response.json())
    # Return the FinancialMetrics objects directly instead of converting to dict
    financial_metrics = metrics_response.financial_metrics

    if not financial_metrics:
        return []

    # Cache the results as dicts
    _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items from API."""
    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
    data = response.json()
    response_model = LineItemResponse(**data)
    search_results = response_model.search_results
    if not search_results:
        return []

    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                        if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                        and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = InsiderTradeResponse(**data)
        insider_trades = response_model.insider_trades
        
        if not insider_trades:
            break
            
        all_trades.extend(insider_trades)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break
            
        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results
    _cache.set_insider_trades(ticker, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Check cache first
    if cached_data := _cache.get_company_news(ticker):
        # Filter cached data by date range
        filtered_data = [CompanyNews(**news) for news in cached_data 
                        if (start_date is None or news["date"] >= start_date)
                        and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date
    
    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
        
        data = response.json()
        response_model = CompanyNewsResponse(**data)
        company_news = response_model.news
        
        if not company_news:
            break
            
        all_news.extend(company_news)
        
        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break
            
        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split('T')[0]
        
        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results
    _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
    return all_news


def get_economic_indicators(end_date: str) -> dict:
    """Fetch economic indicators from FRED API.
    
    This function retrieves key economic indicators such as GDP growth,
    inflation rate, interest rate, and unemployment rate from the Federal
    Reserve Economic Data (FRED) API.
    
    Args:
        end_date: The date for which to fetch economic indicators (YYYY-MM-DD)
        
    Returns:
        A dictionary containing economic indicators and their values
    """
    # Check cache first
    cache_key = f"economic_indicators_{end_date}"
    if cached_data := _cache.get(cache_key):
        return cached_data
    
    # FRED API key - in a real implementation, this would be stored in environment variables
    # For demonstration purposes, we'll use a placeholder
    api_key = os.environ.get("FRED_API_KEY", "YOUR_FRED_API_KEY")
    
    # Series IDs for the economic indicators we want to fetch
    series_ids = {
        "gdp_growth": "GDP",          # Gross Domestic Product
        "inflation_rate": "CPIAUCSL",  # Consumer Price Index for All Urban Consumers
        "interest_rate": "FEDFUNDS",   # Federal Funds Effective Rate
        "unemployment_rate": "UNRATE"  # Unemployment Rate
    }
    
    # Parse the end_date
    try:
        date_obj = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        # Convert to FRED API format (YYYY-MM-DD)
        observation_end = date_obj.strftime("%Y-%m-%d")
        
        # For GDP growth calculation, we need data from a year ago
        year_ago = date_obj - datetime.timedelta(days=365)
        observation_start = year_ago.strftime("%Y-%m-%d")
    except ValueError:
        # Default to current date if format is invalid
        current_date = datetime.datetime.now()
        observation_end = current_date.strftime("%Y-%m-%d")
        year_ago = current_date - datetime.timedelta(days=365)
        observation_start = year_ago.strftime("%Y-%m-%d")
    
    # Initialize results dictionary
    result = {
        "date": end_date,
        "indicators": {}
    }
    
    # If no API key is available, return simulated data
    if api_key == "YOUR_FRED_API_KEY":
        print("Warning: No FRED API key provided. Using simulated economic indicators.")
        return _get_simulated_economic_indicators(end_date)
    
    # Fetch data for each indicator
    for indicator_name, series_id in series_ids.items():
        try:
            # Base URL for FRED API
            base_url = "https://api.stlouisfed.org/fred/series/observations"
            
            # Parameters for the API request
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",  # Get most recent first
                "limit": 12,  # Get enough data for year-over-year calculations
                "observation_end": observation_end
            }
            
            # For GDP growth, we need more historical data
            if indicator_name == "gdp_growth":
                params["observation_start"] = observation_start
            
            # Make the API request
            response = requests.get(base_url, params=params)
            
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                observations = data.get("observations", [])
                
                if observations:
                    # Process the data based on the indicator
                    if indicator_name == "gdp_growth":
                        # Calculate year-over-year GDP growth
                        if len(observations) >= 2:
                            latest_gdp = float(observations[0]["value"])
                            year_ago_gdp = float(observations[-1]["value"])
                            gdp_growth = ((latest_gdp - year_ago_gdp) / year_ago_gdp) * 100
                            result["indicators"][indicator_name] = round(gdp_growth, 1)
                    elif indicator_name == "inflation_rate":
                        # Calculate year-over-year inflation rate
                        if len(observations) >= 12:  # Need at least 12 months of data
                            latest_cpi = float(observations[0]["value"])
                            year_ago_cpi = float(observations[11]["value"])
                            inflation_rate = ((latest_cpi - year_ago_cpi) / year_ago_cpi) * 100
                            result["indicators"][indicator_name] = round(inflation_rate, 1)
                    else:
                        # For other indicators, just use the latest value
                        result["indicators"][indicator_name] = round(float(observations[0]["value"]), 1)
            else:
                print(f"Error fetching {indicator_name} data: {response.status_code}")
                # If API request fails, use simulated data for this indicator
                simulated_data = _get_simulated_economic_indicators(end_date)
                result["indicators"][indicator_name] = simulated_data["indicators"].get(indicator_name)
                
            # Add a small delay to avoid hitting API rate limits
            time.sleep(0.5)
                
        except Exception as e:
            print(f"Error processing {indicator_name} data: {str(e)}")
            # If processing fails, use simulated data for this indicator
            simulated_data = _get_simulated_economic_indicators(end_date)
            result["indicators"][indicator_name] = simulated_data["indicators"].get(indicator_name)
    
    # Check if we have all the indicators
    if len(result["indicators"]) < len(series_ids):
        # Fill in any missing indicators with simulated data
        simulated_data = _get_simulated_economic_indicators(end_date)
        for indicator_name in series_ids.keys():
            if indicator_name not in result["indicators"]:
                result["indicators"][indicator_name] = simulated_data["indicators"].get(indicator_name)
    
    # Cache the result
    _cache.set(cache_key, result)
    
    return result


def _get_simulated_economic_indicators(end_date: str) -> dict:
    """Generate simulated economic indicators based on the date.
    
    This is a fallback function used when the FRED API is not available or fails.
    It generates realistic but simulated economic indicator values.
    
    Args:
        end_date: The date for which to generate indicators (YYYY-MM-DD)
        
    Returns:
        A dictionary containing simulated economic indicators
    """
    # Parse the end_date
    try:
        date_obj = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        # Default to current date if format is invalid
        date_obj = datetime.datetime.now()
    
    # Simulate different economic environments based on the year and month
    year = date_obj.year
    month = date_obj.month
    
    # Base values
    gdp_growth = 2.5  # Default moderate growth
    inflation_rate = 2.0  # Default target inflation
    interest_rate = 3.0  # Default moderate interest rate
    unemployment_rate = 4.5  # Default moderate unemployment
    
    # Adjust based on year (simulating economic cycles)
    year_mod = year % 10  # Create a 10-year cycle
    if year_mod < 3:  # Early cycle - growth phase
        gdp_growth += 1.5
        inflation_rate -= 0.5
        interest_rate -= 1.0
        unemployment_rate -= 1.0
    elif year_mod < 6:  # Mid cycle - stable growth
        gdp_growth += 0.5
        inflation_rate += 0.5
        interest_rate += 0.5
    elif year_mod < 8:  # Late cycle - slowing growth, rising inflation
        gdp_growth -= 0.5
        inflation_rate += 1.5
        interest_rate += 1.5
        unemployment_rate -= 0.5
    else:  # Recession/recovery phase
        gdp_growth -= 1.5
        inflation_rate -= 1.0
        interest_rate -= 0.5
        unemployment_rate += 2.0
    
    # Adjust based on month (simulating seasonal effects)
    if month in [1, 2, 12]:  # Winter months
        gdp_growth -= 0.3
    elif month in [4, 5, 6]:  # Spring months
        gdp_growth += 0.4
        unemployment_rate -= 0.2
    elif month in [7, 8, 9]:  # Summer months
        inflation_rate += 0.2
    
    # Ensure values are in reasonable ranges
    gdp_growth = max(-2.0, min(gdp_growth, 5.0))
    inflation_rate = max(0.0, min(inflation_rate, 8.0))
    interest_rate = max(0.25, min(interest_rate, 8.0))
    unemployment_rate = max(2.5, min(unemployment_rate, 10.0))
    
    # Create the result dictionary
    result = {
        "date": end_date,
        "indicators": {
            "gdp_growth": round(gdp_growth, 1),
            "inflation_rate": round(inflation_rate, 1),
            "interest_rate": round(interest_rate, 1),
            "unemployment_rate": round(unemployment_rate, 1),
        }
    }
    
    return result


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from the API."""
    financial_metrics = get_financial_metrics(ticker, end_date)
    market_cap = financial_metrics[0].market_cap
    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
