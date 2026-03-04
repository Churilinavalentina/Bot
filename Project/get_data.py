import os
from dotenv import load_dotenv
from pathlib import Path

from pandas import DataFrame
from t_tech.invest import Client
from t_tech.invest.services import InstrumentsService
from t_tech.invest.utils import quotation_to_decimal, now
from t_tech.invest.caching.market_data_cache.cache import MarketDataCache
from t_tech.invest.caching.market_data_cache.cache_settings import (
    MarketDataCacheSettings,
)
from params import INTERVAL, START_DATE, END_DATE, TICKERS, FOLDER
from tqdm.autonotebook import tqdm
from concurrent.futures import ThreadPoolExecutor

load_dotenv()
TOKEN = os.environ["INVEST_TOKEN"]
#Параметры выгрузки:

def get_figi(ticker):
    """Example - How to get figi by name of ticker."""

    with Client(TOKEN) as client:
        instruments: InstrumentsService = client.instruments
        tickers = []
        for method in ["shares"]:  #, "bonds", "etfs", "currencies", "futures"]:
            for item in getattr(instruments, method)().instruments:
                for tic in ticker:
                    if tic == item.ticker:
                        tickers.append(
                            {
                                "name": item.name,
                                "ticker": item.ticker,
                                "figi": item.figi,
                            }
                        )
                        
        tickers_df = DataFrame(tickers)
        tickers_df = tickers_df[tickers_df["figi"] != "BBG000BSJK37"]
        ticker_df = tickers_df[["name", "figi"]]

        return ticker_df
    
def get_values(figi, interval, start_date, end_date, num_workers=5):
    with Client(TOKEN) as client:
        folder_path = Path(f"{FOLDER}market_data_{start_date.date()}-{end_date.date()}")
        settings = MarketDataCacheSettings(base_cache_dir=folder_path)
        market_data_cache = MarketDataCache(settings=settings, services=client)

        def cache_candles(figi_item):
          for _ in market_data_cache.get_all_candles(
            figi=figi_item,
            from_=start_date,
            to=end_date,
            interval=interval,
            #interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
          ):
            pass

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
          _ = list(tqdm(executor.map(cache_candles, figi), total=len(figi)))

    return 0

#get_figi()

get_values(get_figi(TICKERS).figi, INTERVAL, START_DATE, END_DATE, num_workers=1)