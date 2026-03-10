import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from pandas import DataFrame
from datetime import datetime, timezone
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
    
def get_values(figi_list, interval, start_date, end_date, num_workers=1):
    # Создаем папку заранее
    folder_path = Path(f"{FOLDER}market_data_{start_date.date()}-{end_date.date()}")
    folder_path.mkdir(parents=True, exist_ok=True)

    def download_one_figi(figi_item):
        # Список должен быть внутри функции для каждого потока свой!
        local_candles = [] 
        
        # Используем НОВЫЙ клиент в каждом потоке (для стабильности ThreadPool)
        with Client(TOKEN) as client:
            try:
                # get_all_candles сам умеет разбивать запрос на куски по времени
                for candle in client.get_all_candles(
                    figi=figi_item,
                    from_=start_date,
                    to=end_date,
                    interval=interval
                ):
                    local_candles.append({
                        "figi":figi_item,
                        "time": candle.time,
                        "open": candle.open.units + candle.open.nano/1e9,
                        "close": candle.close.units + candle.close.nano/1e9,
                        "high": candle.high.units + candle.high.nano/1e9,
                        "low": candle.low.units + candle.low.nano/1e9,
                        "volume": candle.volume
                    })
                
                if local_candles:
                    df = pd.DataFrame(local_candles)
                    # Сохраняем в паркет
                    df.to_parquet(folder_path / f"{figi_item}.parquet", engine='pyarrow')
                    return f"{figi_item}: OK ({len(df)} rows)"
                else:
                    return f"{figi_item}: No data"
            except Exception as e:
                return f"{figi_item}: Error {e}"

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(download_one_figi, figi_list), total=len(figi_list)))
        for res in results:
            print(res)

    return 0

#get_figi()

#get_values(get_figi(TICKERS).figi, INTERVAL, START_DATE, END_DATE, num_workers=1)
#figi = get_figi(["T"]).figi
#get_values(figi, INTERVAL, datetime(2026, 3, 4, tzinfo=timezone.utc), datetime(2026, 3, 5, tzinfo=timezone.utc), num_workers=1)
