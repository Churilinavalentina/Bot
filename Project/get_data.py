import pandas as pd
import ast
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
from datetime import timedelta, datetime, timezone
from pathlib import Path

from pandas import DataFrame
from t_tech.invest import Client, SecurityTradingStatus,CandleInterval, AsyncClient
from t_tech.invest.services import InstrumentsService
from t_tech.invest.utils import quotation_to_decimal, now
from t_tech.invest.caching.market_data_cache.cache import MarketDataCache
from t_tech.invest.caching.market_data_cache.cache_settings import (
    MarketDataCacheSettings,
)
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import time
from matplotlib import pyplot as plt
from copy import copy
from tqdm.autonotebook import tqdm
from concurrent.futures import ThreadPoolExecutor

TOKEN = 't.2kt-ltTDgFp83htFE9vFk5K0Gtb4Wwnf-kfGH4RkSDxfxWf9BS5Hkmp8pMIxA0gs7PHOsCiQ7IyB7BjvEpOxwg'
#Параметры выгрузки:

interval = CandleInterval.CANDLE_INTERVAL_1_MIN
start_date = datetime(2025, 11, 24, tzinfo=timezone.utc)
end_date = datetime(2026, 2, 24, tzinfo=timezone.utc)

# 100 самых ликвидных
ticker = [
    "AFKS", "AFLT", "AKRN", "ALRS", "APTK", "AQUA", "ASTR", "BANEP", "BELU", "BSPB",
    "CBOM", "CHMF", "CNRU", "DATA", "DOMRF", "ELFV", "ENPG", "ETLN", "EUTR",
    "FEES", "FESH", "FIXR", "FLOT", "GAZP", "GEMC", "GMKN", "HEAD", "HNFG", "HYDR",
    "IRAO", "LEAS", "LENT", "LKOH", "LSNGP", "LSRG", "MAGN", "MBNK", "MDMG", "MGNT",
    "MGTSP", "MOEX", "MRKC", "MRKP", "MRKU", "MRKV", "MSNG", "MSRS", "MTLR", "MTLRP", "MTSS",
    "NKHP", "NKNC", "NKNCP", "NLMK", "NMTP", "NVTK", "OGKB", "OZON", "OZPH", "PHOR", "PIKK",
    "PLZL", "POSI", "PRMD", "RAGR", "RENI", "RNFT", "ROSN", "RTKM",
    "RTKMP", "RUAL", "SBER", "SBERP", "SELG", "SFIN", "SGZH", "SMLT", "SNGS", "SNGSP", "SOFL",
    "SPBE", "SVAV", "SVCB", "T", "TATN", "TATNP", "TGKA", "TRMK", "TRNFP", "UGLD", "UPRO",
    "UWGN", "VKCO", "VSEH", "VSMO", "VTBR", "WUSH", "X5", "YDEX"
]


def get_figi():
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
        settings = MarketDataCacheSettings(base_cache_dir=Path(f"market_data_{start_date.date()}-{end_date.date()}"))
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

get_values(get_figi().figi, interval, start_date, end_date, num_workers=1)