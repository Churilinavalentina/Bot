import pandas as pd
import torch
from torch.utils.data import Dataset,  random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from params import FOLDER

from tqdm.autonotebook import tqdm
from copy import copy
import numpy as np

from pathlib import Path
from params import START_DATE, END_DATE, DELTA_T, DELTA_T_FUTURE, K, NUM_EXECUTOR_READ

from concurrent.futures import ProcessPoolExecutor 

class MyParquetBotDataset(Dataset):
    def __init__(self, delta_t, delta_t_future, k, start_date, end_date):
        self.delta_t = delta_t
        # Словарь для быстрого поиска срезов по тикеру
        self.figi_map = {} 

        folder_path = Path(f"{FOLDER}market_data_{start_date.date()}-{end_date.date()}")
        self.files = list(folder_path.rglob('*.parquet'))

        all_results = []
        with ProcessPoolExecutor(max_workers=NUM_EXECUTOR_READ) as executor:
            all_results = list(tqdm(executor.map(self._process_single_file, self.files), total=len(self.files)))

        current_idx = 0
        final_x, final_y, final_prices = [], [], []

        for res in all_results:
            if res is None: continue
            
            x_win, y_tar, p_match, figi_name = res
            num_windows = x_win.shape[0]

            # Запоминаем границы для get_data_by_figi
            self.figi_map[figi_name] = (current_idx, current_idx + num_windows)
            
            final_x.append(x_win)
            final_y.append(y_tar)
            final_prices.append(p_match)
            current_idx += num_windows

        self.X = torch.cat(final_x, dim=0)
        self.y = torch.cat(final_y, dim=0)
        self.prices = torch.cat(final_prices, dim=0)

    def _process_single_file(self, file_path):
        try:
            figi = file_path.stem 
            df = pd.read_parquet(file_path, columns=['open'])
            df['normal_price'] = df['open'].pct_change() * 100
            df = df.dropna().reset_index(drop=True)
            
            if len(df) < self.delta_t: return None

            x_windows = torch.tensor(df['normal_price'].values, dtype=torch.float32).unfold(0, self.delta_t, 1)
            matched_prices = torch.tensor(df['open'].values, dtype=torch.float32)[self.delta_t - 1:]
            y_targets = torch.zeros(len(x_windows), dtype=torch.float32)
            
            return x_windows, y_targets, matched_prices, figi
        except:
            return None

    def get_figi_by_index(self, idx):
        """Находит FIGI, в диапазон которого попадает индекс"""
        for figi, (start, end) in self.figi_map.items():
            if start <= idx < end:
                return figi
        return None

    def get_data_by_figi(self, figi):
        """Быстрый доступ к данным конкретной акции"""
        if figi not in self.figi_map: return None, None
        start, end = self.figi_map[figi]
        return self.X[start:end], self.prices[start:end]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)
