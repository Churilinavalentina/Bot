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

# TODO: чтение из паркета теперь
class MyParquetDataset(Dataset):
    def __init__(self, delta_t, delta_t_future, k, start_date, end_date):
        self.delta_t = delta_t
        self.delta_t_future = delta_t_future
        self.k = k

        folder_path = Path(f"{FOLDER}market_data_{start_date.date()}-{end_date.date()}")
        self.files = list(folder_path.rglob('*.parquet'))

        # Используем ThreadPoolExecutor для параллельной обработки файлов
        # max_workers можно поставить 4-8 в зависимости от процессора
        all_results = []
        with ProcessPoolExecutor (max_workers=NUM_EXECUTOR_READ) as executor:
            all_results = list(tqdm(executor.map(self._process_single_file, self.files)))

        # Собираем результаты (фильтруем None, если файлы были пустые)
        # Собираем все результаты
        all_x = [res[0] for res in all_results if res is not None]
        all_y = [res[1] for res in all_results if res is not None]

        # Конкатенируем уже НАРЕЗАННЫЕ окна
        # Теперь стыки файлов физически не могут перемешаться
        self.X = torch.cat(all_x, dim=0) # Станет [Total_Windows, Delta_T]
        self.y = torch.cat(all_y, dim=0) # Станет [Total_Windows, 1]

        print(f'====> Итоговый размер: X={self.X.shape}, y={self.y.shape}')

    def _process_single_file(self, file_path):
        """Логика обработки одного файла (вынесена для многопоточности)"""
        try:
            # 1. Читаем всё как строки, чтобы быстро обработать 'open'
            df = pd.read_parquet(file_path, columns=['time', 'open'])
            
            # 2. Быстрый фильтр времени
            #df = df[df['time'].str[11:13] != '23'].copy()
            if df.empty: return None

            # 3. Векторизованный парсинг цены
            # nums = df['open'].str.extract(r"(\d+).+?(\d+)").astype(float)
            # df['open'] = nums[0] + nums[1] / 1e9

            # 4. Расчеты в Pandas (уже оптимизированы)
            df['normal_price'] = df['open'].pct_change() * 100
            future_max = df['open'].rolling(window=self.delta_t_future).max().shift(-self.delta_t_future)
            df['predict_bool'] = ((future_max - df['open']) / df['open'] * 100 > self.k).astype(float)
            
            df = df.dropna()
            
            x_values = torch.tensor(df['normal_price'].values, dtype=torch.float32)
            y_values = torch.tensor(df['predict_bool'].values, dtype=torch.float32).reshape(-1, 1)

            # 2. Нарезаем на окна прямо здесь через unfold
            # Это создает тензор формы [количество_окон, delta_t]
            # unfold(измерение, размер_окна, шаг)
            if len(x_values) < self.delta_t: return None
            
            x_windows = x_values.unfold(0, self.delta_t, 1) 
            
            # 3. Таргеты берем со смещением (соответствующий концу окна)
            y_targets = y_values[self.delta_t - 1:] 
            
            # Теперь у x_windows и y_targets одинаковая первая размерность
            return x_windows, y_targets

        except:
            return None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def split_data(dataset, train_part=0.8):
    len_ds = len(dataset)
    train_size = int(train_part * len_ds)

    train_ds = copy(dataset)
    train_ds.X, train_ds.y = train_ds.X[:train_size], train_ds.y[:train_size]

    test_ds = copy(dataset)
    test_ds.X, test_ds.y = test_ds.X[train_size:], test_ds.y[train_size:]

    return train_ds, test_ds

def check_balance(ds):
    print(f'Num posivives: {(ds.y > 0).sum()} / negatives: {(ds.y == 0).sum()}')

