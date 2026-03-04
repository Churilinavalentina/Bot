import pandas as pd
import torch
from torch.utils.data import Dataset,  random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from params import IS_COLLAB, FOLDER

from tqdm.autonotebook import tqdm
from copy import copy
import numpy as np

from pathlib import Path
from params import START_DATE, END_DATE, DELTA_T, DELTA_T_FUTURE, K, NUM_EXECUTOR_READ

from concurrent.futures import ThreadPoolExecutor

class MyCSVDataset(Dataset):
    def __init__(self, delta_t, delta_t_future, k, start_date, end_date):
        self.delta_t = delta_t
        self.delta_t_future = delta_t_future
        self.k = k

        if IS_COLLAB:
            folder_path = Path(f"{FOLDER}/market_data_{start_date.date()}-{end_date.date()}")
        else: 
            folder_path = Path(f"market_data_{start_date.date()}-{end_date.date()}")
        self.files = list(folder_path.rglob('*.csv'))

        # Используем ThreadPoolExecutor для параллельной обработки файлов
        # max_workers можно поставить 4-8 в зависимости от процессора
        all_results = []
        with ThreadPoolExecutor(max_workers=NUM_EXECUTOR_READ) as executor:
            all_results = list(tqdm(executor.map(self._process_single_file, self.files)))

        # Собираем результаты (фильтруем None, если файлы были пустые)
        all_x = [res[0] for res in all_results if res is not None]
        all_y = [res[1] for res in all_results if res is not None]
        all_z = [res[2] for res in all_results if res is not None]

        print(len(all_x))

        if all_x:
            self.X = torch.cat(all_x, dim=0)
            self.y = torch.cat(all_y, dim=0)
            self.z = torch.cat(all_z, dim=0)
        else:
            self.X, self.y= torch.empty(0), torch.empty(0)

        print(f'====> Итоговый размер: X={self.X.shape}, y={self.y.shape}')

    def _process_single_file(self, file_path):
        """Логика обработки одного файла (вынесена для многопоточности)"""
        try:
            df = pd.read_csv(file_path, usecols=['time', 'open'])

            # Ускоренный парсинг времени (берем часы срезом строки, если формат фиксирован)
            # Если формат '2023-01-01 23:00:00', то часы это символы [11:13]
            hours = df['time'].str[11:13]
            df = df[hours != '23'].copy()

            if df.empty: return None

            # Ускоренный парсинг 'open' БЕЗ ast.literal_eval
            # Строка выглядит как "{'units': 100, 'nano': 500000000}"
            # Вытаскиваем числа простым строковым методом или regex
            def fast_parse_open(val):
                # Находим units и nano через простые правила (быстрее чем ast)
                parts = val.replace('{', '').replace('}', '').replace("'", "").split(',')
                d = {p.split(':')[0].strip(): int(p.split(':')[1]) for p in parts}
                return d['units'] + d['nano'] / 1e9

            df['open'] = df['open'].apply(fast_parse_open)
            df['normal_price'] = (df['open']-df['open'].shift(1)) / df['open'].shift(1)*100
            # Отрезаем края, где normal_price дал NaN
            df = df.dropna(subset=['normal_price'])

            # Расчет таргета (векторизован в Pandas, это быстро)
            future_max = df['open'].rolling(window=self.delta_t_future + 1).max().shift(-self.delta_t_future)
            df['predict_bool'] = (future_max - df['open']) / df['open'] * 100 > self.k

            # Отрезаем края, где rolling дал NaN
            df = df.dropna(subset=['predict_bool'])

            if df.empty: return None

            x_tensor = torch.tensor(df['normal_price'].values, dtype=torch.float32)
            y_tensor = torch.tensor(df['predict_bool'].values, dtype=torch.float32).reshape(-1, 1)

            first_delta_t = np.zeros(len(x_tensor))
            first_delta_t[:DELTA_T] = 1
            z_tenzor = torch.tensor(first_delta_t, dtype=torch.float32)

            return x_tensor, y_tensor, z_tenzor

        except Exception as e:
            print(f"Ошибка в файле {file_path}: {e}")
            return None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
      for i in range(DELTA_T+1):
        if self.z[idx] == 0:
          return self.X[idx-DELTA_T:idx], self.y[idx]
        else:
          idx = idx + 1


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

