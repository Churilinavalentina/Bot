import pandas as pd
import ast
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import time

# Задача
"""
Даны катировки актива за предыщие delta_t шагов (разница по времени между шагами в наших данных - 1 минута)
Мы хотим предсказать, что в следующие delta_t_future шагов цена получит относительный прирост на k процентов (это 1),
если условие не выполняется то 0.

Данные:
 csv: для одной акции промежуто за 60 дней (используем цену открытия и время)
 
"""

path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/BBG004730N88/CANDLE_INTERVAL_1_MIN-1765185840-1770369780.csv' # Сбер
epochs = 25
batch_size = 3000
delta_t = 100
k = 0.05


class MyParquetDataset(Dataset):
    def __init__(self, df, delta_t, k):

        # 2. Преобразование времени в формат datetime
        df['second'] = pd.to_datetime(df['time']).dt.time.apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
        df['normal_second_sin'] = np.sin(2 * np.pi * df['second'] / 86400)
        df['normal_second_cos'] = np.cos(2 * np.pi * df['second'] / 86400)

        # 3. Функция для парсинга странной строки в число
        def convert_to_float(val):
            # Превращаем строку "{'units':...}" в реальный словарь Python
            data = ast.literal_eval(val)
            # Считаем итоговое значение: units + nano / 10^9
            return data['units'] + data['nano'] / 1e9

        # Применяем функцию к колонке open
        df['open'] = df['open'].apply(convert_to_float)
        df['normal_price'] = (df['open']-df['open'].shift(1)) / df['open'].shift(1)*100
        df['predict_bool'] = (df['open'].rolling(window=delta_t+1).max().shift(-delta_t)-df['open']) / df['open']*100 > k
        #df['predict_abs'] = (df['open'].rolling(window=delta_t+1).max().shift(-delta_t)-df['open']) / df['open']*100
        #df['predict_max'] = df['open'].rolling(window=delta_t+1).max().shift(-delta_t)
        
        label_columns_features = []
        
        for t in range(0, delta_t):
            df[f'normal_price_{t}'] = df['normal_price'].shift(t)
            label_columns_features.append(f'normal_price_{t}')
            
        # Убираем первые строки с Nan
        df = df.iloc[delta_t+1:]
        
        # Разделяем данные (зависит от структуры вашего CSV)
        # Предположим, последний столбец 'target' — это то, что мы предсказываем
        X_data = df[label_columns_features].values
        y_data = df['predict_bool'].values
        
        # Превращаем в тензоры
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).reshape(-1, 1)
        
        #print(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# 2. Определяем модель
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        # Линейный слой: принимает input_dim признаков, выдает 1 число
        self.linear = nn.Linear(input_dim, 1)
        # Сигмоида превращает любое число в вероятность от 0 до 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out
    
# Собираем все вместе:
df = pd.read_csv(path_df)
df = df[pd.to_datetime(df['time']).dt.time != time(23)]
df = df.sort_values('time').reset_index(drop=True)

number_samples = len(df)
index_80th = round(number_samples * 0.8)

train_data = df.iloc[0:index_80th]
test_data = df.iloc[index_80th:]

dataset_train = MyParquetDataset(train_data, delta_t, k)
dataset_test= MyParquetDataset(test_data, delta_t, k)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Модель берем из предыдущего примера (input_dim = кол-во колонок в X)
model = LogisticRegressionModel(input_dim=dataset_train.X.shape[1])
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = torch.nn.BCELoss()

# Цикл обучения по батчам
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Эпоха [{epoch+1}/{epochs}], Ошибка: {loss.item():.4f}')
        

model.eval()

all_preds = []
all_true = []

with torch.no_grad():
    for batch_X, batch_y in dataset_test: # Используем тестовый загрузчик!
        outputs = model(batch_X)
        
        # Превращаем вероятности (0.7, 0.2...) в классы (1, 0...)
        # Если вероятность > 0.5, то это класс 1, иначе 0
        preds = (outputs > 0.5).float()
        
        all_preds.extend(preds.numpy())
        all_true.extend(batch_y.numpy())

print(len(all_preds))
print(len(all_true))

# Считаем базовые показатели
tp = sum((t == 1 and p == 1) for t, p in zip(all_true, all_preds))
tn = sum((t == 0 and p == 0) for t, p in zip(all_true, all_preds))
fp = sum((t == 0 and p == 1) for t, p in zip(all_true, all_preds))
fn = sum((t == 1 and p == 0) for t, p in zip(all_true, all_preds))

print(f"tp: {tp:.2f}")
print(f"tn: {tn:.2f}")
print(f"fp: {fp:.2f}")
print(f"fn: {fn:.2f}")

# Считаем метрики (добавляем 1e-9, чтобы избежать деления на ноль)
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")