import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
from datetime import time

#path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/BBG004730N88/CANDLE_INTERVAL_HOUR-1765180800-1770267600.csv' # Сбер
#path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/BBG004730ZJ9/CANDLE_INTERVAL_HOUR-1765195200-1770375600.csv' # ВТБ
#path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/FUTVTBR03260/CANDLE_INTERVAL_HOUR-1765198800-1770379200.csv' # Фьючерс ВТБ
#path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/BBG004731032/CANDLE_INTERVAL_HOUR-1765198800-1770379200.csv' # Магнит

path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/BBG004730N88/CANDLE_INTERVAL_1_MIN-1765185840-1770369780.csv' # Сбер
#path_df = 'C:/Users/Valya/.vscode/Bot/market_data_cache/BBG004730ZJ9/CANDLE_INTERVAL_1_MIN-1765561680-1770745620.csv' # ВТБ

##########################################################################
# Парсинг csv
##########################################################################
def init_df(path_df):
    # 1. Загрузка данных
    # Предположим, файл называется 'data.csv'
    df = pd.read_csv(path_df)
    df = df[pd.to_datetime(df['time']).dt.time != time(23, 0)]

    # 2. Преобразование времени в формат datetime
    df['time'] = pd.to_datetime(df['time'])

    # 3. Функция для парсинга странной строки в число
    def convert_to_float(val):
        # Превращаем строку "{'units':...}" в реальный словарь Python
        data = ast.literal_eval(val)
        # Считаем итоговое значение: units + nano / 10^9
        return data.get('units', 0) + data.get('nano', 0) / 1e9

    # Применяем функцию к колонке open
    df['open'] = df['open'].apply(convert_to_float)
    df['close'] = df['close'].apply(convert_to_float)
    df['high'] = df['high'].apply(convert_to_float)

    return df


##########################################################################
# Отрисовка графика
##########################################################################
def print_df(time, value):
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(time, value, marker='o', linestyle='-')

    plt.title('График цены Open')
    plt.xlabel('Время')
    plt.ylabel('Цена')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()
    
    
##########################################################################
# Средняя дневная волатильность
##########################################################################
def high_low():
    # 1. Читаем один раз
    df = init_df(path_df)
    
    # 2. Группируем по дате и сразу вычисляем min и max для 'open'
    stats = df.groupby(df['time'].dt.date)['high'].agg(['max', 'min'])
    
    # 3. Векторное вычисление (без циклов)
    val = (stats['max'] - stats['min']) / stats['min'] * 100
    
    return val.mean()
        

##########################################################################
# Отклонение от средней цены за день в разрезе времени
##########################################################################
def get_deviation_stats(path_df):
    # 1. Загружаем данные один раз
    df = init_df(path_df).copy()
    
    # 2. Считаем среднее по каждому дню и "растягиваем" его на все строки этого дня
    day_mean = df.groupby(df['time'].dt.date)['high'].transform('mean')
    
    print()
    
    # 3. Вычисляем процент отклонения для каждой строки (векторно)
    df['deviation'] = (df['high'] - day_mean) / day_mean * 100
    
    # 4. Группируем по времени суток (time) и считаем среднее отклонение
    df_deviation_mean = (
        df.groupby(df['time'].dt.time)['deviation']
        .mean()
        .reset_index()
        .rename(columns={'time': 'time', 'deviation': 'value'})
    )
    
    # Сортируем по времени для корректного вывода
    df_deviation_mean = df_deviation_mean.sort_values(by='time')
    
    # 5. Поиск максимума и минимума
    max_row = df_deviation_mean.loc[df_deviation_mean['value'].idxmax()]
    min_row = df_deviation_mean.loc[df_deviation_mean['value'].idxmin()]
    
    print(f"Максимальное среднее отклонение: {{'time': '{max_row.time}', 'value': {max_row.value}}}")
    print(f"Минимальное среднее отклонение: {{'time': '{min_row.time}', 'value': {min_row.value}}}")

    return df_deviation_mean


##########################################################################
# Рост цены, которое выполняется с вероятностью k
##########################################################################
def get_garant_deviation(path_df, k):
    df = init_df(path_df).copy()
    
    # 1. Создаем колонку с датой для группировки
    df['date'] = df['time'].dt.date
    
    # 2. Считаем "максимум в будущем" для каждой строки внутри дня
    # Мы разворачиваем каждую группу, считаем кумулятивный максимум и разворачиваем обратно
    df['max_future'] = (
        df.groupby('date')['high']
        .transform(lambda x: x[::-1].cummax()[::-1])
    )
    
    # 3. Вычисляем отклонение сразу для всего столбца (векторно)
    df['deviation'] = (df['max_future'] - df['high']) / df['high'] * 100
    
    threshold = df['deviation'].quantile(1-k)
    
    return threshold
            

if __name__ == "__main__":
    print('Средняя дневная волатильность: ' + str(high_low()))
    #df = get_deviation_stats(path_df)
    #df = init_df(path_df)
    #print_df(df['time'].astype(str), df['value'])
    print(get_garant_deviation(path_df, 0.9))
    