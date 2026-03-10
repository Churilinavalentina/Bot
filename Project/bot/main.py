import asyncio
import sys
import os
import torch

# Добавляем путь к папке config (родительская папка + config)
sys.path.append(os.path.join(os.getcwd(), 'Project'))

from params import INTERVAL, START_DATE, END_DATE, TICKERS, BATCH_SIZE, DELTA_T, DELTA_T_FUTURE, K
from get_data import get_figi, get_values
from read_data_bot import MyParquetBotDataset
from post_order import post_order
from get_account_status import get_account_status
from model import MyModel
from t_tech.invest.utils import now
from datetime import timedelta
from torch.utils.data import DataLoader
from t_tech.invest.schemas import OrderDirection
from tqdm.autonotebook import tqdm
from t_tech.invest import AsyncClient
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()
TOKEN = os.environ["INVEST_TOKEN"]

# 1. Выносим модель и устройство наружу, чтобы загрузить ОДИН РАЗ
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = MyModel(input_dim=DELTA_T).to(device)
model.load_state_dict(torch.load('C:/Users/Valya/.vscode/Bot/model/27_02.pth', map_location=device))
model.eval()

# TODO: пересмотреть. figi смотрим все, но покупаем ту акцию, где предсказание выполняется и максимальное

# 2. Делаем основную функцию асинхронной (async def)
async def trade_step(market_price, ID, prediction_delta, cash, quantity, figi_order):
    # Используем глобальную модель
    global model
    
    # FIGI и данные
    figi = get_figi(TICKERS).figi
    get_values(figi, INTERVAL, now() - timedelta(minutes=DELTA_T+1), now(), num_workers=1)
    #get_values(figi, INTERVAL, datetime(2026, 3, 4, tzinfo=timezone.utc), datetime(2026, 3, 5, tzinfo=timezone.utc), num_workers=1)

    ds = MyParquetBotDataset(DELTA_T, DELTA_T_FUTURE, K, now() - timedelta(minutes=DELTA_T), now())
    #ds = MyParquetBotDataset(DELTA_T, DELTA_T_FUTURE, K, datetime(2026, 3, 4, tzinfo=timezone.utc), datetime(2026, 3, 5, tzinfo=timezone.utc))
    ds_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print("!!!!")
    print(ds.__len__())

    all_preds = []
    with torch.no_grad():
        for batch_X, batch_y in ds_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            all_preds.extend(outputs)

    print("Длина предсказаний " + str(len(all_preds)))
    all_preds = torch.cat(all_preds)
    print(len(all_preds))
    max_idx_tensor = all_preds.argmax().item()
    max_preds = all_preds[max_idx_tensor].item()
    print("Предсказание " + str(max_preds))
    current_price = ds.get_data_by_figi(ds.get_figi_by_index(max_idx_tensor))[1].item()
    print("Текущая цена " + str(current_price))

    # Добавить подсчет количества
    if (max_preds >= prediction_delta) & (quantity == 0):
        figi_order = ds.get_figi_by_index(max_idx_tensor)
        print(ds.get_data_by_figi(figi_order)[1])
        market_price = ds.get_data_by_figi(figi_order)[1].item()
        quantity = cash // market_price

        print(f"Покупаем: {figi_order}, по цене: {market_price}, в количестве: {quantity}")
        #await post_order(OrderDirection.ORDER_DIRECTION_BUY, ID, figi_order, quantity)
        return market_price, quantity, figi_order 
    
    if (current_price >= market_price * (1.0 + K)) & (quantity > 0):
        print(f"Продаем: {figi_order}, по цене: {market_price}, в количестве: {quantity}")
        #await post_order(OrderDirection.ORDER_DIRECTION_SELL, ID, figi_order, quantity)
        
        quantity =0
        figi_order = ''
        market_price = 0.0
        return market_price, quantity, figi_order # обновляем цену для следующего шага
    
    return market_price, quantity, figi_order

# 3. Создаем бесконечный цикл
async def main_loop(start_price, account_id, prediction_delta, figi_order):
    price = start_price
    quantity = 0
    while True:
        try:
            print("\n--- Новый цикл проверки ---")
            cash = await get_account_status(TOKEN, account_id)
            price, quantity, figi_order = await trade_step(price, account_id, prediction_delta, cash, quantity, figi_order)
        except Exception as e:
            print(f"❌ Ошибка в цикле: {e}")
        
        await asyncio.sleep(60) # Пауза 1 минута

# 4. Точка входа
if __name__ == "__main__":
    INITIAL_PRICE = 0.0 # Укажите стартовую цену
    ACCOUNT_ID = "07783257-b762-406d-9596-6df39a6e8180"
    prediction_delta = 0.5
    figi_order = ''
    
    # Запускаем единственный event loop
    asyncio.run(main_loop(INITIAL_PRICE, ACCOUNT_ID, prediction_delta, figi_order))
    
    
