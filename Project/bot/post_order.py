import asyncio
import sys
import os
# Добавляем путь к папке config (родительская папка + config)
sys.path.append(os.path.join(os.getcwd(), 'Project'))

from uuid import uuid4
from dotenv import load_dotenv

from t_tech.invest import AsyncClient
from t_tech.invest.schemas import OrderDirection, OrderType, PostOrderAsyncRequest
from params import TARGET
from t_tech.invest.schemas import Quotation

load_dotenv(os.path.join(os.getcwd(), '.env'), override=True)
TOKEN = os.environ["INVEST_TOKEN"]

def float_to_quotation(value: float):
    units = int(value)
    nano = int(round((value - units) * 1e9))
    return Quotation(units=units, nano=nano)

async def post_order(client, direction, account_id, figi, quantity, price):
    # Преобразуем цену и количество
    quotation_price = float_to_quotation(price)
    
    request = PostOrderAsyncRequest(
        order_type=OrderType.ORDER_TYPE_LIMIT, # Для лимитной заявки цена обязательна
        direction=direction,
        instrument_id=figi,
        quantity=int(quantity), # Исправляем предыдущую ошибку с float
        account_id=account_id,
        order_id=str(uuid4()),
        price=quotation_price,  # Теперь это объект Quotation
    )
    #response = await client.orders.post_order_async(request=request)
    response = await client.sandbox.post_sandbox_order_async(request=request)
    print(response)

        
