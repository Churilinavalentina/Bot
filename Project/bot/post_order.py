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

load_dotenv()
TOKEN = os.environ["INVEST_TOKEN"]


async def post_order(direction, account_id, figi, quantity, price):
    async with AsyncClient(TOKEN, target=TARGET) as client:
        accounts = await client.users.get_accounts()
        request = PostOrderAsyncRequest(
            order_type=OrderType.ORDER_TYPE_MARKET,
            direction=OrderDirection.ORDER_DIRECTION_BUY,
            instrument_id=figi,
            quantity=quantity,
            account_id=account_id,
            order_id=str(uuid4()),
            price=price,
        )
        response = await client.orders.post_order_async(request=request)
        print(response)

        
