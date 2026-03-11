import asyncio
import os
from t_tech.invest import AsyncClient
from dotenv import load_dotenv

load_dotenv(os.path.join(os.getcwd(), '.env'), override=True)
from params import TARGET
TOKEN = os.environ["INVEST_TOKEN"]

async def get_account_status(client, account_id: str):
    # 1. Получаем портфель (активы, акции, валюта на счету)
    portfolio = await client.sandbox.get_sandbox_portfolio(account_id=account_id)
    
    print(f"--- Состояние счета {account_id} ---")
    
    # Общая стоимость портфеля (сумма всех активов)
    total_value = portfolio.total_amount_shares.units + (portfolio.total_amount_shares.nano / 1e9)
    print(f"💰 Общая стоимость активов: {total_value} {portfolio.total_amount_shares.currency}")

    # Выводим список позиций (если они есть)
    if not portfolio.positions:
        print("📭 Портфель пуст.")
    else:
        for pos in portfolio.positions:
            print(f"📌 Актив (FIGI): {pos.figi}, Количество: {pos.quantity.units}")

    # 2. Если нужно узнать именно остаток "свободных" денег (кеш):
    # Для этого в песочнице используется get_sandbox_withdraw_limits
    limits = await client.sandbox.get_sandbox_withdraw_limits(account_id=account_id)
    for money in limits.money:
        print(f"💵 Свободный кеш: {money.units}.{money.nano // 10**7:02d} {money.currency}")
    print(f"Полная стоимость портфеля: {money.units + money.nano / 1e9 +total_value} {money.currency}")   
    return money.units + money.nano / 1e9 +total_value

    
    