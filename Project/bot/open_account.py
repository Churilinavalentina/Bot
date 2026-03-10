import os
import asyncio
from t_tech.invest import AsyncClient, MoneyValue
from t_tech.invest.utils import decimal_to_quotation

from t_tech.invest.sandbox.client import SandboxClient
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.environ["INVEST_TOKEN"]


async def prepare_sandbox_account(token: str, amount_rub: int):
    async with AsyncClient(token) as client:
        # 1. Открываем новый счет в песочнице
        opened_account = await client.sandbox.open_sandbox_account()
        account_id = opened_account.account_id
        print(f"✅ Создан новый счет: {account_id}")

        # 2. Формируем сумму пополнения (units - целые, nano - копейки)
        money = MoneyValue(units=amount_rub, nano=0, currency='rub')

        # 3. Пополняем счет
        pay_in_response = await client.sandbox.sandbox_pay_in(
            account_id=account_id, 
            amount=money
        )
        
        # Проверяем итоговый баланс
        print(f"💰 Счет пополнен на {amount_rub} руб.")
        print(f"📊 Текущий баланс: {pay_in_response.balance.units} {pay_in_response.balance.currency}")
        
        return account_id

# Пример запуска
if __name__ == "__main__":
    # Запускаем асинхронную функцию
    acc_id = asyncio.run(prepare_sandbox_account(TOKEN, 1_000_000))
    print(f"🚀 Готово к торговле! Используйте ID: {acc_id}")
