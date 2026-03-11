import os
import sys
import asyncio
from t_tech.invest import AsyncClient, MoneyValue
from t_tech.invest.utils import decimal_to_quotation

# Добавляем путь к папке config (родительская папка + config)
sys.path.append(os.path.join(os.getcwd(), 'Project'))

from t_tech.invest.sandbox.client import SandboxClient
from dotenv import load_dotenv
from params import TARGET

load_dotenv(os.path.join(os.getcwd(), '.env'), override=True)
TOKEN = os.environ["INVEST_TOKEN"]
print(f"Актуальный токен: {TOKEN[:10]}...")


async def prepare_sandbox_account(token: str, amount_rub: int):
    async with AsyncClient(token, target=TARGET) as client:
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


async def clear_all_sandbox_accounts():
    async with AsyncClient(TOKEN, target=TARGET) as client:
        # 1. Получаем список всех счетов
        response = await client.sandbox.get_sandbox_accounts()
        accounts = response.accounts
        
        if not accounts:
            print("Счетов для удаления не найдено.")
            return

        print(f"Найдено счетов: {len(accounts)}. Начинаю удаление...")
        
        # 2. Проходим циклом и закрываем каждый
        for acc in accounts:
            try:
                await client.sandbox.close_sandbox_account(account_id=acc.id)
                print(f"Аккаунт {acc.id} успешно удален.")
            except Exception as e:
                print(f"Ошибка при удалении {acc.id}: {e}")
        
        print("Все аккаунты в песочнице очищены.")

async def wait_until_no_active_orders(client, account_id, check_interval=60):
    while True:
        # Используем переданный клиент
        response = await client.sandbox.get_sandbox_orders(account_id=account_id)
        active_orders = response.orders
        
        if not active_orders:
            print("✅ Активных заявок нет.")
            break
            
        print(f"⏳ В очереди {len(active_orders)} заявок. Ждем...")
        await asyncio.sleep(check_interval)
        
        
# Пример запуска
if __name__ == "__main__":
    # Запускаем асинхронную функцию
    asyncio.run(clear_all_sandbox_accounts())
    acc_id = asyncio.run(prepare_sandbox_account(TOKEN, 1_000_000))
    print(f"🚀 Готово к торговле! Используйте ID: {acc_id}")
