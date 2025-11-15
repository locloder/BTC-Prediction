import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import json
import os 
import argparse
import sys

output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

MAX_DAYS = 30
VALID_INTERVALS = ['15m', '30m', '1h', '4h', '1d']

parser = argparse.ArgumentParser(
    description="Завантаження історичних даних ціни BTC-USD"
)

parser.add_argument(
    '--days', 
    type=int, 
    default=7, 
    help=f'Кількість днів для завантаження даних (максимум {MAX_DAYS}) | За замовчуванням: 7'
)

parser.add_argument(
    '--interval', 
    type=str, 
    default='1h', 
    help=f'Інтервал даних (Варіанти: {", ".join(VALID_INTERVALS)}) | За замовчуванням: 1h'
)

args = parser.parse_args()

days_to_fetch = args.days
interval_to_fetch = args.interval

if days_to_fetch <= 0 or days_to_fetch > MAX_DAYS:
    print(f"\n Невірно вказана кількість днів. | Максимум {MAX_DAYS} днів")
    sys.exit(1)

if interval_to_fetch not in VALID_INTERVALS:
    display_intervals = ['15m', '30m', '1h', '1d'] 
    print(f"\n Невірно вказаний інтервал | Можливо лише {', '.join(display_intervals)}")
    sys.exit(1)


end_time = datetime.now()
start_time = end_time - timedelta(days=days_to_fetch)

ticker_btc = "BTC-USD"
print(f"Завантаження даних для {ticker_btc} за останні {days_to_fetch} днів з інтервалом {interval_to_fetch}...")

data_btc = yf.download(
    ticker_btc,
    start=start_time.strftime('%Y-%m-%d'),
    end=end_time.strftime('%Y-%m-%d'),
    interval=interval_to_fetch, 
    auto_adjust=True
)

if data_btc.empty:
    print("Не вдалося завантажити дані. Перевірте вказані параметри.")
else:
    vector_value = data_btc[['Close']].reset_index()
    vector_value.columns = ['time', 'price']
    vector_value['time'] = vector_value['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    vector_value['price'] = vector_value['price'].round(2)
    vector_price = vector_value.to_dict('records')

    file_json = "parsing/btc_prices.json"
    try:
        with open(file_json, 'w', encoding='utf-8') as f:
            json.dump(vector_price, f, ensure_ascii=False, indent=4)

        print(f"\n Успішно збережено {len(vector_price)} записів у файл: {os.path.abspath(file_json)}")

    except Exception as e:
        print(f"\n Помилка при збереженні файлу: {e}")