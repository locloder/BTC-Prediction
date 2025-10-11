import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import json
import os 

end_time = datetime.now()
start_time = end_time - timedelta(days=7)

ticker_btc = "BTC-USD"
data_btc = yf.download(
    ticker_btc,
    start=start_time.strftime('%Y-%m-%d'),
    end=end_time.strftime('%Y-%m-%d'),
    interval="1h", 
    auto_adjust=True 
)


# Формуємо кінцевий вектор об'єктів (список словників)

vector_value = data_btc[['Close']].reset_index()

# Перейменовуємо стовпці
vector_value.columns = ['time', 'price']

# Форматуємо time та округлюємо ціну
vector_value['time'] = vector_value['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
vector_value['price'] = vector_value['price'].round(2)

vector_price = vector_value.to_dict('records')

file_json = "btc_prices.json"
try:
    with open(file_json, 'w', encoding='utf-8') as f:
        json.dump(vector_price, f, ensure_ascii=False, indent=4)
        
    # Виводимо результат
    print(f" Успішно збережено дані у файл: {os.path.abspath(file_json)}")
        
except Exception as e:
    print(f" Помилка при збереженні файлу: {e}")