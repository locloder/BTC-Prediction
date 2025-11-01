import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
import numpy as np
import requests
import re
import pandas as pd

def save_btc_data_for_predictor(df, file_path):
    """Mocks saving data."""
    if df is None or df.empty:
        return False

    df_copy = df.copy()

    if df_copy.index.tz is not None:
        df_copy.index = df_copy.index.tz_convert('UTC').tz_localize(None)

    vector_value = df_copy[['Close']].reset_index()
    vector_value.columns = ['time', 'price']
    vector_value['time'] = vector_value['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    vector_value['price'] = vector_value['price'].round(2)
    vector_price = vector_value.to_dict('records')
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vector_price, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"Ошибка сохранения данных: {e}")
        return False


def mock_load_news_data(reddit_enabled, twitter_enabled):
    """Mocks loading news data."""
    news_context = []
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if reddit_enabled:
        news_context.append(
            {"time": current_time, "news": "Reddit: Positive sentiment noted on r/Crypto. Expect bullish movement."})
    if twitter_enabled:
        news_context.append({"time": current_time,
                             "news": "Twitter: Regulatory uncertainty trending. Potential short-term correction."})
    return news_context


class TimeSeriesPredictor:
    def __init__(self, json_file_path, news_context, time_gap_hours=None):
        self.news_context = news_context
        self.price_history = []
        self.json_file_path = json_file_path
        self.load_price_data()
        self.time_gap_hours = 1

    def load_price_data(self):
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self.price_history = json.load(f)
        except:
            self.price_history = []

    def predict(self, num_predictions=5):
        if not self.price_history:
            return []
        last_price = self.price_history[-1]['price']
        return [last_price + np.random.normal(0, last_price * 0.005) * (i + 1) for i in range(num_predictions)]

    def generate_future_dates(self, num_predictions):
        if not self.price_history:
            return []
        last_timestamp = datetime.strptime(self.price_history[-1]['time'], '%Y-%m-%d %H:%M:%S')

        time_gap_seconds = {
            "15m": 15 * 60,
            "30m": 30 * 60,
            "1h": 60 * 60,
            "4h": 4 * 60 * 60,
            "1d": 24 * 60 * 60
        }.get(current_tf, 60 * 60)

        return [(last_timestamp + timedelta(seconds=time_gap_seconds * (i + 1))).strftime('%Y-%m-%d %H:%M:%S') for i in
                range(num_predictions)]

root = tk.Tk()
root.title("BTC-Prediction")
root.geometry("1000x700")
root.configure(bg="#1f1f23")

title_frame = tk.Frame(root, bg="#1f1f23")
title_frame.pack(anchor="w", pady=10, padx=20)
tk.Label(title_frame, text="BTC", fg="#f38b00", bg="#1f1f23", font=("Shippori Antique B1", 14)).pack(side="left")
tk.Label(title_frame, text="-Prediction", fg="#cfcfcf", bg="#1f1f23", font=("Shippori Antique B1", 14)).pack(
    side="left")

toolbar = tk.Frame(root, bg="#1f1f23")
toolbar.pack(anchor="w", padx=20, pady=5, fill="x")

main_frame = tk.Frame(root, bg="#1f1f23")
main_frame.pack(padx=30, pady=20, fill="both", expand=True)

canvas_frame = tk.Frame(main_frame, bg="#2b2b2f", width=700, height=500)
canvas_frame.pack(side="left", fill="both", expand=True)

checkbox_frame = tk.Frame(main_frame, bg="#1f1f23", width=200)
checkbox_frame.pack(side="left", fill="y", padx=10, pady=10)

var1 = tk.BooleanVar()  # Reddit
var2 = tk.BooleanVar()  # Twitter

tk.Checkbutton(checkbox_frame, text="Reddit", variable=var1, fg="#ffffff", bg="#1f1f23", selectcolor="#333",
               activebackground="#1f1f23", activeforeground="#f38b00").pack(anchor="nw", pady=5)
tk.Checkbutton(checkbox_frame, text="Twitter", variable=var2, fg="#ffffff", bg="#1f1f23", selectcolor="#333",
               activebackground="#1f1f23", activeforeground="#f38b00").pack(anchor="nw", pady=5)

current_tf = "1h"
current_price = 0.0
prediction_data = []
btc_data_df = None


def fetch_btc_data(interval="1h"):
    """Fetches BTC data and saves it to a global DataFrame."""
    global btc_data_df
    end = datetime.now()
    days_to_fetch = 60 if interval in ["15m", "30m", "1h", "4h"] else 365
    start = end - timedelta(days=days_to_fetch)

    try:
        data = yf.download("BTC-USD", start=start, end=end, interval=interval, auto_adjust=False)
        if data.empty:
            raise ValueError(f"Нет данных для {interval}")
        btc_data_df = data
        return data
    except Exception as e:
        status_label.config(text=f"Ошибка загрузки: {e}")
        btc_data_df = None
        return None


def update_plot():
    """Clears and redraws the main price chart, including predictions."""
    global current_price, prediction_data
    ax.clear()

    data_raw = fetch_btc_data(current_tf)
    if data_raw is None or data_raw.empty:
        canvas.draw()
        return

    data = data_raw.copy()
    if data.index.tz is not None:
        data.index = data.index.tz_convert('UTC').tz_localize(None)

    ax.plot(data.index, data["Close"], color="#f38b00", linewidth=1.8, label="BTC/USD")
    ax.set_xlim(data.index[0], data.index[-1])

    y_min = data["Close"].min().item()
    y_max = data["Close"].max().item()
    current_price = data["Close"].iloc[-1].item()

    last_time = data.index[-1]
    ax.axhline(y=current_price, color="#888", linestyle="--", linewidth=1)
    ax.text(last_time, current_price, f" ${current_price:.2f}", color="#f0f0f0", fontsize=9, va="bottom")

    if prediction_data:
        try:
            future_times_str, future_prices_raw = zip(*prediction_data)
            future_prices = [float(p) for p in future_prices_raw]
            future_times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in future_times_str]

            last_hist_time = data.index[-1]
            last_hist_price = current_price

            plot_times = [last_hist_time] + future_times
            plot_prices = [last_hist_price] + list(future_prices)

            ax.plot(plot_times, plot_prices, color="#00ffcc", linestyle="--", linewidth=1.5, marker='o', markersize=3,
                    label="Прогноз AI")

            ax.set_xlim(data.index[0], future_times[-1])
            pred_min = min(future_prices)
            pred_max = max(future_prices)
            y_min = min(y_min * 0.98, pred_min * 0.98)
            y_max = max(y_max * 1.02, pred_max * 1.02)
        except Exception as e:
            print(f"Ошибка отрисовки прогноза: {e}")
            prediction_data = []

    ax.set_ylim(y_min, y_max)
    ax.set_facecolor("#2b2b2f")
    fig.patch.set_facecolor("#2b2b2f")
    ax.spines["bottom"].set_color("#dcdcdc")
    ax.spines["left"].set_color("#dcdcdc")
    ax.tick_params(colors="#dcdcdc")
    ax.set_title(f"Bitcoin (BTC/USD) — timeframe: {current_tf}", fontsize=12, color="#ffffff")
    ax.grid(True, alpha=0.3, color="#555")

    if prediction_data:
        ax.legend(facecolor="#2b2b2f", edgecolor="#555", labelcolor=["#f38b00", "#00ffcc"])
    else:
        ax.legend(facecolor="#2b2b2f", edgecolor="#555", labelcolor="#f38b00")

    fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)
    canvas.draw()
    root.after(60000, update_plot)
    status_label.config(text=f"Текущая цена: ${current_price:.2f}")


def change_timeframe(tf):
    global current_tf, prediction_data
    current_tf = tf
    prediction_data = []
    update_plot()


def predict_action():
    global prediction_data

    if btc_data_df is None or btc_data_df.empty:
        status_label.config(text="Ошибка: Нет исторических данных для прогноза.")
        return

    status_label.config(text="Запуск прогнозирования... (требуется работа Ollama/llama2)")
    root.update_idletasks()

    temp_price_path = "temp_btc_prices.json"
    if not save_btc_data_for_predictor(btc_data_df, temp_price_path):
        status_label.config(text="Ошибка: Не удалось сохранить временный файл данных.")
        return

    news_context = mock_load_news_data(var1.get(), var2.get())

    try:
        predictor = TimeSeriesPredictor(json_file_path=temp_price_path, news_context=news_context)
        NUM_PREDICTIONS = 5
        predictions = predictor.predict(NUM_PREDICTIONS)

        if predictions:
            future_dates = predictor.generate_future_dates(NUM_PREDICTIONS)
            prediction_data = list(zip(future_dates, predictions))
            status_label.config(text=f"Прогноз готов: {NUM_PREDICTIONS} точек. ({len(news_context)} новостей учтено)")
        else:
            status_label.config(text="Прогнозирование не дало результата.")
            prediction_data = []

    except Exception as e:
        status_label.config(text=f"Критическая ошибка прогнозирования (Ollama/Network?): {e}")
        prediction_data = []
    finally:
        if os.path.exists(temp_price_path):
            os.remove(temp_price_path)

    update_plot()


predict_button = tk.Button(
    checkbox_frame,
    text="Predict",
    bg="#f38b00",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    relief="flat",
    activebackground="#d67a00",
    activeforeground="white",
    padx=10,
    pady=5,
    command=predict_action
)
predict_button.pack(anchor="nw", pady=(20, 5))

tf_block = tk.Frame(toolbar, bg="#1f1f23")
tf_block.pack(side="left", padx=(0, 20))
tk.Label(tf_block, text="Timeframe", fg="#dcdcdc", bg="#1f1f23", font=("Segoe UI", 10)).pack(anchor="center",
                                                                                             pady=(0, 3))

timeframes = {"15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}
timeframe_frame = tk.Frame(tf_block, bg="#1f1f23")
timeframe_frame.pack(anchor="center")

for name, tf_code in timeframes.items():
    tk.Button(
        timeframe_frame, text=name, bg="#1f1f23", fg="white",
        relief="solid", bd=1, font=("Segoe UI", 9),
        activebackground="#333", activeforeground="#f38b00",
        width=4, command=lambda tf=tf_code: change_timeframe(tf)
    ).pack(side="left", padx=6, pady=(0, 2))

fig, ax = plt.subplots(facecolor="#2b2b2f", figsize=(9.5, 5.8))
canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

status_label = tk.Label(root, text="", bg="#1f1f23", fg="#dcdcdc", font=("Segoe UI", 10))
status_label.pack(pady=(0, 10))


def on_motion(event):
    if event.xdata and event.ydata:
        try:
            x_time = matplotlib.dates.num2date(event.xdata)
            status_label.config(
                text=f"Дата: {x_time.strftime('%Y-%m-%d %H:%M:%S')} (UTC) | Цена: ${event.ydata:.2f} | Текущая BTC: ${current_price:.2f}"
            )
        except:
            status_label.config(text=f"Текущая цена: ${current_price:.2f}")
    else:
        status_label.config(text=f"Текущая цена: ${current_price:.2f}")


fig.canvas.mpl_connect("motion_notify_event", on_motion)


def zoom(event):
    if event.inaxes != ax:
        return
    x_min, x_max = ax.get_xlim()
    xdata = event.xdata
    if xdata is None:
        return
    scale_factor = 1.2 if event.button == 'down' else 1 / 1.2
    new_left = xdata - (xdata - x_min) * scale_factor
    new_right = xdata + (x_max - xdata) * scale_factor
    ax.set_xlim(new_left, new_right)
    canvas.draw_idle()


fig.canvas.mpl_connect("scroll_event", zoom)

is_panning_left = False
last_mouse_pos = None


def on_press_left(event):
    global is_panning_left, last_mouse_pos
    if event.button == 1 and event.inaxes == ax:
        is_panning_left = True
        last_mouse_pos = [event.xdata, event.ydata]


def on_release_left(event):
    global is_panning_left, last_mouse_pos
    if event.button == 1:
        is_panning_left = False
        last_mouse_pos = None


def on_move_left(event):
    global last_mouse_pos
    if is_panning_left and event.xdata is not None and event.ydata is not None and last_mouse_pos is not None:
        dx = last_mouse_pos[0] - event.xdata
        dy = last_mouse_pos[1] - event.ydata
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.set_xlim(x_min + dx, x_max + dx)
        ax.set_ylim(y_min + dy, y_max + dy)
        canvas.draw_idle()
        last_mouse_pos = [event.xdata, event.ydata]


fig.canvas.mpl_connect("button_press_event", on_press_left)
fig.canvas.mpl_connect("button_release_event", on_release_left)
fig.canvas.mpl_connect("motion_notify_event", on_move_left)

update_plot()
root.mainloop()
