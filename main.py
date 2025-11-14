import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
import numpy as np
import pandas as pd

# Import the consolidated predictor
from ai_requests import TimeSeriesPredictor
from ls_method import fit_and_forecast, draw_approximations, DEFAULT_CONFIG

# global vars
is_panning_left = False
current_tf = "1h"
current_price = 0.0
prediction_data = []
btc_data_df = None

# Create data folder if it doesn't exist
def ensure_data_folder():
    """Ensure the data folder exists before any file operations"""
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created {data_folder} directory")

# Call this function at the start
ensure_data_folder()

# event callbacks
def on_press_left(event):
    global is_panning_left
    if event.button != 1 or event.inaxes != ax or event.x is None or event.y is None:
        return
    is_panning_left = True
    ax.start_pan(event.x, event.y, 1)

def on_release_left(event):
    global is_panning_left
    if not is_panning_left:
        return
    is_panning_left = False
    ax.end_pan()
    canvas.draw_idle()

def on_move_left(event):
    if event.button != 1 or not is_panning_left or event.inaxes != ax or event.x is None or event.y is None:
        return
    ax.drag_pan(is_panning_left, None, event.x, event.y)
    canvas.draw_idle()

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

def on_motion(event):
    if event.xdata and event.ydata:
        try:
            x_time = matplotlib.dates.num2date(event.xdata)
            status_label.config(
                text=f"Date: {x_time.strftime('%Y-%m-%d %H:%M:%S')} (UTC) | Price: ${event.ydata:.2f} | Current BTC: ${current_price:.2f}"
            )
        except:
            status_label.config(text=f"Current price: ${current_price:.2f}")
    else:
        status_label.config(text=f"Current price: ${current_price:.2f}")

# data processing
def save_btc_data_for_predictor(df, file_path):
    """Saves DataFrame data to JSON in the format TimeSeriesPredictor expects."""
    if df is None or df.empty:
        return False

    # Ensure directory exists for the file path
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

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
        print(f"Error saving data: {e}")
        return False

def load_news_data():
    """Load news data from news.json file"""
    news_file_path = "data/news.json"
    news_context = []

    if not os.path.exists(news_file_path):
        return news_context

    try:
        with open(news_file_path, 'r', encoding='utf-8') as f:
            news_data = json.load(f)

        articles = news_data.get('articles', [])
        for article in articles[:10]:  # Take only latest 10 articles
            headline = article.get('headline', '')
            description = article.get('description', '')
            news_text = f"{headline}. {description}"
            timestamp = article.get('data', '').replace('T', ' ').replace('Z', '')

            if news_text.strip() and timestamp:
                news_context.append({
                    "time": timestamp,
                    "news": news_text
                })

    except Exception as e:
        print(f"Error loading news data: {e}")

    return news_context

def get_predictions_count(timeframe, days):
    """Calculate number of predictions needed based on timeframe and days"""
    # Map timeframe to periods per day
    periods_per_day = {
        "1h": 24,  # 24 periods per day (1 hour each)
        "4h": 6,  # 6 periods per day (4 hours each)
        "1d": 1,  # 1 period per day
        "5d": 0.2,  # 0.2 periods per day (1 period every 5 days)
        "1wk": 0.142857,  # ~0.143 periods per day (1 period every 7 days)
        "1mo": 0.0333  # ~0.033 periods per day (1 period every 30 days)
    }

    periods_per_day_count = periods_per_day.get(timeframe, 1)
    return max(1, int(periods_per_day_count * days))

def fetch_btc_data(interval="1h"):
    """Fetches BTC data and saves it to a global DataFrame."""
    global btc_data_df

    end = datetime.now()

    # Adjust days to fetch based on timeframe
    if interval == "1mo":
        days_to_fetch = 365 * 3  # 3 years for monthly data
    elif interval == "1wk":
        days_to_fetch = 365 * 2  # 2 years for weekly data
    elif interval == "5d":
        days_to_fetch = 365 * 2  # 2 years for 5-day data
    elif interval == "1d":
        days_to_fetch = 365  # 1 year for daily data
    elif interval == "4h":
        days_to_fetch = 90  # 3 months for 4-hour data
    else:  # 1h and others
        days_to_fetch = 60  # 2 months for hourly data

    start = end - timedelta(days=days_to_fetch)

    try:
        data = yf.download("BTC-USD", start=start, end=end, interval=interval, auto_adjust=False)
        if data.empty:
            raise ValueError(f"No data for {interval}")
        btc_data_df = data
        return data
    except Exception as e:
        status_label.config(text=f"Loading error: {e}")
        btc_data_df = None
        return None

def predict_action():
    global prediction_data

    if btc_data_df is None or btc_data_df.empty:
        status_label.config(text="Error: No historical data for prediction.")
        return

    # Calculate number of predictions needed
    prediction_days = prediction_days_var.get()
    NUM_PREDICTIONS = get_predictions_count(current_tf, prediction_days)

    status_label.config(text=f"Starting prediction for {prediction_days} days ({NUM_PREDICTIONS} points)...")
    root.update_idletasks()

    # Use data folder for temporary files
    temp_price_path = "data/temp_btc_prices.json"
    if not save_btc_data_for_predictor(btc_data_df, temp_price_path):
        status_label.config(text="Error: Could not save temporary data file.")
        return

    # Prepare news data if enabled
    temp_news_path = None
    if consider_news_var.get():
        news_context = load_news_data()
        if news_context:
            temp_news_path = "data/temp_news.json"
            try:
                news_data = {
                    "metadata": {"total_articles": len(news_context)},
                    "articles": [{"headline": "", "description": item["news"], "data": item["time"]} for item in
                                 news_context]
                }
                with open(temp_news_path, 'w', encoding='utf-8') as f:
                    json.dump(news_data, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"Error saving news data: {e}")
                temp_news_path = None

    try:
        # Use the consolidated TimeSeriesPredictor from ai_requests.py
        predictor = TimeSeriesPredictor(json_file_path=temp_price_path, news_file_path=temp_news_path)

        status_label.config(text=f"Generating {NUM_PREDICTIONS} predictions...")
        root.update_idletasks()

        predictions = predictor.predict(NUM_PREDICTIONS)

        if predictions:
            future_dates = predictor.generate_future_dates(NUM_PREDICTIONS, current_tf)
            prediction_data = list(zip(future_dates, predictions))
            news_status = "with news" if consider_news_var.get() and temp_news_path else "without news"
            status_label.config(
                text=f"Prediction ready: {NUM_PREDICTIONS} points ({prediction_days} days) {news_status}")
        else:
            status_label.config(text="Prediction failed.")
            prediction_data = []

    except Exception as e:
        status_label.config(text=f"Critical prediction error (Ollama/Network?): {e}")
        prediction_data = []
    finally:
        # Clean up temporary files
        if os.path.exists(temp_price_path):
            os.remove(temp_price_path)
        if temp_news_path and os.path.exists(temp_news_path):
            os.remove(temp_news_path)

    update_plot()

# custom gui callbacks
def toggle_news():
    if consider_news_var.get():
        consider_news_var.set(False)
        news_button.config(text="Consider News: OFF", bg="#cc0000", activebackground="#990000")
    else:
        consider_news_var.set(True)
        news_button.config(text="Consider News: ON", bg="#00cc00", activebackground="#009900")

def change_timeframe(tf):
    global current_tf, prediction_data
    current_tf = tf
    prediction_data = []
    update_plot()

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
                    label="AI Prediction")

            ax.set_xlim(data.index[0], future_times[-1])
            pred_min = min(future_prices)
            pred_max = max(future_prices)
            y_min = min(y_min * 0.98, pred_min * 0.98)
            y_max = max(y_max * 1.02, pred_max * 1.02)
        except Exception as e:
            print(f"Prediction rendering error: {e}")
            prediction_data = []

    if show_approx_var.get():
        try:
            n_future = get_predictions_count(current_tf, prediction_days_var.get())

            # Optional: tweak LS config here
            ls_cfg = {
                **DEFAULT_CONFIG,
                "poly_deg": 4  # or 7
            }

            ls_result = fit_and_forecast(data, current_tf, n_future, ls_cfg)
            draw_approximations(ax, data, current_tf, ls_result)

            # If there is no AI prediction, extend xlim to cover LS future
            if not prediction_data:
                ax.set_xlim(data.index[0], ls_result["future_times"][-1])

        except Exception as e:
            print(f"Approximations error: {e}")

    ax.set_ylim(y_min, y_max)
    ax.set_facecolor("#2b2b2f")
    fig.patch.set_facecolor("#2b2b2f")
    ax.spines["bottom"].set_color = "#dcdcdc"
    ax.spines["left"].set_color = "#dcdcdc"
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
    status_label.config(text=f"Current price: ${current_price:.2f}")


# main loop
root = tk.Tk()
root.title("BTC-Prediction")
root.geometry("1000x700")
root.configure(bg="#1f1f23")

title_frame = tk.Frame(root, bg="#1f1f23")
title_frame.pack(anchor="w", pady=10, padx=20)
tk.Label(title_frame, text="BTC", fg="#f38b00", bg="#1f1f23", font=("Shippori Antique B1", 14)).pack(side="left")
tk.Label(title_frame, text="- Prediction", fg="#cfcfcf", bg="#1f1f23", font=("Shippori Antique B1", 14)).pack(
    side="left")

toolbar = tk.Frame(root, bg="#1f1f23")
toolbar.pack(anchor="w", padx=20, pady=5, fill="x")


main_frame = tk.Frame(root, bg="#1f1f23")
main_frame.pack(padx=30, pady=20, fill="both", expand=True)

canvas_frame = tk.Frame(main_frame, bg="#2b2b2f", width=700, height=500)
canvas_frame.pack(side="left", fill="both", expand=True)

control_frame = tk.Frame(main_frame, bg="#1f1f23", width=200)
control_frame.pack(side="left", fill="y", padx=10, pady=10)

#Show approximations
show_approx_var = tk.BooleanVar(value=False)
approx_check = tk.Checkbutton(
    control_frame,
    text="Show approximations",
    variable=show_approx_var,
    onvalue=True,
    offvalue=False,
    command=lambda: update_plot(),
    bg="#1f1f23",
    fg="#ffffff",
    activebackground="#1f1f23",
    activeforeground="#ffffff",
    selectcolor="#333333",
    font=("Segoe UI", 9)
)
approx_check.pack(anchor="nw", pady=(0, 10))


#Prediction Length
slider_frame = tk.Frame(control_frame, bg="#1f1f23")
slider_frame.pack(anchor="nw", pady=(0, 15), fill="x")

tk.Label(slider_frame, text="Prediction Length (days):", fg="#ffffff", bg="#1f1f23",
         font=("Segoe UI", 9)).pack(anchor="w")

prediction_days_var = tk.IntVar(value=7)  # Default 7 days

days_slider = tk.Scale(
    slider_frame,
    from_=1,
    to=31,
    orient="horizontal",
    variable=prediction_days_var,
    length=153,
    sliderlength=15,
    showvalue=True,
    bg="#1f1f23",
    fg="#ffffff",
    troughcolor="#333333",
    highlightbackground="#1f1f23",
    highlightcolor="#1f1f23"
)
days_slider.pack(anchor="w", pady=(5, 0))

#Consider News
consider_news_var = tk.BooleanVar(value=True)
news_button = tk.Button(
    control_frame,
    text="Consider News: ON",
    bg="#00cc00",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    relief="flat",
    activebackground="#009900",
    activeforeground="white",
    padx=10,
    pady=5,
    width=16,
    command=lambda: toggle_news()
)
news_button.pack(anchor="nw", pady=(0, 10))

#Predict
predict_button = tk.Button(
    control_frame,
    text="Predict",
    bg="#f38b00",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    relief="flat",
    activebackground="#d67a00",
    activeforeground="white",
    padx=10,
    pady=5,
    width=16,
    command=predict_action
)
predict_button.pack(anchor="nw", pady=(0, 5))

tf_block = tk.Frame(toolbar, bg="#1f1f23")
tf_block.pack(side="left", padx=(0, 20))
tk.Label(tf_block, text="Timeframe", fg="#dcdcdc", bg="#1f1f23", font=("Segoe UI", 10)).pack(anchor="center",pady=(0, 3))

timeframes = {"1h": "1h", "4h": "4h", "1d": "1d", "5d": "5d", "1wk": "1wk", "1mo": "1mo"}
timeframe_frame = tk.Frame(tf_block, bg="#1f1f23")
timeframe_frame.pack(anchor="center")

for name, tf_code in timeframes.items():
    tk.Button(
        timeframe_frame, text=name, bg="#1f1f23", fg="white",
        relief="solid", bd=1, font=("Segoe UI", 9),
        activebackground="#333", activeforeground="#f38b00",
        width=4, command=lambda tf=tf_code: change_timeframe(tf)
    ).pack(side="left", padx=3, pady=(0, 2))

fig, ax = plt.subplots(facecolor="#2b2b2f", figsize=(9.5, 5.8))
canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

status_label = tk.Label(root, text="", bg="#1f1f23", fg="#dcdcdc", font=("Segoe UI", 10))
status_label.pack(pady=(0, 10))

fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("scroll_event", zoom)
fig.canvas.mpl_connect("button_press_event", on_press_left)
fig.canvas.mpl_connect("button_release_event", on_release_left)
fig.canvas.mpl_connect("motion_notify_event", on_move_left)

update_plot()
root.mainloop()