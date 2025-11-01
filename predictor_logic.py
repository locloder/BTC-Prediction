import requests
import json
import datetime
from datetime import datetime, timedelta
import os
import numpy as np
import re
import pandas as pd
import yfinance as yf

# --- Mocking for data collection/saving from crypto.py and news parsing ---


def save_btc_data_for_predictor(df, file_path):
    """Saves DataFrame data to JSON in the format TimeSeriesPredictor expects."""
    if df is None or df.empty:
        return False

    vector_value = df[['Close']].reset_index()
    vector_value.columns = ['time', 'price']
    vector_value['time'] = vector_value['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    vector_value['price'] = vector_value['price'].round(2)
    vector_price = vector_value.to_dict('records')

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vector_price, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"Error saving data for prediction: {e}")
        return False


def mock_load_news_data(news_file_path, reddit_enabled, twitter_enabled):
    """
    Mock function to simulate loading or collecting news data.
    In a real app, this would call your parsing.py logic.
    """
    # Create a mock news context based on checkboxes
    news_context = []
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if reddit_enabled:
        news_context.append({
            "time": current_time,
            "news": "Reddit sentiment: Increased interest in BTC derivatives. Price might trend up slightly."
        })
    if twitter_enabled:
        news_context.append({
            "time": current_time,
            "news": "Twitter trend: Major institution announced new Bitcoin custody service. Positive market signal."
        })

    # Load from file if available (mimicking the original logic)
    if news_file_path and os.path.exists(news_file_path):
        try:
            with open(news_file_path, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
            articles = news_data.get('articles', [])
            for article in articles[:5]:  # Take top 5 from file
                timestamp = article.get('data', '').replace('T', ' ').replace('Z', '')
                headline = article.get('headline', '')
                description = article.get('description', '')
                news_text = f"{headline}. {description}"
                if news_text.strip() and timestamp:
                    news_context.append({"time": timestamp, "news": news_text})
        except Exception as e:
            print(f"Error loading news file: {e}")

    # Keep only the last 10 relevant news items
    return news_context[-10:]


# --- TimeSeriesPredictor Class (Adapted from ai_requests.py) ---

class TimeSeriesPredictor:
    def __init__(self, json_file_path, news_context, time_gap_hours=None):
        self.news_context = news_context
        self.price_history = []
        self.json_file_path = json_file_path
        self.load_price_data()
        self.calculate_time_gap()

    def load_price_data(self):
        """Load price data from JSON file"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.price_history = data
        except Exception as e:
            print(f"Error loading price data: {e}")
            self.price_history = []

    def calculate_time_gap(self):
        """Calculate the typical time gap between data points"""
        if len(self.price_history) < 2:
            self.time_gap_hours = 1
            return

        try:
            time_diffs = []
            for i in range(1, min(10, len(self.price_history))):
                time1 = datetime.strptime(self.price_history[i - 1]['time'], '%Y-%m-%d %H:%M:%S')
                time2 = datetime.strptime(self.price_history[i]['time'], '%Y-%m-%d %H:%M:%S')
                diff_hours = (time2 - time1).total_seconds() / 3600
                time_diffs.append(diff_hours)

            self.time_gap_hours = np.median(time_diffs)
        except Exception as e:
            print(f"Error calculating time gap: {e}. Using default 1 hour.")
            self.time_gap_hours = 1

    def is_reasonable_change(self, new_price, previous_price, max_percent_change=15):
        """Check if the price change is reasonable"""
        if previous_price == 0:
            return True
        percent_change = abs((new_price - previous_price) / previous_price) * 100
        return percent_change <= max_percent_change

    def create_prediction_prompt(self, num_predictions=5):
        """Create a structured prompt for prediction"""

        if self.time_gap_hours == 1:
            time_gap_desc = "1 hour"
        elif self.time_gap_hours == 24:
            time_gap_desc = "1 day"
        else:
            time_gap_desc = f"{self.time_gap_hours:.1f} hours"

        news_context_formatted = "RECENT CRYPTO/NEWS CONTEXT:\n"
        if self.news_context:
            for i, news_item in enumerate(self.news_context[:]):
                news_context_formatted += f"{i + 1}. [{news_item['time']}] {news_item['news']}\n"
        else:
            news_context_formatted = "No recent news context available."

        prompt = f"""You are a financial prediction AI specializing in cryptocurrency markets. Analyze the given price data and news context, then predict exactly {num_predictions} future price points.

CURRENT BITCOIN PRICE DATA (most recent first):
{json.dumps(self.price_history[-10:], indent=2)}

TIME INTERVAL INFORMATION:
- Time gap between data points: {time_gap_desc}
- Predictions should maintain this same time interval

{news_context_formatted}

PREDICTION REQUIREMENTS:
1. Output ONLY a JSON array with exactly {num_predictions} numbers
2. Each number should be a logical continuation of the price trend
3. STRONGLY CONSIDER the news sentiment and market implications in your predictions
4. Do NOT include any explanations, text, or other content
5. Output should be written in same style as sample data
6. You MUST output exactly {num_predictions} price predictions, no fewer

PREDICTION:"""
        return prompt

    def filter_reasonable_predictions(self, predictions, last_known_price, existing_predictions=[]):
        """Filter predictions to only include those with reasonable price changes"""
        if not predictions:
            return []

        reasonable_predictions = []
        reference_price = last_known_price
        if existing_predictions:
            reference_price = existing_predictions[-1]

        if reference_price is None:
            return predictions

        for pred in predictions:
            if self.is_reasonable_change(pred, reference_price):
                reasonable_predictions.append(pred)
                reference_price = pred
            else:
                break
        return reasonable_predictions

    def parse_prediction(self, response_text, num_predictions=5):
        """Extract JSON array or numbers from response"""
        predictions = []
        try:
            start = response_text.find('[')
            end = response_text.find(']') + 1
            if start != -1 and end != 0:
                json_str = response_text[start:end]
                predictions = json.loads(json_str)
                if isinstance(predictions, list) and all(isinstance(p, (int, float)) for p in predictions):
                    return predictions[:num_predictions]
        except:
            pass

        numbers = re.findall(r'[\d,]+\.?\d*', response_text)
        for pattern in numbers:
            if len(predictions) >= num_predictions: break
            try:
                clean_num = float(pattern.replace(',', ''))
                predictions.append(clean_num)
            except:
                pass

        return predictions[:num_predictions]

    def generate_supplemental_predictions(self, num_needed, existing_predictions, last_known_price):
        """Generate supplemental predictions with reasonable changes"""
        if existing_predictions:
            reference_sequence = existing_predictions
            last_price = existing_predictions[-1]
        elif self.price_history:
            reference_sequence = [item['price'] for item in self.price_history[-5:]]
            last_price = self.price_history[-1]['price']
        else:
            return [50000.0 + i * 100 for i in range(num_needed)]

        if len(reference_sequence) >= 2:
            trend = sum(
                reference_sequence[i] - reference_sequence[i - 1] for i in range(1, len(reference_sequence))) / (
                                len(reference_sequence) - 1)
        else:
            trend = last_price * 0.01

        max_allowed_change = last_price * 0.05
        trend = max(min(trend, max_allowed_change), -max_allowed_change)

        supplemental = []
        current_price = last_price
        for i in range(num_needed):
            next_price = current_price + trend + np.random.normal(0, current_price * 0.005)  # Add small noise
            if not self.is_reasonable_change(next_price, current_price, max_percent_change=10):
                max_change = current_price * 0.1
                next_price = current_price + max_change * np.sign(trend)
            next_price = max(0, next_price)
            supplemental.append(next_price)
            current_price = next_price

        return supplemental

    def predict(self, num_predictions=5):
        """Get prediction from Ollama and supplement with generated predictions if needed"""
        max_attempts = 3
        attempts = 0
        all_predictions = []
        last_known_price = self.price_history[-1]['price'] if self.price_history else None

        while attempts < max_attempts and len(all_predictions) < num_predictions:
            predictions_needed = num_predictions - len(all_predictions)
            prompt = self.create_prediction_prompt(predictions_needed)

            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama2",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1}
                    },
                    timeout=45
                )

                result = response.json()
                new_predictions = self.parse_prediction(result["response"], predictions_needed)
                filtered_predictions = self.filter_reasonable_predictions(new_predictions, last_known_price,
                                                                          all_predictions)
                all_predictions.extend(filtered_predictions)
                attempts += 1

                if len(filtered_predictions) == 0:
                    import time;
                    time.sleep(1)

            except requests.exceptions.Timeout:
                attempts += 1
            except Exception as e:
                attempts += 1

        if len(all_predictions) < num_predictions:
            remaining = num_predictions - len(all_predictions)
            supplemental_predictions = self.generate_supplemental_predictions(remaining, all_predictions,
                                                                              last_known_price)
            all_predictions.extend(supplemental_predictions)

        return all_predictions[:num_predictions]

    def generate_future_dates(self, num_predictions):
        """Generate future dates for predictions using the detected time gap"""
        if not self.price_history:
            return []

        last_timestamp = datetime.strptime(self.price_history[-1]['time'], '%Y-%m-%d %H:%M:%S')
        future_dates = []
        for i in range(1, num_predictions + 1):
            future_date = last_timestamp + timedelta(hours=self.time_gap_hours * i)
            future_dates.append(future_date.strftime('%Y-%m-%d %H:%M:%S'))

        return future_dates