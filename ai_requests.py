import requests
import json
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np


class TimeSeriesPredictor:
    def __init__(self, json_file_path="btc_prices.json", news_file_path=None):
        self.news_context = []
        self.price_history = []
        self.json_file_path = json_file_path
        self.news_file_path = news_file_path
        self.time_gap_hours = None  # Will be calculated from data
        self.load_price_data()
        self.load_news_data()
        self.calculate_time_gap()

    def load_price_data(self):
        """Load price data from JSON file"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.price_history = data
                print(f"Loaded {len(self.price_history)} price data points from {self.json_file_path}")
        except Exception as e:
            print(f"Error loading price data: {e}")
            self.price_history = []

    def load_news_data(self):
        """Load news data from JSON file and add to context"""
        if not self.news_file_path:
            print("No news file path provided. Skipping news loading.")
            return

        try:
            with open(self.news_file_path, 'r', encoding='utf-8') as f:
                news_data = json.load(f)

            # Extract articles from the news data
            articles = news_data.get('articles', [])
            print(f"Loaded {len(articles)} news articles from {self.news_file_path}")

            # Add each article to the news context
            for article in articles:
                headline = article.get('headline', '')
                description = article.get('description', '')
                news_text = f"{headline}. {description}"
                timestamp = article.get('data', '').replace('T', ' ').replace('Z', '')

                if news_text.strip() and timestamp:
                    self.add_news(news_text, timestamp)

        except Exception as e:
            print(f"Error loading news data: {e}")

    def calculate_time_gap(self):
        """Calculate the typical time gap between data points"""
        if len(self.price_history) < 2:
            print("Not enough data points to calculate time gap. Using default 1 hour.")
            self.time_gap_hours = 1
            return

        try:
            time_diffs = []
            for i in range(1, min(10, len(self.price_history))):  # Check first 10 gaps
                time1 = datetime.strptime(self.price_history[i - 1]['time'], '%Y-%m-%d %H:%M:%S')
                time2 = datetime.strptime(self.price_history[i]['time'], '%Y-%m-%d %H:%M:%S')
                diff_hours = (time2 - time1).total_seconds() / 3600
                time_diffs.append(diff_hours)

            # Use the most common time gap
            self.time_gap_hours = np.median(time_diffs)
            print(f"Detected time gap: {self.time_gap_hours:.1f} hours between data points")

        except Exception as e:
            print(f"Error calculating time gap: {e}. Using default 1 hour.")
            self.time_gap_hours = 1

    def add_news(self, news_text, timestamp):
        """Add news data to context"""
        self.news_context.append({
            "time": timestamp,
            "news": news_text
        })
        # Keep only recent news (last 24 hours or last 10 items)
        self.news_context = self.news_context[-10:]  # Keep last 10 news items

    def add_price_data(self, price_data):
        """Add price data points"""
        self.price_history.append(price_data)
        # Keep reasonable history
        self.price_history = self.price_history[-1000:]  # Keep last 1000 price points

    def is_reasonable_change(self, new_price, previous_price, max_percent_change=15):
        """Check if the price change is reasonable (within max_percent_change%)"""
        if previous_price == 0:
            return True

        percent_change = abs((new_price - previous_price) / previous_price) * 100
        return percent_change <= max_percent_change

    def create_prediction_prompt(self, num_predictions=5):
        """Create a structured prompt for prediction"""

        # Calculate time gap description for the prompt
        if self.time_gap_hours == 1:
            time_gap_desc = "1 hour"
        elif self.time_gap_hours == 24:
            time_gap_desc = "1 day"
        else:
            time_gap_desc = f"{self.time_gap_hours:.1f} hours"

        # Format news context for better readability
        news_context_formatted = ""
        if self.news_context:
            news_context_formatted = "RECENT CRYPTO/NEWS CONTEXT:\n"
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
4. Positive news (adoption, institutional interest) should generally support higher prices
5. Negative news (regulations, economic concerns) should generally pressure prices lower
6. Do NOT include any explanations, text, or other content
7. Output should be written in same style as sample data
8. Price changes should be reasonable (typically less than 15% change between consecutive points)
9. You MUST output exactly {num_predictions} price predictions, no fewer
10. The time interval between your predictions should match the historical data interval of {time_gap_desc}

PREDICTION:"""
        return prompt

    def predict(self, num_predictions=5):
        """Get prediction from Ollama and supplement with generated predictions if needed"""
        max_attempts = 5  # Increased attempts due to filtering
        attempts = 0
        all_predictions = []

        # Get the last known price for change validation
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
                        "options": {
                            "temperature": 0.1,  # Low temp for consistent formatting
                            "num_predict": 150  # Limit output length
                        }
                    },
                    timeout=30  # Add timeout
                )

                result = response.json()
                new_predictions = self.parse_prediction(result["response"], predictions_needed)

                # Filter predictions for reasonable changes
                filtered_predictions = self.filter_reasonable_predictions(new_predictions, last_known_price,
                                                                          all_predictions)

                # Add the filtered predictions to our collection
                all_predictions.extend(filtered_predictions)

                attempts += 1
                print(
                    f"Attempt {attempts}: Got {len(new_predictions)} predictions, kept {len(filtered_predictions)} reasonable ones. Total so far: {len(all_predictions)}")

                # If we got no reasonable predictions in this attempt, wait a bit before retry
                if len(filtered_predictions) == 0:
                    print("No reasonable predictions in this attempt, waiting before retry...")
                    import time
                    time.sleep(1)

            except requests.exceptions.Timeout:
                print(f"Attempt {attempts + 1}: Request timeout")
                attempts += 1
            except Exception as e:
                print(f"Attempt {attempts + 1}: Error - {e}")
                attempts += 1

        # If we still don't have enough predictions, generate the remaining ones
        if len(all_predictions) < num_predictions:
            remaining = num_predictions - len(all_predictions)
            print(f"Generating {remaining} additional reasonable predictions to reach {num_predictions} total...")
            supplemental_predictions = self.generate_supplemental_predictions(remaining, all_predictions,
                                                                              last_known_price)
            all_predictions.extend(supplemental_predictions)

        return all_predictions[:num_predictions]

    def filter_reasonable_predictions(self, predictions, last_known_price, existing_predictions=[]):
        """Filter predictions to only include those with reasonable price changes"""
        if not predictions:
            return []

        reasonable_predictions = []
        reference_price = last_known_price

        # If we have existing predictions, use the last one as reference
        if existing_predictions:
            reference_price = existing_predictions[-1]

        if reference_price is None:
            # If no reference price, accept all (shouldn't happen normally)
            return predictions

        for i, pred in enumerate(predictions):
            # Check if this prediction is reasonable compared to reference
            if self.is_reasonable_change(pred, reference_price):
                reasonable_predictions.append(pred)
                # Update reference for next prediction
                reference_price = pred
            else:
                break

        return reasonable_predictions

    def parse_prediction(self, response_text, num_predictions=5):
        """Extract JSON array from response with enhanced parsing"""
        predictions = []

        # Method 1: Try to parse JSON array first
        try:
            # Find JSON array in response
            start = response_text.find('[')
            end = response_text.find(']') + 1
            if start != -1 and end != 0:
                json_str = response_text[start:end]
                predictions = json.loads(json_str)
                if len(predictions) >= num_predictions:
                    return predictions[:num_predictions]
        except:
            pass

        # Method 2: Try to find all numbers in the response
        import re
        numbers = re.findall(r'\d+\.?\d*', response_text)
        if numbers:
            predictions = [float(num) for num in numbers]
            if len(predictions) >= num_predictions:
                return predictions[:num_predictions]

        # Method 3: If we still don't have enough, look for number patterns
        # This handles cases where numbers might be formatted with commas
        number_patterns = re.findall(r'[\d,]+\.?\d*', response_text)
        for pattern in number_patterns:
            if len(predictions) >= num_predictions:
                break
            try:
                # Remove commas and convert to float
                clean_num = float(pattern.replace(',', ''))
                predictions.append(clean_num)
            except:
                continue

        return predictions[:num_predictions]

    def generate_supplemental_predictions(self, num_needed, existing_predictions, last_known_price):
        """Generate supplemental predictions with reasonable changes"""
        if not self.price_history and not existing_predictions:
            # If no data at all, create simple incremental predictions
            return [50000.0 + i * 100 for i in range(num_needed)]

        # Determine starting point and reference for trend calculation
        if existing_predictions:
            reference_sequence = existing_predictions
            last_price = existing_predictions[-1]
        else:
            reference_sequence = [item['price'] for item in self.price_history[-5:]]
            last_price = self.price_history[-1]['price'] if self.price_history else 50000.0

        # Calculate trend from reference sequence
        if len(reference_sequence) >= 2:
            trend = sum(
                reference_sequence[i] - reference_sequence[i - 1] for i in range(1, len(reference_sequence))) / (
                            len(reference_sequence) - 1)
        else:
            # Small conservative trend if not enough data
            trend = last_price * 0.01  # 1% of last price

        # Ensure trend is reasonable (max 5% change per step)
        max_allowed_change = last_price * 0.05
        trend = max(min(trend, max_allowed_change), -max_allowed_change)

        # Generate supplemental predictions with reasonable changes
        supplemental = []
        current_price = last_price

        for i in range(num_needed):
            next_price = current_price + trend
            # Add small random variation but ensure it's reasonable

            # Ensure the change is within reasonable bounds
            if not self.is_reasonable_change(next_price, current_price, max_percent_change=10):
                # If not reasonable, cap the change
                max_change = current_price * 0.1  # 10% max change
                if next_price > current_price:
                    next_price = current_price + max_change
                else:
                    next_price = current_price - max_change

            # Ensure non-negative
            next_price = max(0, next_price)
            supplemental.append(next_price)
            current_price = next_price

        return supplemental

    def generate_future_dates(self, num_predictions):
        """Generate future dates for predictions using the detected time gap"""
        if not self.price_history:
            return []

        # Get the last timestamp from historical data
        last_timestamp = datetime.strptime(self.price_history[-1]['time'], '%Y-%m-%d %H:%M:%S')

        # Generate future dates using the detected time gap
        future_dates = []
        for i in range(1, num_predictions + 1):
            future_date = last_timestamp + timedelta(hours=self.time_gap_hours * i)
            future_dates.append(future_date.strftime('%Y-%m-%d %H:%M:%S'))

        return future_dates

    def plot_predictions(self, predictions, num_predictions=5):
        """Plot historical data and predictions on the same graph"""
        if not self.price_history:
            print("No historical data to plot")
            return

        # Prepare historical data
        historical_times = [datetime.strptime(item['time'], '%Y-%m-%d %H:%M:%S') for item in self.price_history]
        historical_prices = [item['price'] for item in self.price_history]

        # Prepare prediction data
        future_times = self.generate_future_dates(num_predictions)
        future_times_dt = [datetime.strptime(time, '%Y-%m-%d %H:%M:%S') for time in future_times]

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot historical data in blue
        plt.plot(historical_times, historical_prices,
                 color='blue', linewidth=2, marker='o', markersize=3,
                 label='Historical Data')

        # Plot predictions in red
        plt.plot(future_times_dt, predictions,
                 color='red', linewidth=2, marker='s', markersize=4,
                 label='Predictions')

        # Add a vertical line to separate historical data from predictions
        last_historical_time = historical_times[-1]
        plt.axvline(x=last_historical_time, color='gray', linestyle='--', alpha=0.7, label='Prediction Start')

        plt.title(f'Bitcoin price prediction', fontsize=14,
                  fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show the plot
        plt.show()

        # Also print the data for reference
        print("\n" + "=" * 60)
        print(f"TIME INTERVAL: {self.time_gap_hours:.1f} hours between data points")
        print(f"NEWS CONTEXT: {len(self.news_context)} recent news items considered")
        print("HISTORICAL DATA (Last 5 points):")
        for i, item in enumerate(self.price_history[-5:]):
            print(f"  {item['time']}: ${item['price']:,.2f}")

        print("\nRECENT NEWS (affecting predictions):")
        for i, news in enumerate(self.news_context[-3:]):  # Show last 3 news items
            print(f"  {i + 1}. {news['news'][:100]}...")

        print("\nPREDICTIONS:")
        for i, (time, price) in enumerate(zip(future_times, predictions)):
            change_from_previous = ""
            if i == 0:
                last_hist_price = self.price_history[-1]['price']
                change_pct = (price - last_hist_price) / last_hist_price * 100
                change_from_previous = f" ({change_pct:+.1f}% from historical)"
            elif i > 0:
                change_pct = (price - predictions[i - 1]) / predictions[i - 1] * 100
                change_from_previous = f" ({change_pct:+.1f}% from previous)"

            print(f"  {time}: ${price:,.2f}{change_from_previous}")
        print("=" * 60)


current_dir = os.path.dirname(os.path.abspath(__file__))
price_json_path = os.path.join(current_dir, "parsing", "btc_prices.json")
news_json_path = os.path.join(current_dir,"parsing",  "news.json")  # Path to your news.json file

# Initialize predictor with both price and news data
predictor = TimeSeriesPredictor(price_json_path, news_json_path)

# Set number of predictions to match the number of historical data points loaded
num_predictions = len(predictor.price_history)
print(f"Loaded {num_predictions} historical data points, making {num_predictions} predictions")

# Get prediction - now it will consider both price trends and news sentiment
predictions = predictor.predict(num_predictions)
print(f"Final predictions ({len(predictions)}):", predictions)

# Plot the results
if predictions and len(predictions) >= num_predictions:
    predictor.plot_predictions(predictions, num_predictions)
else:
    print(f"Failed to generate {num_predictions} predictions. Got {len(predictions)} instead.")