import requests
import json
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
import re
import time
import math


class TimeSeriesPredictor:
    def __init__(self, json_file_path="data/btc_prices.json", news_file_path=None):
        self.news_context = []
        self.price_history = []
        self.json_file_path = json_file_path
        self.news_file_path = news_file_path
        self.time_gap_hours = None
        self.load_price_data()
        if news_file_path:
            self.load_news_data()
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

    def load_news_data(self):
        """Load news data from JSON file and add to context"""
        if not self.news_file_path:
            return

        try:
            with open(self.news_file_path, 'r', encoding='utf-8') as f:
                news_data = json.load(f)

            articles = news_data.get('articles', [])
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
            self.time_gap_hours = 24  # Default to 1 day
            return

        try:
            time_diffs = []
            for i in range(1, min(10, len(self.price_history))):
                time1 = datetime.strptime(self.price_history[i - 1]['time'], '%Y-%m-%d %H:%M:%S')
                time2 = datetime.strptime(self.price_history[i]['time'], '%Y-%m-%d %H:%M:%S')
                diff_hours = (time2 - time1).total_seconds() / 3600
                time_diffs.append(diff_hours)

            self.time_gap_hours = np.median(time_diffs)

            # If we detect very large gaps (like weekly or monthly), adjust accordingly
            if self.time_gap_hours > 24 * 20:  # More than 20 days gap
                self.time_gap_hours = 24 * 30  # Assume monthly data
            elif self.time_gap_hours > 24 * 5:  # More than 5 days gap
                self.time_gap_hours = 24 * 7  # Assume weekly data
            elif self.time_gap_hours > 12:  # More than 12 hours gap
                self.time_gap_hours = 24  # Assume daily data

        except Exception as e:
            print(f"Error calculating time gap: {e}. Using default 24 hours.")
            self.time_gap_hours = 24

    def add_news(self, news_text, timestamp):
        """Add news data to context"""
        self.news_context.append({
            "time": timestamp,
            "news": news_text
        })
        self.news_context = self.news_context[-10:]

    def calculate_realistic_limits(self, num_predictions, reference_price):
        """Calculate realistic price limits based on prediction count and timeframe"""
        # Maximum realistic growth percentages based on timeframe
        # These are ANNUALIZED percentages scaled down for the prediction period

        # Convert prediction period to approximate days
        total_hours = num_predictions * self.time_gap_hours
        total_days = total_hours / 24

        # Annual maximum realistic movement (very optimistic/bearish scenarios)
        max_annual_growth = 300  # 300% per year maximum in extreme bull market
        max_annual_decline = 80  # 80% per year maximum in extreme bear market

        # Scale to our prediction period (square root time scaling for volatility)
        time_scale = math.sqrt(total_days / 365)

        max_growth_pct = max_annual_growth * time_scale
        max_decline_pct = max_annual_decline * time_scale

        upper_limit = reference_price * (1 + max_growth_pct / 100)
        lower_limit = reference_price * (1 - max_decline_pct / 100)

        # Absolute limits (never exceed these)
        absolute_upper = 1000000  # $1M
        absolute_lower = 100  # $100

        upper_limit = min(upper_limit, absolute_upper)
        lower_limit = max(lower_limit, absolute_lower)

        print(f"Realistic limits for {num_predictions} predictions ({total_days:.1f} days):")
        print(f"  Reference: ${reference_price:.2f}")
        print(f"  Upper limit: ${upper_limit:.2f} ({max_growth_pct:.1f}% max growth)")
        print(f"  Lower limit: ${lower_limit:.2f} ({max_decline_pct:.1f}% max decline)")

        return lower_limit, upper_limit

    def is_reasonable_price(self, price, previous_price, num_predictions, reference_price, position=None):
        """Check if the price is reasonable and legitimate with enhanced validation"""
        # Calculate realistic limits for the entire prediction sequence
        lower_limit, upper_limit = self.calculate_realistic_limits(num_predictions, reference_price)

        # Check absolute price bounds
        if price <= 0:
            return False, "Price must be positive"
        if price > upper_limit:
            return False, f"Price exceeds realistic maximum ${upper_limit:.2f}"
        if price < lower_limit:
            return False, f"Price below realistic minimum ${lower_limit:.2f}"

        # Check for reasonable change from previous price (consecutive points)
        if previous_price is not None and previous_price > 0:
            percent_change = abs((price - previous_price) / previous_price) * 100

            # Dynamic max change based on timeframe - longer timeframes allow larger moves
            base_max_change = 15  # Base maximum change per period
            # Longer timeframes allow much larger moves (weekly/monthly can have bigger swings)
            volatility_factor = min(5.0, 1.0 + (self.time_gap_hours / 24) * 0.5)
            max_consecutive_change = base_max_change * volatility_factor

            if percent_change > max_consecutive_change:
                return False, f"Price change too large: {percent_change:.1f}% (max: {max_consecutive_change:.1f}%)"

        # Check for unrealistic acceleration (if we have position context)
        if position is not None and previous_price is not None and position > 5:
            # Calculate if this point shows unrealistic acceleration in trend
            avg_growth = (price - reference_price) / reference_price * 100
            time_factor = (position + 1) / num_predictions  # How far we are in the sequence

            # Maximum reasonable cumulative growth at this point in the sequence
            max_cumulative_growth = 100 * (math.pow(
                1 + self.calculate_realistic_limits(num_predictions, reference_price)[1] / reference_price - 1,
                time_factor) - 1)

            if abs(avg_growth) > abs(max_cumulative_growth) * 1.5:  # 50% buffer
                return False, f"Unrealistic cumulative growth: {avg_growth:.1f}% at position {position}"

        return True, "Valid price"

    def validate_predictions_sequence(self, predictions, reference_price):
        """Validate a sequence of predictions and identify invalid ranges with enhanced checks"""
        valid_predictions = []
        invalid_ranges = []  # List of (start_idx, end_idx, before_price, after_price)
        current_invalid_start = None
        last_valid_price = reference_price
        num_predictions = len(predictions)

        for i, pred in enumerate(predictions):
            is_valid, error_msg = self.is_reasonable_price(
                pred, last_valid_price, num_predictions, reference_price, i
            )

            if is_valid:
                valid_predictions.append(pred)
                last_valid_price = pred

                # Close any open invalid range
                if current_invalid_start is not None:
                    invalid_ranges.append((current_invalid_start, i - 1,
                                           valid_predictions[
                                               current_invalid_start - 1] if current_invalid_start > 0 else reference_price,
                                           pred))
                    current_invalid_start = None
            else:
                print(f"Invalid prediction at position {i + 1}: {error_msg} | Price: ${pred:.2f}")
                # Start or continue an invalid range
                if current_invalid_start is None:
                    current_invalid_start = i

                # Placeholder for invalid prediction
                valid_predictions.append(None)

        # Close any remaining open invalid range at the end
        if current_invalid_start is not None:
            invalid_ranges.append((current_invalid_start, len(predictions) - 1,
                                   valid_predictions[
                                       current_invalid_start - 1] if current_invalid_start > 0 else reference_price,
                                   None))

        return valid_predictions, invalid_ranges

    def create_initial_prediction_prompt(self, num_predictions=100):
        """Create prompt for initial batch prediction with realistic growth guidance"""
        # Convert time gap to human-readable description
        if self.time_gap_hours == 1:
            time_gap_desc = "1 hour"
        elif self.time_gap_hours == 4:
            time_gap_desc = "4 hours"
        elif self.time_gap_hours == 24:
            time_gap_desc = "1 day"
        elif self.time_gap_hours == 24 * 5:
            time_gap_desc = "5 days"
        elif self.time_gap_hours == 24 * 7:
            time_gap_desc = "1 week"
        elif self.time_gap_hours == 24 * 30:
            time_gap_desc = "1 month"
        else:
            time_gap_desc = f"{self.time_gap_hours:.1f} hours"

        # Calculate realistic growth limits for context
        reference_price = self.price_history[-1]['price'] if self.price_history else 50000.0
        lower_limit, upper_limit = self.calculate_realistic_limits(num_predictions, reference_price)
        max_total_growth = (upper_limit / reference_price - 1) * 100
        max_total_decline = (1 - lower_limit / reference_price) * 100

        news_context_formatted = ""
        if self.news_context:
            news_context_formatted = "RECENT CRYPTO/NEWS CONTEXT:\n"
            for i, news_item in enumerate(self.news_context[:]):
                news_context_formatted += f"{i + 1}. [{news_item['time']}] {news_item['news']}\n"
        else:
            news_context_formatted = "No recent news context available."

        # Calculate total prediction period
        total_hours = num_predictions * self.time_gap_hours
        total_days = total_hours / 24

        if total_days >= 30:
            period_desc = f"{total_days / 30:.1f} months"
        elif total_days >= 7:
            period_desc = f"{total_days / 7:.1f} weeks"
        else:
            period_desc = f"{total_days:.1f} days"

        prompt = f"""You are a financial prediction AI specializing in cryptocurrency markets. Analyze the given price data and news context, then predict exactly {num_predictions} future price points.

            CURRENT BITCOIN PRICE DATA (most recent first):
            {json.dumps(self.price_history[-10:], indent=2)}
            
            TIME INTERVAL INFORMATION:
            - Time gap between data points: {time_gap_desc}
            - Total prediction period: {num_predictions} intervals ({period_desc})
            - Predictions should maintain this same time interval
            
            REALISTIC GROWTH CONSTRAINTS:
            - Current price: ${reference_price:.2f}
            - Maximum realistic target: ${upper_limit:.2f} ({max_total_growth:.1f}% total growth)
            - Minimum realistic target: ${lower_limit:.2f} ({max_total_decline:.1f}% total decline)
            - Price changes should be SMOOTH and REALISTIC
            - For {time_gap_desc.lower()} intervals, consider appropriate volatility levels
            - Weekly/Monthly predictions can have larger swings than hourly/daily predictions
            
            {news_context_formatted}
            
            PREDICTION REQUIREMENTS:
            1. Output ONLY a JSON array with exactly {num_predictions} numbers
            2. Each number should be a logical continuation of the price trend
            3. STRONGLY CONSIDER the news sentiment and market implications
            4. Positive news should generally support higher prices, but within realistic bounds
            5. Negative news should generally pressure prices lower, but within realistic bounds
            6. Do NOT include any explanations, text, or other content
            7. Price changes between consecutive points should be reasonable for the timeframe
            8. The overall trend should stay within realistic growth limits
            9. You MUST output exactly {num_predictions} price predictions, no fewer
            10. ALL prices must be POSITIVE numbers between ${lower_limit:.2f} and ${upper_limit:.2f}
            11. Remember: This is for Bitcoin price prediction - be realistic about volatility
            12. For {time_gap_desc.lower()} intervals, adjust your volatility expectations accordingly
            
            PREDICTION:"""
        return prompt

    def create_gap_patching_prompt(self, gap_info, num_predictions, reference_price):
        """Create prompt for patching specific gaps in predictions with realistic constraints"""
        start_idx, end_idx, before_price, after_price, gap_length = gap_info

        # Calculate realistic limits for this gap
        gap_lower_limit, gap_upper_limit = self.calculate_realistic_limits(num_predictions, reference_price)

        # Convert time gap to human-readable description
        if self.time_gap_hours == 1:
            time_gap_desc = "1 hour"
        elif self.time_gap_hours == 4:
            time_gap_desc = "4 hours"
        elif self.time_gap_hours == 24:
            time_gap_desc = "1 day"
        elif self.time_gap_hours == 24 * 5:
            time_gap_desc = "5 days"
        elif self.time_gap_hours == 24 * 7:
            time_gap_desc = "1 week"
        elif self.time_gap_hours == 24 * 30:
            time_gap_desc = "1 month"
        else:
            time_gap_desc = f"{self.time_gap_hours:.1f} hours"

        prompt = f"""You are a financial prediction AI specializing in cryptocurrency markets. You need to patch a gap in Bitcoin price predictions.

                CONTEXT:
                - There is a gap in predictions from position {start_idx + 1} to {end_idx + 1}
                - Price before the gap: ${before_price:.2f}
                - Price after the gap: ${after_price:.2f if after_price else 'N/A'}
                - Gap length: {gap_length} time periods
                - Time interval between predictions: {time_gap_desc}
                - Realistic price range: ${gap_lower_limit:.2f} to ${gap_upper_limit:.2f}
                
                REQUIREMENTS:
                1. Output ONLY a JSON array with exactly {gap_length} numbers
                2. These numbers should form a smooth transition from ${before_price:.2f} to ${after_price:.2f if after_price else 'a realistic continuation'}
                3. Each prediction should be a logical step in the price progression
                4. Price changes between consecutive points should be reasonable for {time_gap_desc} intervals
                5. ALL prices must be POSITIVE numbers between ${gap_lower_limit:.2f} and ${gap_upper_limit:.2f}
                6. Do NOT include any explanations, text, or other content
                7. The transition should be smooth and realistic for Bitcoin price movement
                8. If no target price is given, continue the trend realistically
                9. Consider that {time_gap_desc} intervals allow for different volatility than shorter timeframes
                
                GAP PATCH PREDICTION:"""
        return prompt

    def make_prediction_request(self, prompt, timeout=45):
        """Make a single prediction request to Ollama"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 500
                    }
                },
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result["response"], True
            else:
                return f"HTTP error {response.status_code}", False

        except requests.exceptions.Timeout:
            return "Request timeout", False
        except Exception as e:
            return f"Request error: {str(e)}", False

    def patch_prediction_gaps(self, predictions, invalid_ranges, reference_price):
        """Patch invalid ranges in predictions using AI"""
        patched_predictions = predictions.copy()
        num_predictions = len(predictions)

        for gap_info in invalid_ranges:
            start_idx, end_idx, before_price, after_price = gap_info
            gap_length = end_idx - start_idx + 1

            print(f"Patching gap {start_idx + 1}-{end_idx + 1} (length: {gap_length})...")

            # If no after_price (gap at the end), estimate a reasonable target
            if after_price is None:
                # Use realistic growth projection based on timeframe
                total_periods = gap_length
                if self.time_gap_hours >= 24 * 7:  # Weekly or monthly
                    realistic_period_growth = 1.10  # 10% per period maximum for longer timeframes
                elif self.time_gap_hours >= 24:  # Daily or 5-day
                    realistic_period_growth = 1.05  # 5% per period maximum
                else:  # Hourly
                    realistic_period_growth = 1.02  # 2% per period maximum

                after_price = before_price * (realistic_period_growth ** gap_length)

            gap_info_with_length = (start_idx, end_idx, before_price, after_price, gap_length)
            prompt = self.create_gap_patching_prompt(gap_info_with_length, num_predictions, reference_price)

            response_text, success = self.make_prediction_request(prompt, timeout=30)

            if success:
                patch_predictions = self.parse_prediction(response_text, gap_length)

                if patch_predictions and len(patch_predictions) == gap_length:
                    # Validate the patch with our enhanced checks
                    patch_valid = True
                    test_price = before_price

                    for i, patch_pred in enumerate(patch_predictions):
                        is_valid, error_msg = self.is_reasonable_price(patch_pred, test_price, num_predictions,
                                                                       reference_price, start_idx + i)
                        if not is_valid:
                            print(f"Patch validation failed at position {start_idx + i + 1}: {error_msg}")
                            patch_valid = False
                            break
                        test_price = patch_pred

                    # Also check if the patch connects well to after_price
                    if patch_valid and after_price is not None:
                        final_gap_change = abs(patch_predictions[-1] - after_price) / after_price
                        if final_gap_change > 0.15:
                            patch_valid = False
                            print(f"Patch doesn't connect well to target: {final_gap_change:.1%} difference")

                    if patch_valid:
                        # Apply the patch
                        for i in range(gap_length):
                            patched_predictions[start_idx + i] = patch_predictions[i]
                        print(f"Successfully patched gap {start_idx + 1}-{end_idx + 1}")
                    else:
                        print(f"Patch for gap {start_idx + 1}-{end_idx + 1} failed validation")
                        # Don't set to None - use linear interpolation instead
                        self.apply_linear_interpolation(patched_predictions, start_idx, end_idx, before_price,
                                                        after_price)
                else:
                    print(f"Failed to parse patch for gap {start_idx + 1}-{end_idx + 1}")
                    # Use linear interpolation as fallback
                    self.apply_linear_interpolation(patched_predictions, start_idx, end_idx, before_price, after_price)
            else:
                print(f"Request failed for gap {start_idx + 1}-{end_idx + 1}: {response_text}")
                # Use linear interpolation as fallback
                self.apply_linear_interpolation(patched_predictions, start_idx, end_idx, before_price, after_price)

            time.sleep(0.1)

        return patched_predictions

    def apply_linear_interpolation(self, predictions, start_idx, end_idx, before_price, after_price):
        """Apply linear interpolation as a fallback for failed gap patching"""
        gap_length = end_idx - start_idx + 1

        if after_price is not None:
            # Linear interpolation between before_price and after_price
            price_step = (after_price - before_price) / (gap_length + 1)
            for i in range(gap_length):
                predictions[start_idx + i] = before_price + (price_step * (i + 1))
            print(f"Applied linear interpolation for gap {start_idx + 1}-{end_idx + 1}")
        else:
            # Just continue with slight growth if no target price
            # Adjust growth rate based on timeframe
            if self.time_gap_hours >= 24 * 7:  # Weekly or monthly
                growth_rate = 1.005  # 0.5% per period
            elif self.time_gap_hours >= 24:  # Daily or 5-day
                growth_rate = 1.002  # 0.2% per period
            else:  # Hourly
                growth_rate = 1.001  # 0.1% per period

            for i in range(gap_length):
                predictions[start_idx + i] = before_price * (growth_rate ** (i + 1))
            print(f"Applied growth extrapolation for gap {start_idx + 1}-{end_idx + 1}")

    def predict(self, num_predictions=100, max_attempts=100, current_attempt=1):
        """Get ALL predictions using batch + gap patching strategy with recursive filling"""
        print(f"Starting prediction for {num_predictions} points (attempt {current_attempt}/{max_attempts})...")
        start_time = time.time()

        reference_price = self.price_history[-1]['price'] if self.price_history else 50000.0

        # Show realistic limits at start
        self.calculate_realistic_limits(num_predictions, reference_price)

        # ULTIMATE FALLBACK: Always have a basic prediction ready
        # Adjust growth based on timeframe
        if self.time_gap_hours >= 24 * 7:  # Weekly or monthly
            growth_rate = 1.005  # 0.5% per period
        elif self.time_gap_hours >= 24:  # Daily or 5-day
            growth_rate = 1.002  # 0.2% per period
        else:  # Hourly
            growth_rate = 1.001  # 0.1% per period

        ultimate_fallback = [reference_price * (growth_rate ** i) for i in range(num_predictions)]

        try:
            # Get initial batch predictions
            prompt = self.create_initial_prediction_prompt(num_predictions)
            response_text, success = self.make_prediction_request(prompt)

            if not success:
                print(f"Initial prediction failed: {response_text}")
                if current_attempt < max_attempts:
                    print("Retrying...")
                    time.sleep(1)
                    return self.predict(num_predictions, max_attempts, current_attempt + 1)
                else:
                    print("Max attempts reached, returning fallback results")
                    return ultimate_fallback

            # Parse initial predictions
            predictions = self.parse_prediction(response_text, num_predictions)

            # Check if we got enough predictions
            if not predictions:
                print("No predictions parsed, using fallback")
                return ultimate_fallback

            predictions_received = len(predictions)
            print(f"Received {predictions_received}/{num_predictions} predictions")

            # If we didn't get enough, recursively request the remaining ones
            if predictions_received < num_predictions and current_attempt < max_attempts:
                remaining_count = num_predictions - predictions_received
                print(f"Requesting {remaining_count} more predictions...")

                remaining_predictions = self.predict(remaining_count, max_attempts, current_attempt + 1)

                if remaining_predictions:
                    predictions.extend(remaining_predictions)
                    predictions_received = len(predictions)
                    print(f"After recursive call: {predictions_received}/{num_predictions} predictions")

            # If we still don't have enough after recursion, pad with reasonable values
            if predictions_received < num_predictions:
                missing_count = num_predictions - predictions_received
                print(f"Still missing {missing_count} predictions, padding with extrapolated values")

                last_price = predictions[-1] if predictions else reference_price

                # Use realistic growth for padding based on timeframe
                if self.time_gap_hours >= 24 * 7:  # Weekly or monthly
                    padding_growth = 1.003  # 0.3% per period
                elif self.time_gap_hours >= 24:  # Daily or 5-day
                    padding_growth = 1.0015  # 0.15% per period
                else:  # Hourly
                    padding_growth = 1.0008  # 0.08% per period

                for i in range(missing_count):
                    predictions.append(last_price * (padding_growth ** (i + 1)))

                print(f"Final count after padding: {len(predictions)}/{num_predictions}")

            # Enhanced validation and gap patching
            valid_predictions, invalid_ranges = self.validate_predictions_sequence(predictions, reference_price)

            print(
                f"Initial validation: {len([p for p in valid_predictions if p is not None])}/{num_predictions} valid points")
            print(f"Found {len(invalid_ranges)} gaps to patch")

            # Patch gaps iteratively
            max_patch_iterations = 3
            for iteration in range(max_patch_iterations):
                if not invalid_ranges:
                    break

                print(f"\nGap patching iteration {iteration + 1}/{max_patch_iterations}")
                valid_predictions = self.patch_prediction_gaps(valid_predictions, invalid_ranges, reference_price)

                valid_predictions, invalid_ranges = self.validate_predictions_sequence(
                    [p if p is not None else -1 for p in valid_predictions],
                    reference_price
                )

                valid_predictions = [None if p == -1 else p for p in valid_predictions]

                print(
                    f"After iteration {iteration + 1}: {len([p for p in valid_predictions if p is not None])}/{num_predictions} valid points")
                print(f"Remaining gaps: {len(invalid_ranges)}")

            # FINAL FALLBACK: Ensure we always return valid data
            final_predictions = []
            last_valid = reference_price

            for i, pred in enumerate(valid_predictions):
                if pred is not None:
                    final_predictions.append(pred)
                    last_valid = pred
                else:
                    # If we have None values even after patching, use a simple extrapolation
                    # This ensures we always return valid numbers
                    if self.time_gap_hours >= 24 * 7:  # Weekly or monthly
                        fallback_growth = 1.002  # 0.2% per period
                    elif self.time_gap_hours >= 24:  # Daily or 5-day
                        fallback_growth = 1.001  # 0.1% per period
                    else:  # Hourly
                        fallback_growth = 1.0005  # 0.05% per period

                    fallback_price = last_valid * (fallback_growth ** (i + 1))
                    final_predictions.append(fallback_price)
                    last_valid = fallback_price
                    print(f"Using fallback for position {i + 1}: ${fallback_price:.2f}")

            # Final validation to ensure we have exactly num_predictions
            if len(final_predictions) < num_predictions:
                missing = num_predictions - len(final_predictions)
                print(f"Still missing {missing} predictions, using final fallback")
                last_price = final_predictions[-1] if final_predictions else reference_price

                if self.time_gap_hours >= 24 * 7:  # Weekly or monthly
                    final_growth = 1.001  # 0.1% per period
                elif self.time_gap_hours >= 24:  # Daily or 5-day
                    final_growth = 1.0005  # 0.05% per period
                else:  # Hourly
                    final_growth = 1.0002  # 0.02% per period

                for i in range(missing):
                    final_predictions.append(last_price * (final_growth ** (i + 1)))

            total_time = time.time() - start_time
            valid_count = len([p for p in final_predictions if p is not None])

            print(f"Prediction complete: {len(final_predictions)}/{num_predictions} points in {total_time:.1f}s")
            print(f"Price range: ${min(final_predictions):.2f} - ${max(final_predictions):.2f}")

            return final_predictions[:num_predictions]

        except Exception as e:
            print(f"CRITICAL ERROR in prediction process: {e}")
            print("Returning ultimate fallback predictions")
            # Return the ultimate fallback no matter what
            return ultimate_fallback

    def parse_prediction(self, response_text, num_predictions=5):
        """Extract JSON array from response with enhanced parsing"""
        predictions = []

        # Method 1: Try to parse JSON array first
        try:
            start = response_text.find('[')
            end = response_text.find(']') + 1
            if start != -1 and end != 0:
                json_str = response_text[start:end]
                predictions = json.loads(json_str)
                if isinstance(predictions, list) and all(isinstance(p, (int, float)) for p in predictions):
                    if len(predictions) >= num_predictions:
                        return predictions[:num_predictions]
                    else:
                        return predictions
        except json.JSONDecodeError:
            pass

        # Method 2: Look for number sequences
        numbers = re.findall(r'\b\d+\.?\d*\b', response_text)
        if numbers:
            try:
                predictions = []
                for num in numbers:
                    if len(predictions) >= num_predictions:
                        break
                    try:
                        predictions.append(float(num))
                    except ValueError:
                        continue
                return predictions
            except ValueError:
                pass

        # Method 3: Look for numbers with commas
        number_patterns = re.findall(r'\b[\d,]+\.?\d*\b', response_text)
        for pattern in number_patterns:
            if len(predictions) >= num_predictions:
                break
            try:
                clean_num = float(pattern.replace(',', ''))
                predictions.append(clean_num)
            except:
                continue

        return predictions[:num_predictions]

    def generate_future_dates(self, num_predictions, timeframe):
        """Generate future dates for predictions based on timeframe"""
        if not self.price_history:
            return []

        last_timestamp = datetime.strptime(self.price_history[-1]['time'], '%Y-%m-%d %H:%M:%S')

        timeframe_hours = {
            "1h": 1,
            "4h": 4,
            "1d": 24,
            "5d": 24 * 5,
            "1wk": 24 * 7,
            "1mo": 24 * 30
        }.get(timeframe, 24)  # Default to 24 hours if timeframe not found

        future_dates = []
        for i in range(1, num_predictions + 1):
            future_date = last_timestamp + timedelta(hours=timeframe_hours * i)
            future_dates.append(future_date.strftime('%Y-%m-%d %H:%M:%S'))

        return future_dates