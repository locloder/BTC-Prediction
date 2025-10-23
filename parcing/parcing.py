from newsapi import NewsApiClient
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys

NEWS_API_KEY = "d7f7f2f45ad14d1cbaa6f68952f8a8e1"

KEYWORDS = "bitcoin OR crypto OR ethereum OR solana OR nft"
LANGUAGE = 'en'
FILENAME = "news.txt"


def get_date_range(time_str):
    now = datetime.now()
    time_str = time_str.strip().lower()

    if time_str.endswith('d'):
        days = int(time_str.replace('d', ''))
        from_date = now - timedelta(days=days)
    elif time_str.endswith('w'):
        weeks = int(time_str.replace('w', ''))
        from_date = now - timedelta(weeks=weeks)
    elif time_str.endswith('m'):
        months = int(time_str.replace('m', ''))
        from_date = now - relativedelta(months=months)
    else:
        print("Невірний формат інтервалу. Використовуємо 7 днів (1w).")
        from_date = now - timedelta(weeks=1)

    if (now - from_date).days > 30:
        print("NewsAPI зазвичай обмежує пошук 30 днями. Обрізаємо до 30 днів.")
        from_date = now - timedelta(days=30)

    return from_date.strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')


def fetch_crypto_news(from_date, to_date):
    if NEWS_API_KEY == "ВСТАВТЕ_ВАШ_NEWS_API_KEY_СЮДИ":
        print("\n❗ КЛЮЧ API ВІДСУТНІЙ. Будь ласка, отримайте безкоштовний ключ на NewsAPI.org і вставте його в код.")
        return []

    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)

        all_articles = newsapi.get_everything(
            q=KEYWORDS,
            language=LANGUAGE,
            from_param=from_date,
            to=to_date,
            sort_by='relevancy',
            page_size=100
        )

        if all_articles['status'] == 'ok':
            print(
                f"\n✅ Знайдено {all_articles['totalResults']} результатів. Отримано: {len(all_articles['articles'])} статей.")
            return all_articles['articles']
        else:
            print(f"❌ Помилка NewsAPI: {all_articles.get('message', 'Невідома помилка')}")
            return []

    except Exception as e:
        print(f"❌ Критична помилка підключення або API: {e}")
        return []


def save_to_txt(articles, filename, time_period):
    if not articles:
        print("Не знайдено жодної відповідної статті.")
        return

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"=== ЗВІТ ПРО КРИПТОВАЛЮТНІ НОВИНИ ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
        f.write(f"Запитаний період: {time_period}\n")
        f.write(f"Джерело: NewsAPI.org\n\n")

        for i, article in enumerate(articles):
            f.write(f"--- СТАТТЯ {i + 1} ---\n")
            f.write(f"ЗАГОЛОВОК: {article.get('title', 'N/A')}\n")
            f.write(f"ДЖЕРЕЛО: {article['source'].get('name', 'N/A')}\n")
            f.write(f"ДАТА ПУБЛІКАЦІЇ: {article.get('publishedAt', 'N/A')}\n")
            f.write(f"ОПИС: {article.get('description', 'Опис відсутній.')}\n")
            f.write(f"ПОСИЛАННЯ: {article.get('url', 'N/A')}\n")
            f.write("-" * 50 + "\n\n")

    print(f"\n✅ Успішно збережено {len(articles)} статей у файл: {filename}")


if __name__ == "__main__":
    user_input_time = input("Введіть часовий інтервал для парсингу (наприклад, 1d, 7d, 1m): ")

    start_date, end_date = get_date_range(user_input_time)
    print(f"Парсинг буде виконано з {start_date} по {end_date}.")

    news_articles = fetch_crypto_news(start_date, end_date)

    save_to_txt(news_articles, FILENAME, user_input_time)