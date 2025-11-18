import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Путь к CSV с текстами
csv_path = r"C:\Users\psobo\PycharmProjects\NeuralNetwork_coursework\data\text\poems.csv"

# Загружаем CSV
df = pd.read_csv(csv_path)

# Проверяем наличие нужной колонки
if "text" not in df.columns:
    raise ValueError(f"В файле нет колонки 'text'. Найдены колонки: {df.columns}")

# Получаем список всех стихотворений
texts = df["text"].astype(str).tolist()

print(f"Загружено текстов: {len(texts)}")

# Создаем TF-IDF векторизатор
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2)
)

# Обучаем
vectorizer.fit(texts)

# Сохраняем
joblib.dump(vectorizer, "vectorizer.pkl")

print(f"Готово! Длина idf: {len(vectorizer.idf_)}")
