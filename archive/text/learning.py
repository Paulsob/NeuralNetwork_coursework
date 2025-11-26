import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

csv_path = r"/data/text/poems.csv"

df = pd.read_csv(csv_path)

if "text" not in df.columns:
    raise ValueError(f"В файле нет колонки 'text'. Найдены колонки: {df.columns}")

texts = df["text"].astype(str).tolist()

print(f"Загружено текстов: {len(texts)}")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2)
)

vectorizer.fit(texts)

joblib.dump(vectorizer, "vectorizer.pkl")

print(f"Готово! Длина idf: {len(vectorizer.idf_)}")
