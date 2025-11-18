import joblib
v = joblib.load("vectorizer.pkl")
print(hasattr(v, "idf_"))
print(len(getattr(v, "idf_", [])))
