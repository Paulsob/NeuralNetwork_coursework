"""
Модуль для предсказания автора текста с помощью обученной QRNN модели.
Используется для интеграции в веб-приложение без необходимости переобучения модели.
"""

import os
import torch
import torch.nn as nn
import joblib
import json
# Используем локальную версию torchqrnn из папки проекта
from .torchqrnn import QRNN


class AttributionQRNN(nn.Module):
    """Модель для атрибуции текста (определение автора)"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(AttributionQRNN, self).__init__()
        self.qrnn = QRNN(input_size, hidden_size, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, input_size)
        output, _ = self.qrnn(x)
        last = output[:, -1, :]  # последний временной шаг
        last = self.dropout(last)
        return self.fc(last)


class TextPredictor:
    """Класс для предсказания автора текста"""
    
    def __init__(self, model_path="qrnn_text.pth", 
                 vectorizer_path="vectorizer.pkl",
                 label_encoder_path="label_encoder.pkl",
                 config_path="config.json"):
        """
        Инициализация предсказателя
        
        Args:
            model_path: путь к файлу с весами модели (.pth)
            vectorizer_path: путь к сохраненному TfidfVectorizer (.pkl)
            label_encoder_path: путь к сохраненному LabelEncoder (.pkl)
            config_path: путь к файлу конфигурации (.json)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загружаем конфигурацию
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # Значения по умолчанию (если config.json не найден)
            print(f"⚠️ Файл {config_path} не найден, используем значения по умолчанию")
            self.config = {
                "input_size": 20000,
                "hidden_size": 256,
                "num_layers": 2,
                "num_classes": 20,
                "dropout": 0.3
            }
        
        # Загружаем vectorizer
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Файл {vectorizer_path} не найден. "
                f"Запустите save_text.py для создания необходимых файлов."
            )
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"✅ Загружен vectorizer из {vectorizer_path}")
        
        # Загружаем label encoder
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(
                f"Файл {label_encoder_path} не найден. "
                f"Запустите save_text.py для создания необходимых файлов."
            )
        self.label_encoder = joblib.load(label_encoder_path)
        print(f"✅ Загружен label_encoder из {label_encoder_path}")
        
        # Создаем и загружаем модель
        self.model = AttributionQRNN(
            input_size=self.config["input_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            num_classes=self.config["num_classes"],
            dropout=self.config.get("dropout", 0.3)
        )
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()  # Переводим в режим оценки
            print(f"✅ Загружена модель из {model_path}")
        else:
            raise FileNotFoundError(
                f"Файл модели {model_path} не найден. "
                f"Убедитесь, что модель была обучена и сохранена."
            )
    
    def predict(self, text, return_probabilities=False):
        """
        Предсказание автора для заданного текста
        
        Args:
            text: строка с текстом для анализа
            return_probabilities: если True, возвращает также вероятности для всех классов
        
        Returns:
            Если return_probabilities=False:
                (author_name, confidence_percent) - имя автора и уверенность в процентах
            Если return_probabilities=True:
                (author_name, confidence_percent, probabilities_dict) - дополнительно словарь с вероятностями всех авторов
        """
        self.model.eval()
        
        # Векторизуем текст
        X_vec = self.vectorizer.transform([text]).toarray()
        X_tensor = torch.tensor(X_vec, dtype=torch.float32).to(self.device)
        
        # Делаем предсказание
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probabilities, dim=1)
            
            predicted_poet = self.label_encoder.inverse_transform([pred.item()])[0]
            confidence_percent = confidence.item() * 100
        
        if return_probabilities:
            # Создаем словарь с вероятностями всех авторов
            probs_dict = {}
            all_probs = probabilities[0].cpu().numpy()
            all_authors = self.label_encoder.classes_
            for i, author in enumerate(all_authors):
                probs_dict[author] = float(all_probs[i] * 100)
            
            # Сортируем по убыванию вероятности
            probs_dict = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
            
            return predicted_poet, confidence_percent, probs_dict
        
        return predicted_poet, confidence_percent
    
    def get_available_authors(self):
        """Возвращает список всех доступных авторов"""
        return list(self.label_encoder.classes_)


# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========
if __name__ == "__main__":
    # Инициализация предсказателя (загружает модель один раз)
    print("Инициализация модели...")
    predictor = TextPredictor()
    
    print(f"\nДоступно авторов: {len(predictor.get_available_authors())}")
    print("Примеры авторов:", predictor.get_available_authors()[:5])
    
    # Пример предсказания
    example_text = """В мире слов разнообразных, Что блестят, горят и жгут,—
Золотых, стальных, алмазных,—Нет священней слова: «труд»!
Троглодит стал человекомВ тот заветный день, когда
Он сошник повел к просекам,Начиная круг труда."""
    
    print(f"\n📝 Анализируем текст:")
    print(f"{example_text[:100]}...")
    
    # Простое предсказание
    author, confidence = predictor.predict(example_text)
    print(f"\n🎯 Предсказанный автор: {author}")
    print(f"📊 Уверенность: {confidence:.2f}%")
    
    # Предсказание с вероятностями всех авторов
    print("\n📊 Топ-5 наиболее вероятных авторов:")
    author, confidence, probs = predictor.predict(example_text, return_probabilities=True)
    for i, (auth, prob) in enumerate(list(probs.items())[:5], 1):
        print(f"  {i}. {auth}: {prob:.2f}%")

