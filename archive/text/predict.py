import os
import torch
import torch.nn as nn
import joblib
import json
from torchqrnn import QRNN


class AttributionQRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(AttributionQRNN, self).__init__()
        self.qrnn = QRNN(input_size, hidden_size, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        output, _ = self.qrnn(x)
        last = output[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


class TextPredictor:

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
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            print(f"Файл {config_path} не найден, используем значения по умолчанию")
            self.config = {
                "input_size": 20000,
                "hidden_size": 256,
                "num_layers": 2,
                "num_classes": 20,
                "dropout": 0.3
            }
        
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Файл {vectorizer_path} не найден. "
                f"Запустите save_text.py для создания необходимых файлов."
            )
        self.vectorizer = joblib.load(vectorizer_path)
        print(f"Загружен vectorizer из {vectorizer_path}")
        
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(
                f"Файл {label_encoder_path} не найден. "
                f"Запустите save_text.py для создания необходимых файлов."
            )
        self.label_encoder = joblib.load(label_encoder_path)
        print(f"Загружен label_encoder из {label_encoder_path}")
        
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
            print(f"Загружена модель из {model_path}")
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
            (author_name, confidence_percent) - имя автора и уверенность в процентах

        """
        self.model.eval()
        
        X_vec = self.vectorizer.transform([text]).toarray()
        X_tensor = torch.tensor(X_vec, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probabilities, dim=1)
            
            predicted_poet = self.label_encoder.inverse_transform([pred.item()])[0]
            confidence_percent = confidence.item() * 100
        
        if return_probabilities:
            probs_dict = {}
            all_probs = probabilities[0].cpu().numpy()
            all_authors = self.label_encoder.classes_
            for i, author in enumerate(all_authors):
                probs_dict[author] = float(all_probs[i] * 100)
            
            probs_dict = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
            
            return predicted_poet, confidence_percent, probs_dict
        
        return predicted_poet, confidence_percent
    
    def get_available_authors(self):
        return list(self.label_encoder.classes_)


if __name__ == "__main__":
    print("Инициализация модели...")
    predictor = TextPredictor()
    
    print(f"\nДоступно авторов: {len(predictor.get_available_authors())}")
    print("Примеры авторов:", predictor.get_available_authors()[:5])
    
    example_text = """В мире слов разнообразных, Что блестят, горят и жгут,—
Золотых, стальных, алмазных,—Нет священней слова: «труд»!
Троглодит стал человекомВ тот заветный день, когда
Он сошник повел к просекам,Начиная круг труда."""
    
    print(f"\nАнализируем текст:")
    print(f"{example_text[:100]}...")
    
    author, confidence = predictor.predict(example_text)
    print(f"\nПредсказанный автор: {author}")
    print(f"Уверенность: {confidence:.2f}%")
    
    print("\nТоп-5 наиболее вероятных авторов:")
    author, confidence, probs = predictor.predict(example_text, return_probabilities=True)
    for i, (auth, prob) in enumerate(list(probs.items())[:5], 1):
        print(f"  {i}. {auth}: {prob:.2f}%")

