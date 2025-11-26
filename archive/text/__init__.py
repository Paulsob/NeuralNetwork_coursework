import os
from pathlib import Path
from .predict import TextPredictor


class Text:
    """
    Класс-обёртка для модели распознавания автора текста.
    Используется в server.py для обработки запросов на распознавание стихов.
    """
    
    def __init__(self):
        """Инициализация модели с правильными путями к файлам"""
        model_dir = Path(__file__).resolve().parent
        
        model_path = str(model_dir / "qrnn_text.pth")
        vectorizer_path = str(model_dir / "vectorizer.pkl")
        label_encoder_path = str(model_dir / "label_encoder.pkl")
        config_path = str(model_dir / "config.json")
        
        self.predictor = TextPredictor(
            model_path=model_path,
            vectorizer_path=vectorizer_path,
            label_encoder_path=label_encoder_path,
            config_path=config_path
        )
    
    def predict(self, text):
        """
        Предсказание автора для заданного текста
        
        Args:
            text: строка с текстом для анализа
            
        Returns:
            dict: {"author": имя_автора, "confidence": уверенность_в_процентах}
        """
        author, confidence = self.predictor.predict(text, return_probabilities=False)
        return {
            "author": author,
            "confidence": round(confidence, 2)
        }
    
    def get_available_authors(self):
        return self.predictor.get_available_authors()

