import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path
from typing import Optional, Union


class Image:
    """
    Заглушка для интеграции модели живописи.

    Загружает модель из server/models/images/final_inceptionresnetv3_art_model_v5.h5,
    а также словарь классов из server/models/images/class_indices.json.
    Метод predict по умолчанию использует один тестовый пример из data/images/Хорошие примеры.

    Как адаптировать под реальную интеграцию:
      * передавать путь к изображению через аргумент image_path;
      * нормализовать входные данные так же, как при обучении модели;
      * возвращать структуру данных, которую ждёт сервер (например, top-1 класс и вероятность).
    """

    MODEL_PATH = Path(__file__).resolve().parent / "images" / "final_inceptionresnetv3_art_model_v5.h5"
    CLASSES_PATH = Path(__file__).resolve().parent / "images" / "class_indices.json"
    DEFAULT_SAMPLE = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "images"
        / "good_examples"
        / "Kazimir_Malevich_87.jpg"
    )

    def __init__(self):
        self.model = load_model(self.MODEL_PATH)
        with open(self.CLASSES_PATH, "r", encoding="utf-8") as fp:
            raw_mapping = json.load(fp)
        self.idx_to_class = {int(k): v for k, v in raw_mapping.items()}

    def _prepare_image(self, image_path: Path) -> np.ndarray:
        """Готовит изображение к подаче в сеть."""
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def predict(self, image_path: Optional[Union[str, Path]] = None):
        """
        Возвращает пару (имя_класса, вероятность).

        Параметры:
          image_path — путь к изображению. Если не задан, используется DEFAULT_SAMPLE.
        """
        image_path = Path(image_path) if image_path else self.DEFAULT_SAMPLE
        input_batch = self._prepare_image(image_path)
        preds = self.model.predict(input_batch)

        top_index = int(np.argmax(preds, axis=1)[0])
        top_class = self.idx_to_class.get(top_index, "unknown")
        top_confidence = float(preds[0][top_index])

        return top_class, top_confidence
