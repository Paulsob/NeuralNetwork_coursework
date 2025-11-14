from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import random
from server.models.music.predict_one_h5 import run_prediction
from server.models.use_model import run_prediction_image
from server.models.images import Image
import os


app = Flask(__name__, static_folder='../client')
CORS(app)
image_model = Image()



# ==================== Статические файлы ====================
@app.route("/")
def home():
    return send_from_directory(app.static_folder, 'index.html')


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/favicon.ico')
def favicon():
    return '', 204


# ==================== Функции предсказаний ====================

# def run_prediction_music(file_storage):
#     """
#     Ваша функция загрузки модели и предсказания для музыки
#     file_storage - FileStorage объект из Flask
#     """
#     return



# def images_predict_stub(file_storage):
#     """Заглушка для изображений"""
#     return {"prediction": random.randint(1, 10)}


def text_predict_stub(text):
    """Заглушка для текста"""
    return {"prediction": random.randint(1, 10)}


# ==================== API ====================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Обработка запросов:
    - FormData с file + type='music' или 'image'
    - JSON с text + type='text'
    """
    try:
        # Файлы (музыка, изображения)
        if request.content_type and 'multipart/form-data' in request.content_type:
            file = request.files.get('file')
            file_type = request.form.get('type')

            if not file:
                return jsonify({"error": "Файл не найден"}), 400

            # Музыка - реальная модель
            if file_type == 'music':
                result = run_prediction(wav_path=file)
                return jsonify({
                    "type": "music",
                    "Автор": result
                })

            # Изображения - заглушка
            # Инициализируем модель живописи один раз (лучше сделать выше, см. ниже)
            
            elif file_type == 'image':
                # Сохраняем временно загруженный файл
                temp_path = f"temp_{file.filename}"
                file.save(temp_path)

                # Запускаем предсказание
                artist, confidence = image_model.predict(temp_path)

                # Удаляем временный файл
                os.remove(temp_path)

                return jsonify({
                    "type": "image",
                    "artist": artist,
                    "confidence": confidence
                })



        # Текст (JSON)
        else:
            content = request.get_json()
            if not content:
                return jsonify({"error": "JSON данные не найдены"}), 400

            text = content.get("text", "")
            if not text:
                return jsonify({"error": "Текст не найден"}), 400

            result = text_predict_stub(text)
            return jsonify({
                "type": "text",
                "result": result
            })

        return jsonify({"error": "Неизвестный тип данных"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)