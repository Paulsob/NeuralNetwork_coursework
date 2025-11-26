from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from models.music.predict_one_h5 import run_prediction
from models.images import Image
from models.text_model import predict as predict_text
import os
import docx

app = Flask(__name__, static_folder='../client')
CORS(app)

image_model = Image()

@app.route("/")
def home():
    return send_from_directory(app.static_folder, 'index.html')


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.content_type and 'multipart/form-data' in request.content_type:
            req_type = request.form.get('type')

            # текст
            if req_type == 'text':
                text_content = ""

                if 'file' in request.files:
                    file = request.files['file']
                    filename = file.filename.lower()

                    if filename.endswith('.txt'):
                        text_content = file.read().decode('utf-8')
                    elif filename.endswith('.docx'):
                        doc = docx.Document(file)
                        text_content = "\n".join([p.text for p in doc.paragraphs])
                    else:
                        return jsonify({"error": "Поддерживаются только .txt и .docx"}), 400

                elif 'text' in request.form:
                    text_content = request.form['text']

                else:
                    return jsonify({"error": "Нет текста для анализа"}), 400

                if not text_content.strip():
                    return jsonify({"error": "Текст пуст"}), 400

                poet, confidence = predict_text(text_content)
                return jsonify({
                    "type": "text",
                    "author": poet,
                    "confidence": round(float(confidence),3)
                })

            # музыка
            elif req_type == 'music':
                file = request.files.get('file')
                if not file:
                    return jsonify({"error": "Аудиофайл не найден"}), 400

                result = run_prediction(wav_path=file)
                return jsonify({
                    "type": "music",
                    "author": result
                })

            # картины
            elif req_type == 'image':
                file = request.files.get('file')
                if not file:
                    return jsonify({"error": "Изображение не найдено"}), 400

                temp_path = f"temp_{file.filename}"
                file.save(temp_path)

                try:
                    artist, confidence = image_model.predict(temp_path)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                return jsonify({
                    "type": "image",
                    "artist": artist,
                    "confidence": confidence
                })

        elif request.is_json:
            content = request.get_json()
            text = content.get("text", "")
            if not text:
                return jsonify({"error": "Текст не найден"}), 400

            poet, confidence = predict_text(text)
            return jsonify({
                "type": "text",
                "author": poet,
                "confidence": float(confidence * 100)
            })

        return jsonify({"error": "Неизвестный тип данных или запроса"}), 400

    except Exception as e:
        print(f"Server Error: {e}")  # Полезно видеть ошибку в консоли сервера
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)