from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Input

# 1️⃣ Создаём ту же архитектуру, которая использовалась при обучении
input_tensor = Input(shape=(299, 299, 3))
base_model = InceptionResNetV2(weights=None, include_top=False, input_tensor=input_tensor)

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)  # <-- поменяй число классов!

model = Model(inputs=input_tensor, outputs=output)

# 2️⃣ Загружаем веса
model.load_weights(r"C:\Users\psobo\PycharmProjects\NeuralNetwork_coursework\server\models\images\final_inceptionresnetv3_art_model_v5.h5")

print("✅ Веса успешно загружены!")
