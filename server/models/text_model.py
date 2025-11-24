import os
import torch
import torch.nn as nn
import joblib
from torchqrnn import QRNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = os.path.join(BASE_DIR, "text")

MODEL_PATH = os.path.join(TEXT_DIR, "qrnn_text.pth")
VECTORIZER_PATH = os.path.join(TEXT_DIR, "vectorizer.pkl")
LABELENC_PATH = os.path.join(TEXT_DIR, "label_encoder.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttributionQRNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, num_classes=20, dropout=0.3):
        super().__init__()
        self.qrnn = QRNN(input_size, hidden_size, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, seq=1, features)
        output, _ = self.qrnn(x)
        last = output[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)


print("Загружаем TF-IDF и LabelEncoder…")
vectorizer = joblib.load(VECTORIZER_PATH)
le = joblib.load(LABELENC_PATH)

input_size = len(vectorizer.get_feature_names_out())
num_classes = len(le.classes_)

print("Загружаем модель…")
model = AttributionQRNN(
    input_size=input_size,
    hidden_size=256,
    num_layers=2,
    num_classes=num_classes
).to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(state, strict=False)
model.eval()

print("Модель автора стихотворений загружена!")


def predict(text: str):
    X = vectorizer.transform([text]).toarray()
    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    poet = le.inverse_transform([pred.item()])[0]
    return poet, float(conf.item())

if __name__ == "__main__":
    while True:
        print("\nВведите стихотворение (построчно).")
        print("Чтобы завершить ввод — введите пустую строку два раза подряд.")
        print("Чтобы выйти полностью — введите 'exit' на новой строке.")
        print()

        lines = []
        while True:
            line = input()
            if line.strip().lower() == "exit":
                print("Выход.")
                exit()

            if line == "":
                # пустая строка — завершаем ввод стихотворения
                break

            lines.append(line)

        if not lines:  # если пользователь просто нажал Enter
            print("Выход.")
            break

        text = "\n".join(lines)

        poet, confidence = predict(text)
        print(f"\nПредсказанный автор: {poet}")
        print(f"Уверенность: {confidence:.2f}%")

