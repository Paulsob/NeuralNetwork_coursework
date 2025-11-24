from predict import TextPredictor


if __name__ == "__main__":
    print("Инициализация модели...")
    predictor = TextPredictor(
        model_path="qrnn_text.pth",
        vectorizer_path="vectorizer.pkl",
        label_encoder_path="label_encoder.pkl",
        config_path="config.json"
    )

    while True:
        print("\nВведите стихотворение (несколько строк).")
        print("Пустая строка — завершение ввода. 'exit' — выход.\n")

        lines = []
        while True:
            line = input()
            if line.strip().lower() == "exit":
                exit()
            if line == "":
                break
            lines.append(line)

        if not lines:
            exit()

        text = "\n".join(lines)

        author, conf = predictor.predict(text)
        print(f"\n🎯 Автор: {author}")
        print(f"📊 Уверенность: {conf:.2f}%\n")
