console.log("Loading script...");
console.log("Image button:", document.getElementById("predictBtnImg"));
console.log("Music button:", document.getElementById("predictBtnMusic"));
console.log("Text button:", document.getElementById("predictBtnTxt"));

document.getElementById("predictBtnImg").onclick = async () => {
  const fileInput = document.getElementById("inputImg");
  if (!fileInput.files.length) {
    alert("Выберите изображение");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("type", "image");

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await resp.json();
    let text = "";

    if (data?.error) {
      text = `Ошибка: ${data.error}`;
    } else if (data?.type === "image") {
      const { artist, confidence } = data;
      text = `Автор: ${artist ?? "unknown"}, уверенность: ${
        confidence !== undefined ? confidence.toFixed(3) : "—"
      }`;
    } else if (data?.artist) {
      text = `Автор: ${data.artist}`;
    } else {
      text = JSON.stringify(data, null, 2);
    }

    document.getElementById("outputImg").innerText = text;
  } catch (e) {
    console.error("Error:", e);
    document.getElementById("outputImg").innerText = `Ошибка: ${e}`;
  }
};

document.getElementById("uploadBtnImg").onclick = () => {
  document.getElementById("inputImg").click();
};

document.getElementById("inputImg").addEventListener("change", (event) => {
  const fileNameEl = document.getElementById("fileNameImg");
  const file = event.target.files[0];
  fileNameEl.textContent = file ? file.name : "Файл не выбран";
});

document.getElementById("predictBtnMusic").onclick = async () => {
  const fileInput = document.getElementById("inputMusic");
  if (!fileInput.files.length) {
    alert("Выберите аудиофайл");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("type", "music");

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await resp.json();
    let text = "";

    if (data && data.type === "music") {
      const r = data.result || {};

      if (Array.isArray(r.labels) && Array.isArray(r.scores)) {
        const lines = r.labels.map(
          (label, i) => `${i + 1}. ${label}: ${(r.scores[i] ?? 0).toFixed(3)}`
        );
        text = `Топ-${lines.length}:\n` + lines.join("\n");
      } else if (
        Array.isArray(r.indices) &&
        (Array.isArray(r.scores) || Array.isArray(r.distances))
      ) {
        const vals = (r.scores || r.distances || []).map((v) => Number(v));
        const lines = r.indices.map(
          (idx, i) => `${i + 1}. #${idx}: ${(vals[i] ?? 0).toFixed(3)}`
        );
        text = `Топ-${lines.length} (индексы):\n` + lines.join("\n");
      } else if (r.prediction !== undefined) {
        text = `Предсказание: ${r.prediction}`;
      } else if (data.author) {
        const confidence = data.confidence !== undefined ? `${data.confidence}%` : "—";
        text = `Автор: ${data.author}`;
      } else if (data.embedding_dim) {
        text = `Вектор признаков размерности ${data.embedding_dim}.`;
      } else {
        text = JSON.stringify(data, null, 2);
      }
    } else if (data && data.note) {
      text = `Модель не загружена: ${data.note}`;
    } else if (data && data.final_result !== undefined) {
      text = `Результат: ${data.final_result}`;
    } else {
      text = JSON.stringify(data, null, 2);
    }

    document.getElementById("outputMusic").innerText = text;
  } catch (e) {
    console.error("Error:", e);
    document.getElementById("outputMusic").innerText = `Ошибка: ${e}`;
  }
};

document.getElementById("uploadBtnMusic").onclick = () => {
  document.getElementById("inputMusic").click();
};

document.getElementById("inputMusic").addEventListener("change", (event) => {
  const fileNameEl = document.getElementById("fileNameMusic");
  const file = event.target.files[0];
  fileNameEl.textContent = file ? file.name : "Файл не выбран";
});

document.getElementById("uploadBtnTxt").onclick = () => {
  document.getElementById("inputTxtFile").click();
};

document.getElementById("inputTxtFile").addEventListener("change", (event) => {
  const fileNameEl = document.getElementById("fileNameTxt");
  const file = event.target.files[0];
  fileNameEl.textContent = file ? file.name : "Файл не выбран";

  if (file) {
      document.getElementById("inputTxt").value = "";
      document.getElementById("inputTxt").placeholder = "Выбран файл. Для ввода текста вручную удалите файл или перезагрузите страницу.";
      document.getElementById("inputTxt").disabled = true;
  } else {
      document.getElementById("inputTxt").disabled = false;
      document.getElementById("inputTxt").placeholder = "Или введите текст стихотворения здесь...";
  }
});

document.getElementById("predictBtnTxt").onclick = async () => {
  const fileInput = document.getElementById("inputTxtFile");
  const textInput = document.getElementById("inputTxt");
  const textVal = textInput.value.trim();

  const hasFile = fileInput.files.length > 0;
  const hasText = textVal.length > 0;

  if (!hasFile && !hasText) {
    alert("Загрузите файл или введите текст стихотворения");
    return;
  }

  const formData = new FormData();
  formData.append("type", "text");

  if (hasFile) {
    formData.append("file", fileInput.files[0]);
  }
  else {
    formData.append("text", textVal);
  }

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await resp.json();

    if (data.error) {
      document.getElementById("outputTxt").innerText = `Ошибка: ${data.error}`;
      return;
    }

    if (data.type === "text" && data.author) {
      document.getElementById("outputTxt").innerText =
        `Автор: ${data.author}\nУверенность: ${data.confidence}`;
    } else {
      document.getElementById("outputTxt").innerText =
        `Результат: ${JSON.stringify(data, null, 2)}`;
    }
  } catch (e) {
    console.error("Error:", e);
    document.getElementById("outputTxt").innerText = `Ошибка: ${e.message}`;
  }
};
