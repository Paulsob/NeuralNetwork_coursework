// Проверяем, что элементы существуют
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

  const resp = await fetch("/predict", {
    method: "POST",
    body: formData,
  });

  try {
    const data = await resp.json();
    document.getElementById(
      "outputImg"
    ).innerText = `Результат: ${data.final_result}`;
  } catch (e) {
    console.error("Error:", e);
    document.getElementById("outputImg").innerText = `Ошибка: ${e}`;
  }
};

// Кнопка выбора файла
document.getElementById("uploadBtnImg").onclick = () => {
  document.getElementById("inputImg").click();
};

// Отображение имени выбранного файла
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
        text =
          `Топ-${lines.length}:
` + lines.join("\n");
      } else if (
        Array.isArray(r.indices) &&
        (Array.isArray(r.scores) || Array.isArray(r.distances))
      ) {
        const vals = (r.scores || r.distances || []).map((v) => Number(v));
        const lines = r.indices.map(
          (idx, i) => `${i + 1}. #${idx}: ${(vals[i] ?? 0).toFixed(3)}`
        );
        text =
          `Топ-${lines.length} (индексы):
` + lines.join("\n");
      } else if (r.prediction !== undefined) {
        text = `Предсказание: ${r.prediction}`;
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
    document.getElementById("outputMusic").innerText = `Ошибка: ${e}`;
  }
};

// Кнопка выбора файла
document.getElementById("uploadBtnMusic").onclick = () => {
  document.getElementById("inputMusic").click();
};

// Отображение имени выбранного файла
document.getElementById("inputMusic").addEventListener("change", (event) => {
  const fileNameEl = document.getElementById("fileNameMusic");
  const file = event.target.files[0];
  fileNameEl.textContent = file ? file.name : "Файл не выбран";
});


document.getElementById("predictBtnTxt").onclick = async () => {
  const text = document.getElementById("inputTxt").value.trim();
  if (!text) {
    alert("Выберите текст");
    return;
  }

  const resp = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, type: "text" }),
  });

  try {
    const data = await resp.json();
    document.getElementById(
      "outputTxt"
    ).innerText = `Результат: ${data.final_result}`;
  } catch (e) {
    console.error("Error:", e);
    document.getElementById("outputTxt").innerText = `Ошибка: ${e}`;
  }
};

// Кнопка выбора файла
document.getElementById("uploadBtnTxt").onclick = () => {
  document.getElementById("inputTxt").click();
};

// Отображение имени выбранного файла
document.getElementById("inputTxt").addEventListener("change", (event) => {
  const fileNameEl = document.getElementById("fileNameTxt");
  const file = event.target.files[0];
  fileNameEl.textContent = file ? file.name : "Файл не выбран";
});
