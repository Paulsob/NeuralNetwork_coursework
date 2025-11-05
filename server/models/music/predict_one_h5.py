# -*- coding: utf-8 -*-
"""
predict_one_h5.py
Извлекает embedding для одного wav (gu.wav) и сравнивает с artist20-temp.joblib.pkl.
Использует полностью сохранённую модель в формате .h5
Запуск (в папке src):
    python predict_one_h5.py
"""
import os
import numpy as np
from scipy.spatial.distance import cosine
import librosa
import joblib
from tensorflow.keras.models import load_model

HERE = os.path.dirname(os.path.abspath(__file__))

# Параметры
SR = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

DEFAULT_JOBLIB = os.path.join(HERE, 'artist20-temp.joblib.pkl')
DEFAULT_WAV = os.path.join(HERE, 'data/Prince_-_I_Wanna_Be_Your_Lover.wav')

DEFAULT_MODEL = os.path.join(HERE, '20_32_0.h5')


def load_joblib(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Joblib file not found: {path}")
    obj = joblib.load(path)
    print("[INFO] Loaded joblib:", path, " type:", type(obj))
    return obj


def build_enroll_dict(obj):
    # если dict — возвращаем как есть
    if isinstance(obj, dict):
        return obj
    # если list-like: [predictY_train, Y_train, ...]
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        predictY_train = np.array(obj[0])
        Y_train = np.array(obj[1])
        enroll = {}
        for y in set(Y_train):
            enroll[y] = np.mean(predictY_train[np.where(Y_train == y)], axis=0)
        print("[INFO] Built enroll_dict from list-like joblib")
        return enroll
    raise ValueError("Unknown joblib format")


def wav_to_slices(wav_path, slice_length):
    y, sr = librosa.load(wav_path, sr=SR)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    log_S = librosa.amplitude_to_db(S, ref=1.0)
    T = log_S.shape[1]
    slices = []

    if T < slice_length:
        pad = np.zeros((N_MELS, slice_length))
        pad[:, :T] = log_S
        slices.append(pad)
    else:
        n_slices = int(T / slice_length)
        for i in range(n_slices):
            seg = log_S[:, slice_length * i: slice_length * (i + 1)]
            if seg.shape[1] == slice_length:
                slices.append(seg)
    if len(slices) == 0:
        pad = np.zeros((N_MELS, slice_length))
        start = max(0, T - slice_length)
        seg = log_S[:, start:T]
        pad[:, :seg.shape[1]] = seg
        slices.append(pad)

    X = np.array(slices)
    return X


def compare_embedding(emb, enroll_dict, topn=10):
    emb = np.array(emb).reshape(-1)
    res = []
    for lbl, v in enroll_dict.items():
        vec = np.array(v).reshape(-1)
        try:
            d = float(cosine(vec, emb))
        except Exception:
            from sklearn.metrics.pairwise import cosine_similarity
            d = 1.0 - float(cosine_similarity(vec.reshape(1, -1), emb.reshape(1, -1))[0, 0])
        res.append((lbl, d))
    res.sort(key=lambda x: x[1])
    return res[:topn]


def run_prediction(
        wav_path=DEFAULT_WAV,
        slice_len=32,
        model_path=DEFAULT_MODEL,
        joblib_path=DEFAULT_JOBLIB
):
    print("[INFO] wav:", wav_path)
    print("[INFO] slice_length:", slice_len)
    print("[INFO] model_path:", model_path)
    print("[INFO] artist joblib:", joblib_path)

    # 1) load artists joblib
    job = load_joblib(joblib_path)
    enroll = build_enroll_dict(job)

    # 2) convert wav -> slices
    X = wav_to_slices(wav_path, slice_len)
    print("[INFO] Generated {} slices from wav".format(X.shape[0]))
    X_in = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))

    # 3) load full model
    model = load_model(model_path)
    print("[INFO] Model loaded:", model_path)

    # 4) compute embeddings (например, берем предпоследний слой)
    from tensorflow.keras.models import Model
    embedding_model = Model(inputs=model.inputs, outputs=model.layers[-4].output)
    preds = embedding_model.predict(X_in)
    emb = np.mean(preds, axis=0)
    print("[INFO] Obtained embedding vector of length", emb.shape)

    # 5) compare embeddings
    # 5) compare embeddings
    top = compare_embedding(emb, enroll, topn=10)
    best_lbl, best_dist = top[0]
    print(f"Best match: {best_lbl}, distance={best_dist:.4f}")
    return best_lbl
