import io
import json
from typing import Any, Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from tensorflow.keras.models import load_model


def _l2_normalize(vec: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = np.linalg.norm(vec) + 1e-12
    return (vec / norm).astype(np.float32)


def _cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class Music:
    """Music embedding extractor + nearest search wrapper.

    Expects a Keras .h5 model that outputs an embedding for a single audio input.
    Compares the embedding to an index/classifier stored in a joblib .pkl file.
    """

    def __init__(
        self,
        model_path: str,
        index_path: str,
        sample_rate: int = 22050,
        n_mels: int = 128,
        hop_length: int = 512,
        duration_sec: int = 30,
        top_k: int = 5,
    ) -> None:
        self.model_path = model_path
        self.index_path = index_path
        self.sample_rate = int(sample_rate)
        self.n_mels = int(n_mels)
        self.hop_length = int(hop_length)
        self.duration_sec = int(duration_sec)
        self.top_k = int(top_k)

        # Load model and index once
        self.model = load_model(self.model_path)
        self.index = joblib.load(self.index_path)

        # Try to detect index type characteristics
        self.index_has_kneighbors = hasattr(self.index, "kneighbors")
        self.index_has_predict_proba = hasattr(self.index, "predict_proba")
        self.index_has_predict = hasattr(self.index, "predict")

    def _read_audio(self, wav_bytes: bytes) -> NDArray[np.float32]:
        # Prefer soundfile to preserve dtype; fallback to librosa via temp file for broader formats (e.g., mp3)
        try:
            data, sr = sf.read(io.BytesIO(wav_bytes), always_2d=False)
            if sr != self.sample_rate:
                y = librosa.resample(y=data.astype(np.float32), orig_sr=sr, target_sr=self.sample_rate)
            else:
                y = data.astype(np.float32)
        except Exception:
            import tempfile, os
            # Try to guess extension by header to help external decoders
            header = wav_bytes[:12]
            if header.startswith(b"RIFF"):
                ext = ".wav"
            elif header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
                ext = ".mp3"
            else:
                ext = ".tmp"

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name
            try:
                y, _ = librosa.load(tmp_path, sr=self.sample_rate, mono=True)
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        # Force mono
        if y.ndim == 2:
            y = np.mean(y, axis=1)

        # Trim/pad to duration_sec
        target_len = self.sample_rate * self.duration_sec
        if y.shape[0] > target_len:
            y = y[:target_len]
        elif y.shape[0] < target_len:
            pad = target_len - y.shape[0]
            y = np.pad(y, (0, pad), mode="constant")

        return y.astype(np.float32)

    def _to_features(self, y: NDArray[np.float32]) -> NDArray[np.float32]:
        # Default: log-mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=self.sample_rate, n_mels=self.n_mels, hop_length=self.hop_length)
        log_mel = librosa.power_to_db(mel + 1e-9)
        # Normalize per-feature
        mean = np.mean(log_mel, axis=1, keepdims=True)
        std = np.std(log_mel, axis=1, keepdims=True) + 1e-9
        norm = (log_mel - mean) / std
        return norm.astype(np.float32)

    def _prepare_input_for_model(self, feat: NDArray[np.float32]) -> NDArray[np.float32]:
        x = feat
        # Common conventions: (time, n_mels) or (n_mels, time)
        # Try (time, n_mels, 1)
        x_tn = np.transpose(x, (1, 0))  # (time, n_mels)
        x_tn = np.expand_dims(x_tn, axis=-1)  # (time, n_mels, 1)
        x_batch1 = np.expand_dims(x_tn, axis=0)  # (1, time, n_mels, 1)
        return x_batch1.astype(np.float32)

    def _forward_embedding(self, model_input: NDArray[np.float32]) -> NDArray[np.float32]:
        emb = self.model.predict(model_input, verbose=0)
        emb = np.array(emb)
        if emb.ndim == 2 and emb.shape[0] == 1:
            emb = emb[0]
        emb = emb.reshape(-1).astype(np.float32)
        return _l2_normalize(emb)

    def _query_index(
        self,
        embedding: NDArray[np.float32],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        k = top_k or self.top_k

        # Case 1: NearestNeighbors-like index
        if self.index_has_kneighbors:
            distances, indices = self.index.kneighbors(embedding.reshape(1, -1), n_neighbors=k)
            # Convert distances to cosine-like scores if metric is euclidean
            d = distances[0].astype(float).tolist()
            idx = indices[0].astype(int).tolist()
            return {"indices": idx, "distances": d}

        # Case 2: Classifier with probabilities
        if self.index_has_predict_proba:
            probs = self.index.predict_proba(embedding.reshape(1, -1))[0]
            # Top-k
            top_idx = np.argsort(probs)[::-1][:k]
            top_probs = probs[top_idx].astype(float).tolist()
            idx = top_idx.astype(int).tolist()
            return {"indices": idx, "scores": top_probs}

        # Case 3: Generic predict
        if self.index_has_predict:
            pred = self.index.predict(embedding.reshape(1, -1))
            return {"prediction": pred[0] if len(pred) else None}

        # Case 4: Assume stored embeddings and optional labels
        if isinstance(self.index, dict) and "embeddings" in self.index:
            db_emb = np.asarray(self.index["embeddings"], dtype=np.float32)
            # Cosine similarities
            sims = (db_emb @ embedding) / (
                np.linalg.norm(db_emb, axis=1) * (np.linalg.norm(embedding) + 1e-12) + 1e-12
            )
            top_idx = np.argsort(sims)[::-1][:k]
            top_scores = sims[top_idx].astype(float).tolist()
            labels = None
            if "labels" in self.index:
                labels = [self.index["labels"][i] for i in top_idx]
            return {"indices": top_idx.astype(int).tolist(), "scores": top_scores, "labels": labels}

        # Fallback
        return {"embedding_dim": int(embedding.shape[0])}

    def infer_from_wav_bytes(self, wav_bytes: bytes) -> Dict[str, Any]:
        y = self._read_audio(wav_bytes)
        feat = self._to_features(y)
        model_input = self._prepare_input_for_model(feat)
        embedding = self._forward_embedding(model_input)
        query = self._query_index(embedding)
        return {
            "type": "music",
            "embedding_dim": int(embedding.shape[0]),
            "result": query,
        }

