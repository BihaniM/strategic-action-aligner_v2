from __future__ import annotations

import json
from typing import Any

import numpy as np
import requests
from sklearn.feature_extraction.text import HashingVectorizer

from src.config import HF_API_BASE, HF_CHAT_MODEL, HF_EMBEDDING_MODEL, HF_TOKEN


class HFClient:
    def __init__(self) -> None:
        self.api_base = HF_API_BASE
        self.token = HF_TOKEN
        self.embedding_model = HF_EMBEDDING_MODEL
        self.chat_model = HF_CHAT_MODEL
        self.last_embedding_backend = "unknown"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _model_urls(self, model_name: str) -> list[str]:
        primary = f"{self.api_base}/{model_name}"
        urls = [primary]

        if self.api_base == "https://api-inference.huggingface.co/models":
            urls.append(f"https://router.huggingface.co/hf-inference/models/{model_name}")

        deduped_urls: list[str] = []
        for url in urls:
            if url not in deduped_urls:
                deduped_urls.append(url)
        return deduped_urls

    def _feature_extraction_urls(self, model_name: str) -> list[str]:
        urls = [f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"]

        if self.api_base == "https://api-inference.huggingface.co/models":
            urls.append(f"https://router.huggingface.co/hf-inference/pipeline/feature-extraction/{model_name}")

        deduped_urls: list[str] = []
        for url in urls:
            if url not in deduped_urls:
                deduped_urls.append(url)
        return deduped_urls

    def _post_model(self, model_name: str, payload: dict[str, Any], timeout: int) -> requests.Response:
        attempted_urls = self._model_urls(model_name)
        last_status: int | None = None
        last_text = ""

        for url in attempted_urls:
            response = requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=timeout,
            )
            if response.ok:
                return response

            last_status = response.status_code
            last_text = response.text[:500]

            if response.status_code in (404, 410):
                continue

            break

        raise RuntimeError(
            f"Hugging Face API request failed for model '{model_name}'. "
            f"Tried URLs: {attempted_urls}. "
            f"Last status: {last_status}. "
            f"Last response: {last_text}"
        )

    def _post_feature_extraction(self, model_name: str, payload: dict[str, Any], timeout: int) -> requests.Response:
        attempted_urls = self._feature_extraction_urls(model_name)
        last_status: int | None = None
        last_text = ""

        for url in attempted_urls:
            response = requests.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=timeout,
            )
            if response.ok:
                return response

            last_status = response.status_code
            last_text = response.text[:500]

            if response.status_code in (404, 410):
                continue

            break

        raise RuntimeError(
            f"Hugging Face feature-extraction request failed for model '{model_name}'. "
            f"Tried URLs: {attempted_urls}. "
            f"Last status: {last_status}. "
            f"Last response: {last_text}"
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        try:
            for text in texts:
                payload = {
                    "inputs": text,
                    "options": {"wait_for_model": True},
                }

                try:
                    response = self._post_model(
                        self.embedding_model,
                        payload,
                        timeout=120,
                    )
                except RuntimeError as exc:
                    error_text = str(exc)
                    if "SentenceSimilarityPipeline.__call__() missing 1 required positional argument: 'sentences'" not in error_text:
                        raise

                    response = self._post_feature_extraction(
                        self.embedding_model,
                        payload,
                        timeout=120,
                    )

                payload = response.json()
                vectors.append(self._normalize_embedding_payload(payload))

            self.last_embedding_backend = "huggingface"
            return vectors
        except Exception:
            self.last_embedding_backend = "local"
            return self._local_embed_texts(texts)

    def _local_embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectorizer = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        matrix = vectorizer.transform(texts)
        return matrix.toarray().astype(float).tolist()

    def _normalize_embedding_payload(self, payload: Any) -> list[float]:
        if isinstance(payload, list) and payload and isinstance(payload[0], (int, float)):
            return [float(value) for value in payload]

        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            matrix = np.array(payload, dtype=float)
            if matrix.ndim == 2:
                return matrix.mean(axis=0).tolist()

        raise ValueError("Unexpected Hugging Face embedding payload format.")

    def generate_json(self, system_prompt: str, user_payload: dict[str, Any], max_new_tokens: int = 700) -> dict[str, Any]:
        prompt = (
            f"{system_prompt}\n"
            "Return strict JSON only.\n"
            f"Input:\n{json.dumps(user_payload, ensure_ascii=False)}"
        )

        response = self._post_model(
            self.chat_model,
            {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                    "return_full_text": False,
                },
                "options": {"wait_for_model": True},
            },
            timeout=180,
        )
        payload = response.json()
        text = self._extract_generated_text(payload)
        return self._extract_json_object(text)

    def _extract_generated_text(self, payload: Any) -> str:
        if isinstance(payload, list) and payload and "generated_text" in payload[0]:
            return str(payload[0]["generated_text"])
        if isinstance(payload, dict) and "generated_text" in payload:
            return str(payload["generated_text"])
        raise ValueError("Unexpected Hugging Face generation payload format.")

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in Hugging Face model output.")
        return json.loads(text[start : end + 1])
