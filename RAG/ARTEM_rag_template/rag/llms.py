from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from openai import OpenAI

from .config import LLMConfig


from transformers import AutoTokenizer, AutoModelForCausalLM


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class OpenAILikeClient(BaseLLMClient):
    """
    Клиент для OpenAI / OpenRouter / self-hosted vLLM и др. openai-суместимых API.
    """

    def __init__(self, model: str, base_url: Optional[str], api_key: str,
                 temperature: float = 0.2, max_tokens: int = 512):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Ты полезный русскоязычный ассистент."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()


class HFTGIClient(BaseLLMClient):
    """
    Клиент для HuggingFace Text Generation Inference (TGI):
      POST {base_url}/generate
      body: {"inputs": prompt, "parameters": {...}}
    """

    def __init__(self, base_url: str, temperature: float = 0.2, max_new_tokens: int = 512,
                 api_key: Optional[str] = None):
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        self.base_url = base_url
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
            },
        }
        resp = httpx.post(f"{self.base_url}/generate", json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # TGI обычно возвращает список dict с "generated_text"
        if isinstance(data, list):
            return data[0].get("generated_text", "").strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        return str(data)


class LocalTransformersClient(BaseLLMClient):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        import torch

        # Устройство
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Выбор dtype
        if dtype is None:
            # По умолчанию на MPS — float32 для стабильности
            if self.device == "mps":
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            if dtype == "float16":
                torch_dtype = torch.float16
            elif dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

        print(f"🔌 Local model: {model_name}, device={self.device}, dtype={torch_dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        import torch

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,  # ⬅ greedy decoding, без multinomial
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()


def create_llm(llm_cfg: LLMConfig) -> BaseLLMClient:
    provider = llm_cfg.provider.lower()
    if provider == "openai":
        api_key = os.environ.get(llm_cfg.api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"API key not found in env var {llm_cfg.api_key_env}. "
                f"Export it first: export {llm_cfg.api_key_env}=..."
            )
        return OpenAILikeClient(
            model=llm_cfg.model,
            base_url=llm_cfg.base_url,
            api_key=api_key,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
        )

    if provider == "tgi":
        base_url = llm_cfg.base_url
        if not base_url:
            raise ValueError("For provider=tgi нужно указать llm.base_url в config.yaml")
        api_key = os.environ.get(llm_cfg.api_key_env, None)
        return HFTGIClient(
            base_url=base_url,
            temperature=llm_cfg.temperature,
            max_new_tokens=llm_cfg.max_tokens,
            api_key=api_key,
        )
    
    if provider == "local":
        return LocalTransformersClient(
            model_name=llm_cfg.model,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            device=llm_cfg.device,
            dtype=llm_cfg.dtype,
        )

    raise ValueError(f"Unknown llm.provider: {llm_cfg.provider}")
