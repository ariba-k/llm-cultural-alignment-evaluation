import anthropic
import time
import openai
from mistralai import Mistral
import google.generativeai as genai
from typing import Dict, Optional, Any, Type
from environment import TokenManager


class BaseExperiment:
    MODEL_DELAYS: Dict[str, Dict[str, int]] = {
        "GPT": {"default": 1},
        "Claude": {"default": 1},
        "LLaMA": {"default": 3},
        "Gemini": {"default": 10},
        "Mistral": {"default": 10}
    }

    def __init__(self, model_name: str, model_type: str, temperature: float = 0):
        self.model_name = model_name
        self.model_type = model_type
        self.temperature = temperature
        self.delay = self._get_delay()
        self.client: Any = None
        self.setup_client()

    def _get_delay(self) -> int:
        model_delays = self.MODEL_DELAYS.get(self.model_type, {})
        return model_delays.get(self.model_name, model_delays.get('default', 2))

    def setup_client(self) -> None:
        raise NotImplementedError

    def generate_response(self, prompt: str) -> Optional[str]:
        response = self._make_request(prompt)
        time.sleep(self.delay)
        return response

    def _make_request(self, prompt: str) -> Optional[str]:
        raise NotImplementedError

    def handle_error(self, error: Exception, model_info: str) -> None:
        print(f"Error generating response with {model_info}: {error}")
        return None


def get_experiment_instance(model_type: str, model_name: str) -> BaseExperiment:
    experiment_classes: Dict[str, Type[BaseExperiment]] = {
        "GPT": GPTExperiment,
        "Claude": ClaudeExperiment,
        "LLaMA": LLaMAExperiment,
        "Gemini": GeminiExperiment,
        "Mistral": MistralExperiment,
    }
    if model_type not in experiment_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    return experiment_classes[model_type](model_name=model_name, model_type=model_type)


class GPTExperiment(BaseExperiment):
    def setup_client(self) -> None:
        self.client = openai.OpenAI(api_key=TokenManager.OA_TOKEN)

    def _make_request(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return self.handle_error(e, self.model_name)


class ClaudeExperiment(BaseExperiment):
    def setup_client(self) -> None:
        self.client = anthropic.Anthropic(api_key=TokenManager.AN_TOKEN)

    def _make_request(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=self.temperature,
            )
            return response.content[0].text.strip()
        except Exception as e:
            return self.handle_error(e, self.model_name)


class LLaMAExperiment(BaseExperiment):
    def setup_client(self) -> None:
        self.client = openai.OpenAI(
            api_key=TokenManager.DI_TOKEN,
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def _make_request(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model=f"meta-llama/{self.model_name}",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return self.handle_error(e, self.model_name)


class GeminiExperiment(BaseExperiment):
    def setup_client(self) -> None:
        genai.configure(api_key=TokenManager.GM_TOKEN)
        self.client = genai.GenerativeModel(f"models/{self.model_name}")

    def _make_request(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return self.handle_error(e, f"Gemini model {self.model_name}")


class MistralExperiment(BaseExperiment):
    def setup_client(self) -> None:
        self.client = Mistral(api_key=TokenManager.MA_TOKEN)

    def _make_request(self, prompt: str) -> Optional[str]:
        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return self.handle_error(e, self.model_name)