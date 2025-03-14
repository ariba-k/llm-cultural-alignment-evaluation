class TokenManager:
    OA_TOKEN = ""  # OpenAI API key
    AN_TOKEN = ""  # Anthropic API key
    DI_TOKEN = ""  # DeepInfra API key
    HF_TOKEN = ""  # HuggingFace API key
    MA_TOKEN = ""  # Mistral API key
    GM_TOKEN = ""  # Gemini API key


MODEL_MAP = {
    'LLaMA': [
        "Meta-Llama-3.1-405B-Instruct"
    ],
    'Claude': [
        "claude-3-5-sonnet-20241022"
    ],
    'GPT': [
        "gpt-4o"
    ],
    'Gemini': [
        "gemini-2.0-flash-exp"
    ],
    'Mistral': [
        "mistral-large-latest"
    ]
}

MODEL_NAME_MAP = {
    'LLaMA': 'Llama 3.1 405B',
    'Claude': 'Claude 3.5 Sonnet',
    'GPT': 'GPT-4o',
    'Gemini': 'Gemini 2.0 Flash',
    'Mistral': 'Mistral Large'
}
