class Parameters:
    num_trials: int = 1
    num_rows: int = 30  # Set to None to process the entire dataset
    contexts: list = ['Job Applicant']  # Specify a list of context ids, e.g., ['context_1', 'context_2']
    likert_scales: list = [5]  # Specify a list of Likert scales, e.g., [3, 5, 7]
    dimensions: list = ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']
    temperature: float = 0
    buffer_tokens: int = 500  # Tokens reserved for model response
    batching_method: str = 'bulk'  # Options: 'bulk', 'incremental', 'none'
    use_reasoning: bool = False
    verbose: bool = True  # Set to True for detailed logging
    batch_size: int = 30 if use_reasoning else num_rows  # Maximum number of comparisons per batch

    original_context: tuple = (
        "There is a hiring manager for an organization focused on improvement and progress.", "Hiring Manager")
    context_file: str = 'data/contexts.xlsx'
    data_folder: str = 'data/cover_letters'
    results_folder: str = "results"

    @classmethod
    def display(cls):
        print("Experiment Parameters:")
        for attr, value in cls.__dict__.items():
            if not attr.startswith("__"):
                print(f"  {attr}: {value}")


DIMENSION_MAP = {
    'pdi': ('High_Power_Distance', 'Low_Power_Distance'),
    'idv': ('Individualism', 'Collectivism'),
    'mas': ('Masculinity', 'Femininity'),
    'uai': ('High_Uncertainty_Avoidance_Index', 'Low_Uncertainty_Avoidance_Index'),
    'lto': ('Long_Term_Orientation', 'Short_Term_Orientation'),
    'ivr': ('Indulgence', 'Restraint')
}

DIMENSION_NAME_MAP = {
    'pdi': 'Power Dist.',
    'idv': 'Indiv./Collect.',
    'mas': 'Masc./Fem.',
    'uai': 'Uncert. Avoid. Index',
    'lto': 'Long/Short Term Orient.',
    'ivr': 'Indulg./Restraint'
}

MODEL_TOKEN_LIMITS = {
    'gpt-4o': 128000,  # 128k tokens
    'Meta-Llama-3.1-405B-Instruct': 32000,  # 32k tokens
    'claude-3-5-sonnet-20241022': 200000,  # 200k tokens
    'gemini-2.0-flash-exp': 1056768,  # 1.05M tokens
    'mistral-large-latest': 128000,  # 128k tokens
    'QwQ-32B-Preview': 32768,  # 32k tokens
}

FAMILY_MODELS_COLORS = {
    'GPT': 'blue',
    'Claude': 'red',
    'LLaMA': 'green',
    'Gemini': 'purple',
    'Mistral': 'orange',
    'Qwen': 'cyan',
}

FAMILY_MODELS_HATCHES = {
    'GPT': 'x',
    'Claude': '+',
    'LLaMA': '/',
    'Gemini': '*',
    'Mistral': '.',
    'Qwen': '|',
}

FAMILY_MODELS_EXTRA_SPACING = {
    'GPT': ['gpt-4o'],
    'Claude': ['claude-3-5-sonnet-20241022'],
    'LLaMA': ['Meta-Llama-3.1-405B-Instruct'],
    'Gemini': ['gemini-2.0-flash-exp'],
    'Mistral': ['Mistral-7B-Instruct-v0.3'],
    'Qwen': ['QwQ-32B-Preview']
}

MODEL_FAMILY_DISPLAY = {
    'LLaMA': 'LLaMA Models',
    'Claude': 'Claude Models',
    'GPT': 'GPT Models',
    'Gemini': 'Gemini Models',
    'Mistral': 'Mistral Models',
    'Qwen': 'QwQ Models',
}


LIKERT_SCALE_MARKERS = {
    2: 'o',  # Circle
    3: 's',  # Square
    4: 'v',  # Downward triangle
    5: 'D',  # Diamond
    6: '*',  # Star
    7: '^'  # Upward triangle
}

ORDER_COLOR_MAP = {
    1: 'blue',
    2: 'green',
    3: 'red'
}
