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