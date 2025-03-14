import tiktoken
from transformers import AutoTokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

import re
from stability.implicit.constants_implicit_stability import DIMENSION_MAP
import google.generativeai as genai
from environment import TokenManager


def extract_likert_n(likert_scale):
    try:
        return int(likert_scale)
    except (ValueError, TypeError):
        print(f"Invalid likert_scale value: {likert_scale}")
        return None


def extract_rating_from_string(rating_str, label_to_rating, likert_scale):
    # Attempt to find numerical rating anywhere in the string
    match = re.search(r'\b([1-{}])\b'.format(likert_scale), rating_str)
    if match:
        return int(match.group(1))
    else:
        # Normalize and match verbal label
        normalized_label = re.sub(r'\s+', ' ', rating_str.strip().lower())
        if normalized_label in label_to_rating:
            return label_to_rating[normalized_label]
    return None


def get_label_to_rating_mapping(likert_scale: int, dimension):
    labels = get_likert_scale_data(likert_scale, dimension, output='labels')
    label_to_rating = {}
    for idx, label in enumerate(labels, 1):
        # Normalize the label: convert to lower case and remove extra spaces
        normalized_label = re.sub(r'\s+', ' ', label.strip().lower())
        label_to_rating[normalized_label] = idx
    return label_to_rating


def get_likert_scale_data(likert_scale: int, dimension, output="description"):
    if dimension in DIMENSION_MAP:
        dim_a, dim_b = DIMENSION_MAP[dimension]
    else:
        dim_a, dim_b = dimension
    base_labels = {
        2: [f"Prefer {dim_a}", f"Prefer {dim_b}"],
        3: [f"Prefer {dim_a}", "No Preference", f"Prefer {dim_b}"],
        4: [f"Strongly prefer {dim_a}", f"Somewhat prefer {dim_a}", f"Somewhat prefer {dim_b}",
            f"Strongly prefer {dim_b}"],
        5: [f"Strongly prefer {dim_a}", f"Somewhat prefer {dim_a}", "No Preference", f"Somewhat prefer {dim_b}",
            f"Strongly prefer {dim_b}"],
        6: [f"Strongly prefer {dim_a}", f"Moderately prefer {dim_a}", f"Slightly prefer {dim_a}",
            f"Slightly prefer {dim_b}", f"Moderately prefer {dim_b}", f"Strongly prefer {dim_b}"],
        7: [f"Strongly prefer {dim_a}", f"Moderately prefer {dim_a}", f"Slightly prefer {dim_a}", "No Preference",
            f"Slightly prefer {dim_b}", f"Moderately prefer {dim_b}", f"Strongly prefer {dim_b}"]
    }
    labels = base_labels.get(likert_scale)
    if output == "labels":
        return labels if labels else ["Unsupported Likert scale"]
    if labels:
        return "\n".join([f"{i + 1}. {label}" for i, label in enumerate(labels)])
    return "Unsupported Likert scale."


def get_likert_scale_colors(likert_scale: int):
    if likert_scale == 2:
        colors = ['#d73027', '#4575b4']  # Red to Blue
    elif likert_scale == 3:
        colors = ['#d73027',
                  '#ffffbf',
                  '#4575b4']  # Red, Yellow, Blue
    elif likert_scale == 4:
        colors = ['#d73027',
                  '#fc8d59',
                  '#91bfdb',
                  '#4575b4']
    elif likert_scale == 5:
        colors = [
            '#d73027',  # Strongly prefer Cover Letter A
            '#fc8d59',  # Somewhat prefer Cover Letter A
            '#ffffbf',  # No Preference
            '#91bfdb',  # Somewhat prefer Cover Letter B
            '#4575b4',  # Strongly prefer Cover Letter B
        ]
    elif likert_scale == 6:
        colors = [
            '#d73027',  # Strongly prefer Cover Letter A
            '#fc8d59',  # Moderately prefer Cover Letter A
            '#fee08b',  # Slightly prefer Cover Letter A
            '#91bfdb',  # Slightly prefer Cover Letter B
            '#4575b4',  # Moderately prefer Cover Letter B
            '#313695',  # Strongly prefer Cover Letter B
        ]
    elif likert_scale == 7:
        colors = [
            '#d73027',  # Strongly prefer Cover Letter A
            '#fc8d59',  # Moderately prefer Cover Letter A
            '#fee08b',  # Slightly prefer Cover Letter A
            '#ffffbf',  # No Preference
            '#91bfdb',  # Slightly prefer Cover Letter B
            '#4575b4',  # Moderately prefer Cover Letter B
            '#313695',  # Strongly prefer Cover Letter B
        ]
    else:
        raise ValueError(f"Unsupported Likert scale: {likert_scale}")

    return colors


def count_tokens(model_name: str, text: str):
    if 'gpt' in model_name.lower():
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    elif 'claude' in model_name.lower():
        return len(text.split())
    elif 'llama' in model_name.lower():
        try:
            tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model_name}")
            tokens = tokenizer.tokenize(text)
            return len(tokens)
        except Exception as e:
            print(f"Could not load LLaMA tokenizer: {e}")
            return int(len(text.split()) * 1.5)
    elif 'mistral' in model_name.lower():
        try:
            tokenizer = MistralTokenizer.from_model("mistral-large-2411", strict=True)
            tokens = tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=True, eos=True)
            return len(tokens)
        except Exception as e:
            print(f"Could not load Mistral tokenizer: {e}")
            return int(len(text.split()) * 1.5)
    elif 'gemini' in model_name.lower():
        genai.configure(api_key=TokenManager.GM_TOKEN)
        model = genai.GenerativeModel(f'models/{model_name}')
        tokens = model.count_tokens(text)
        return int(tokens.total_tokens)
    else:
        return len(text.split())
