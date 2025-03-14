import datasets
import ast
import re
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Tuple, Any


def remove_parentheticals(text: str) -> str:
    return text.replace(' (Old national sample)', '').replace(' (Non-national sample)', '').replace(
        ' (Current national sample)', '')


def parse_selection_string(selection_string: str) -> Dict:
    dict_string = re.search(r"defaultdict\(.*?, (.*?)\)$", selection_string).group(1)
    return ast.literal_eval(dict_string)


def parse_option_string(options_string: str) -> List:
    return ast.literal_eval(options_string)


def get_top_countries(data: Any, num_countries: int = 15) -> Tuple[List[str], List[List[str]]]:
    all_selection_countries = [list(parse_selection_string(s).keys()) for s in data['train']['selections']]
    all_selection_countries = [[remove_parentheticals(c) for c in sc] for sc in all_selection_countries]

    all_counts = {}
    for sc in all_selection_countries:
        for c in sc:
            all_counts[c] = all_counts.get(c, 0) + 1

    sorted_counts = dict(sorted(all_counts.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_counts.keys())[:num_countries], all_selection_countries


def filter_selections_by_countries(selections: List[Dict], all_selection_countries: List[List[str]],
                                   top_countries: List[str]) -> List[Dict]:
    selections_top = []
    for s, sc in zip(selections, all_selection_countries):
        filtered_top = {}
        ks = list(s.keys())
        for k, v in s.items():
            clean_k = remove_parentheticals(k)
            if clean_k in top_countries:
                if clean_k in ks and clean_k != k:
                    continue
                else:
                    filtered_top[clean_k] = v
        selections_top.append(filtered_top)
    return selections_top


def process_options_and_selections(options_map: List[Dict], questions: List[str],
                                   selections_top: List[Dict]) -> Tuple[List, List[str], List[Dict], List[int]]:
    options_processed = []
    questions_processed = []
    selections_top_processed = []
    question_indices = []

    for idx, (option_pair, question, st) in enumerate(zip(options_map, questions, selections_top)):
        original_options = option_pair['original_options']
        mapped_options = option_pair['mapped_options']

        if mapped_options != 'reject':
            questions_processed.append(question)
            question_indices.append(idx)

            if mapped_options is None:
                options_processed.append(original_options)
                reordered_indices = list(range(len(original_options)))
            else:
                options_processed.append(mapped_options)
                option_to_index = {opt: idx for idx, opt in enumerate(original_options)}
                try:
                    reordered_indices = [option_to_index[opt] for opt in mapped_options]
                except KeyError as e:
                    print(f"Error: Option '{e.args[0]}' not found in original options for question index {idx}.")
                    continue

            st_means = {}
            for country, counts in st.items():
                reordered_counts = [counts[i] for i in reordered_indices]
                total = np.sum(reordered_counts)
                if total == 0:
                    mean = np.nan
                else:
                    cs_norm = np.array(reordered_counts) / total
                    cs_values = np.linspace(0, 1, len(cs_norm))
                    mean = np.dot(cs_norm, cs_values)
                st_means[country] = mean
            selections_top_processed.append(st_means)

    return options_processed, questions_processed, selections_top_processed, question_indices


def generate_question_variations() -> List[Dict[str, str]]:
    variations = []
    for choice_format in ['numbers', 'letters']:
        for direction_format in ['ascending', 'descending']:
            for response_format in ['identifier_only', 'option_text']:
                variations.append({
                    'choice_format': choice_format,
                    'direction_format': direction_format,
                    'response_format': response_format,
                })
    return variations


def create_dataset(questions_processed: List[str], question_indices: List[int],
                           options_processed: List, selections_top_processed: List[Dict],
                           variations: List[Dict[str, str]]) -> List[Dict]:
    dataset_entries = []

    for idx, (question_idx, question, options_base, st_means) in enumerate(
            zip(question_indices, questions_processed, options_processed, selections_top_processed)):
        option_to_scale = {opt: pos for pos, opt in enumerate(options_base, start=1)}

        for variation in variations:
            options_variant = options_base.copy()

            if variation['direction_format'] == 'descending':
                options_variant = list(reversed(options_variant))

            if variation['choice_format'] == 'numbers':
                labels = [f"{i + 1}." for i in range(len(options_variant))]
            else:
                labels = [f"{chr(65 + i)}." for i in range(len(options_variant))]

            options_labeled = [f"{label} {option}" for label, option in zip(labels, options_variant)]

            label_to_scale = {}
            for label, option in zip(labels, options_variant):
                scale_pos = option_to_scale.get(option)
                if scale_pos is not None:
                    label_to_scale[label.strip('.').upper()] = scale_pos
                else:
                    print(f"Warning: Option '{option}' not found in original options.")

            if variation['response_format'] == 'identifier_only':
                instruction = ("Your task is to carefully evaluate the question and options "
                               "provided by the user and respond only with the most appropriate "
                               "option's identifier (e.g., a letter like 'A', or a number like '1'). "
                               "Do not explain your reasoning, add any additional text, "
                               "or provide context. Only respond with the choice's identifier.")
            elif variation['response_format'] == 'option_text':
                instruction = ("Your task is to carefully evaluate the question and options "
                               "provided by the user and respond only with the full text of the "
                               "most appropriate option. Do not explain your reasoning, add any "
                               "additional text, or provide context. Only respond with the chosen option's text.")
            else:
                instruction = "Select the most appropriate option."

            mean_weights_processed = {}
            for country, value in st_means.items():
                if isinstance(value, np.floating):
                    mean_weights_processed[country] = float(value)
                elif isinstance(value, np.integer):
                    mean_weights_processed[country] = int(value)
                else:
                    mean_weights_processed[country] = value

            dataset_entries.append({
                'question_idx': question_idx,
                'instruction': instruction,
                'question_text': question,
                'options': options_variant,
                'labels': labels,
                'options_labeled': options_labeled,
                'variation': variation,
                'label_to_scale': label_to_scale,
                'mean_weights': mean_weights_processed,
            })

    return dataset_entries


def main() -> None:
    data = datasets.load_dataset('Anthropic/llm_global_opinions')
    with open('data/options_map.json', 'r') as f:
        options_map = json.load(f)

    top_countries, all_selection_countries = get_top_countries(data)

    selections = []
    questions = []
    options_list = []

    for s, o, q, sc in zip(data['train']['selections'],
                           data['train']['options'],
                           data['train']['question'],
                           all_selection_countries):
        if 'refused' not in o.lower():
            if all(c in sc for c in top_countries):
                selections.append(parse_selection_string(s))
                options_list.append(parse_option_string(o))
                questions.append(q)

    selections_top = filter_selections_by_countries(selections, all_selection_countries, top_countries)

    options_processed, questions_processed, selections_top_processed, question_indices = process_options_and_selections(
        options_map, questions, selections_top)

    variations = generate_question_variations()

    dataset = create_dataset(questions_processed, question_indices, options_processed,
                                             selections_top_processed, variations)

    df_dataset = pd.DataFrame(dataset)
    df_dataset.to_csv('data/global_opinion_data.csv', index=False)
    print(f"Dataset saved with {len(df_dataset)} entries")


if __name__ == "__main__":
    main()