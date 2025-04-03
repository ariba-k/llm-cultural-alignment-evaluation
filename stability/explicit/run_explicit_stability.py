import os

import pandas as pd
import re
import ast
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from models import get_experiment_instance
from environment import MODEL_MAP


def load_data(file_path: str) -> pd.DataFrame:
    df_dataset = pd.read_csv(file_path)
    df_dataset = df_dataset.reset_index().rename(columns={'index': 'row_id'})

    columns_to_parse = ['options', 'labels', 'options_labeled', 'variation', 'label_to_scale', 'mean_weights']
    for column in columns_to_parse:
        df_dataset[column] = df_dataset[column].apply(lambda x: ast.literal_eval(x))

    return df_dataset


def parse_response_identifier_only(response: str, labels: List[str]) -> Optional[str]:
    response = response.strip().upper()

    pattern = r'^([A-Z]|\d+)[\.\)]?(?:\s|$)'
    match = re.match(pattern, response)
    if match:
        return match.group(1)

    for label in labels:
        label_clean = label.strip('.').strip().upper()
        if label_clean in response:
            return label_clean

    return None


def parse_response_option_text(response: str, options: List[str]) -> Union[int, float]:
    response_clean = response.strip()
    pattern = r'^([A-Z]|\d+)[\.\)]?\s*'
    response_text = re.sub(pattern, '', response_clean, flags=re.IGNORECASE).strip()

    # First try exact match (case-insensitive)
    for idx, option in enumerate(options, start=1):
        if response_text.lower() == option.lower():
            return idx

    # Try substring match if exact match fails
    for idx, option in enumerate(options, start=1):
        clean_option = ' '.join(option.lower().split())
        clean_response = ' '.join(response_text.lower().split())

        if clean_option in clean_response or clean_response in clean_option:
            return idx

    # If no match found
    print(f"No match found for response: '{response_text}' among options: {options}")
    return np.nan


def is_refusal(response: str) -> bool:
    refusal_keywords = ['unable to', 'cannot', 'refuse', 'unfortunately', 'no comment', 'cannot provide']
    return any(keyword in response.lower() for keyword in refusal_keywords)


def get_scale_position(selected_label: str, label_to_scale: Dict) -> Optional[int]:
    # Try direct lookup
    scale_pos = label_to_scale.get(selected_label)
    if scale_pos is not None:
        return scale_pos

    # Try mapping alphabetic to numeric if needed
    if selected_label.isalpha():
        numeric_label = str(ord(selected_label) - ord('A') + 1)
        return label_to_scale.get(numeric_label)

    return None


def create_instruction(response_format: str) -> str:
    instructions = {
        'identifier_only': (
            "Your task is to carefully evaluate the question and options "
            "provided by the user and respond only with the most appropriate "
            "option's identifier (e.g., a letter like 'A', or a number like '1'). "
            "Do not explain your reasoning, add any additional text, "
            "or provide context. Only respond with the choice's identifier."
        ),
        'option_text': (
            "Your task is to carefully evaluate the question and options "
            "provided by the user and respond only with the full text of the "
            "most appropriate option. Do not explain your reasoning, add any "
            "additional text, or provide context. Only respond with the chosen option's text."
        )
    }
    return instructions.get(response_format,
                            "Select the most appropriate option. Do not include any explanation or additional context.")


def construct_prompt(row: pd.Series) -> Tuple[str, str]:
    response_format = row['variation'].get('response_format')
    system_instruction = create_instruction(response_format)

    question_text = row['question_text']
    options_labeled = row['options_labeled']

    user_prompt = f"{question_text}\n\n"
    user_prompt += "\n".join(options_labeled)

    prompt_suffix = {
        'identifier_only': (
            "\nPlease select the most appropriate option by specifying only its corresponding "
            "identifier (e.g., 'A' or '1'). Do not include the full text, any explanation, "
            "or additional context."
        ),
        'option_text': (
            "\nPlease select the most appropriate option by providing its exact text as shown "
            "in the options list. Do not include any explanation or additional context."
        )
    }

    user_prompt += prompt_suffix.get(response_format,
                                     "\nPlease select the most appropriate option. Do not include any explanation or "
                                     "additional context.")

    return system_instruction, user_prompt


def collect_model_responses(experiment, df_dataset: pd.DataFrame) -> pd.DataFrame:
    responses = []
    row_ids = []
    total_prompts = len(df_dataset)

    for i, row in df_dataset.iterrows():
        row_id = row['row_id']
        variation = row['variation']
        response_format = variation.get('response_format')

        system_instruction = create_instruction(response_format)
        prompt = construct_prompt(row)[1]  # Only need the user prompt part

        print(f"Collecting response {int(i) + 1}/{total_prompts} (row_id: {row_id})")

        try:
            full_prompt = f"{system_instruction}\n\n{prompt}"
            assistant_reply = experiment.generate_response(full_prompt)

            if assistant_reply is None or is_refusal(assistant_reply):
                raise ValueError("Received refusal response or null response.")

            responses.append(assistant_reply)
            row_ids.append(row_id)
            print(f"Response: {assistant_reply}")
        except Exception as e:
            print(f"Error during API call: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            try:
                # Adjust the prompt to be more direct on retry
                adjusted_prompt = (f"{system_instruction}\n\n{prompt}\n"
                                   f"Please provide only the identifier without any additional text.")
                assistant_reply = experiment.generate_response(adjusted_prompt)

                if assistant_reply is None or is_refusal(assistant_reply):
                    raise ValueError("Received refusal response after retry.")

                responses.append(assistant_reply)
                row_ids.append(row_id)
                print(f"Response after retry: {assistant_reply}")
            except Exception as e:
                print(f"Failed after retry: {e}. Appending NaN.")
                responses.append(np.nan)
                row_ids.append(row_id)

        time.sleep(0.1)  # Rate limiting

    return pd.DataFrame({
        'row_id': row_ids,
        'response_text': responses
    })


def process_responses(df_responses: pd.DataFrame, df_dataset: pd.DataFrame) -> pd.DataFrame:
    df_responses = df_responses.merge(
        df_dataset[['row_id', 'question_idx', 'variation', 'label_to_scale', 'options', 'labels']],
        on='row_id',
        how='left'
    )

    parsed_selections = []

    for _, row in df_responses.iterrows():
        response = row['response_text']
        variation = row['variation']
        response_format = variation.get('response_format')
        selected_label = None
        scale_pos = None

        if pd.isna(response) or is_refusal(response):
            pass  # Leave selected_label and scale_pos as None
        elif response_format == 'identifier_only':
            labels_clean = [label.strip('.').strip().upper() for label in row['labels']]
            selected_label = parse_response_identifier_only(response, labels_clean)

            if selected_label:
                scale_pos = get_scale_position(selected_label, row['label_to_scale'])
                if scale_pos is None:
                    print(
                        f"Warning: Could not map label '{selected_label}' to scale position for row_id {row['row_id']}")
            else:
                print(f"Warning: Could not parse response '{response}' for row_id {row['row_id']}")

        elif response_format == 'option_text':
            scale_pos = parse_response_option_text(response, row['options'])

            if not pd.isna(scale_pos):
                labels_clean = [label.strip('.').strip().upper() for label in row['labels']]
                selected_label = parse_response_identifier_only(response, labels_clean)

                if not selected_label and 0 <= scale_pos - 1 < len(row['labels']):
                    selected_label = row['labels'][scale_pos - 1].strip('.').strip()
            else:
                print(f"Warning: Could not match response '{response}' to any option for row_id {row['row_id']}")

        parsed_selections.append({
            'row_id': row['row_id'],
            'question_idx': row['question_idx'],
            'variation': variation,
            'selected_label': selected_label,
            'scale_position': scale_pos if scale_pos is not None else np.nan,
            'response_text': response,
        })

    return pd.DataFrame(parsed_selections)


def main(num_rows: Optional[int] = None):
    df_dataset = load_data('data/global_opinion_data.csv')
    if num_rows:
        df_dataset = df_dataset.head(num_rows)

    for model_type, model_names in MODEL_MAP.items():
        for model_name in model_names:
            print(f"\nProcessing model: {model_type} - {model_name}")

            try:
                experiment = get_experiment_instance(model_type, model_name)
                df_responses = collect_model_responses(experiment, df_dataset)
                df_parsed = process_responses(df_responses, df_dataset)
                os.makedirs('results', exist_ok=True)
                output_file = f'results/{model_name.replace("-", "_")}_responses_explicit_stability.csv'
                df_parsed.to_csv(output_file, index=False)

                print(f"Processed responses saved successfully to '{output_file}'.")

            except Exception as e:
                print(f"Error processing model {model_type} - {model_name}: {e}")
                continue  # Continue with next model if one fails

            print(f"Completed processing for {model_type} - {model_name}")


if __name__ == "__main__":
    main()
