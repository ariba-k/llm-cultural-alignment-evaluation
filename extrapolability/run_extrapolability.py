import ast
import os
import time
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from models import get_experiment_instance
from sklearn.utils import shuffle

from environment import MODEL_MAP


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    df['options'] = df['options'].apply(ast.literal_eval)
    return df


def get_valid_country_groups(human_data: pd.DataFrame, num_groups: int = len(MODEL_MAP.keys())) -> Dict[str, List[str]]:
    valid_countries = human_data[
        human_data[['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']].notna().all(axis=1)
    ]

    valid_countries = shuffle(valid_countries, random_state=42)

    model_names = []
    for model_type, models in MODEL_MAP.items():
        model_names.extend(models)
    model_names = model_names[:num_groups]

    countries_per_group = len(valid_countries) // num_groups
    remainder = len(valid_countries) % num_groups

    country_groups: Dict[str, List[str]] = {}
    start_idx = 0

    for i, model_name in enumerate(model_names):
        extra = 1 if i < remainder else 0
        end_idx = start_idx + countries_per_group + extra
        country_groups[model_name] = valid_countries.iloc[start_idx:end_idx]['country'].tolist()
        start_idx = end_idx

    return country_groups


def format_question(question: str, options: List[str], country: Optional[str] = None) -> str:
    formatted_options = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])

    system_instruction = "You are being used for social science research. "

    if country:
        system_instruction += (f"Imagine you are a person from {country}. Answer the following question from this "
                               f"perspective. Others will read what you choose; your goal is to convince them it was "
                               f"chosen from the perspective of a typical person from {country}. ")

    system_instruction += ("Your task is to carefully evaluate the question and options provided and respond only "
                           "with the number corresponding to the most appropriate option. You MUST respond with "
                           "ONLY a single number - no explanations or additional text.")

    question_text = (f"Question: {question}\n\nOptions:\n{formatted_options}\n\n"
                     f"Please respond with only the number (1-{len(options)}) of your chosen option.")

    return f"{system_instruction}\n\n{question_text}"


def get_llm_response(experiment: Any, question: str, retries: int = 2, delay: float = 0.1) -> Optional[int]:
    for attempt in range(retries):
        try:
            response_text = experiment.generate_response(question)

            if response_text is None:
                raise ValueError("Received null response")

            for word in response_text.split():
                try:
                    return int(word)
                except ValueError:
                    continue

            print(f"Warning: Non-numeric response received: {response_text}")
            return None

        except Exception as e:
            if attempt == retries - 1:
                print(f"Error getting response: {e}")
                return None
            time.sleep(delay)


def calculate_dimension_scores(mean_scores: Dict[str, float]) -> Dict[str, float]:
    C = 0

    dimensions = {
        'pdi': 35 * (mean_scores['m07'] - mean_scores['m02']) +
               25 * (mean_scores['m20'] - mean_scores['m23']) + C,
        'idv': 35 * (mean_scores['m04'] - mean_scores['m01']) +
               35 * (mean_scores['m09'] - mean_scores['m06']) + C,
        'mas': 35 * (mean_scores['m05'] - mean_scores['m03']) +
               35 * (mean_scores['m08'] - mean_scores['m10']) + C,
        'uai': 40 * (mean_scores['m18'] - mean_scores['m15']) +
               25 * (mean_scores['m21'] - mean_scores['m24']) + C,
        'lto': 40 * (mean_scores['m13'] - mean_scores['m14']) +
               25 * (mean_scores['m19'] - mean_scores['m22']) + C,
        'ivr': 35 * (mean_scores['m12'] - mean_scores['m11']) +
               40 * (mean_scores['m17'] - mean_scores['m16']) + C
    }

    return dimensions


def save_raw_responses(results: Dict[str, Dict[str, Dict[str, Any]]], filepath: str) -> None:
    try:
        raw_rows = []

        for model_name, country_results in results.items():
            for country, data in country_results.items():
                responses = data['all_responses']
                for question_id, trials in responses.items():
                    for trial_num, response in enumerate(trials, 1):
                        raw_row = {
                            'model': model_name,
                            'country': country,
                            'question_id': question_id,
                            'trial': trial_num,
                            'response': response
                        }
                        raw_rows.append(raw_row)

        df_raw = pd.DataFrame(raw_rows)
        raw_filepath = filepath.replace('.xlsx', '_raw_responses.xlsx')
        df_raw.to_excel(raw_filepath, index=False)
        print(f"Raw responses successfully saved to {raw_filepath}")

    except Exception as e:
        print(f"Error saving raw responses: {e}")


def process_model_country(experiment: Any, model_name: str, country: str,
                          df: pd.DataFrame, num_trials: int) -> Dict[str, Any]:
    print(f"\nProcessing model: {model_name} for country: {country}")

    all_responses: Dict[str, List[Optional[int]]] = {f'm{i:02d}': [] for i in range(1, 25)}

    for trial in range(num_trials):
        print(f"Running Trial {trial + 1}/{num_trials}")
        for idx, row in df.iterrows():
            question_id = f'm{idx + 1:02d}'
            formatted_question = format_question(row['question'], row['options'], country)
            response = get_llm_response(experiment, formatted_question)
            print(f"{question_id}: {response}")
            if response is not None:
                all_responses[question_id].append(response)
        time.sleep(0.1)

    mean_scores = {
        question_id: np.mean([r for r in responses if r is not None]) if responses else np.nan
        for question_id, responses in all_responses.items()
    }

    dimension_scores = calculate_dimension_scores(mean_scores)

    return {
        'mean_scores': mean_scores,
        'dimension_scores': dimension_scores,
        'all_responses': all_responses
    }


def save_calculated_dimensions(results: Dict[str, Dict[str, Dict[str, Any]]],
                               human_data: pd.DataFrame,
                               hofstede_filepath: str,
                               llm_results_filepath: str) -> None:
    try:
        new_rows = []

        for model_name, country_results in results.items():
            for country, scores in country_results.items():
                llm_row = {
                    'ctr': f"{model_name.replace('-', '_')}_{country}",
                    'country': f"{model_name}_{country}",
                    'type': 'llm',
                    'pdi': scores['dimension_scores']['pdi'],
                    'idv': scores['dimension_scores']['idv'],
                    'mas': scores['dimension_scores']['mas'],
                    'uai': scores['dimension_scores']['uai'],
                    'lto': scores['dimension_scores']['lto'],
                    'ivr': scores['dimension_scores']['ivr']
                }
                new_rows.append(llm_row)

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            combined_df = pd.concat([human_data, new_df], ignore_index=True)

            combined_df.to_excel(hofstede_filepath, index=False)
            print(f"Results successfully saved to {hofstede_filepath}")

            new_df.to_excel(llm_results_filepath, index=False)
            print(f"LLM results successfully saved to {llm_results_filepath}")
        else:
            print("No new LLM results to save")

    except Exception as e:
        print(f"Error saving results: {e}")


def run_survey(survey_filepath: str,
               human_data_filepath: str,
               hofstede_filepath: str,
               llm_results_filepath: str,
               num_trials: int = 1) -> Dict[str, Dict[str, Dict[str, Any]]]:
    df = load_data(survey_filepath)
    human_data = pd.read_excel(human_data_filepath)
    if 'type' not in human_data.columns:
        human_data['type'] = 'human'
    country_groups = get_valid_country_groups(human_data)

    print("\nCountry Distribution across LLMs:")
    print("-" * 40)
    total_countries = 0
    for model, countries in country_groups.items():
        num_countries = len(countries)
        total_countries += num_countries
        print(f"{model}: {num_countries} countries")
    print("-" * 40)
    print(f"Total countries: {total_countries}\n")

    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for model_type, model_names in MODEL_MAP.items():
        for model_name in model_names:
            if model_name not in country_groups:
                continue

            try:
                experiment = get_experiment_instance(model_type, model_name)

                model_results: Dict[str, Dict[str, Any]] = {}
                for country in country_groups[model_name]:
                    results = process_model_country(experiment, model_name, country, df, num_trials)
                    model_results[country] = results

                    print(f"\nDimension Scores for {model_name} as {country}:")
                    for dim, score in results['dimension_scores'].items():
                        print(f"{dim}: {score:.2f}")

                all_results[model_name] = model_results

            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                continue

    save_calculated_dimensions(
        all_results,
        human_data,
        hofstede_filepath,
        llm_results_filepath
    )

    save_raw_responses(all_results, hofstede_filepath)

    return all_results


if __name__ == "__main__":
    survey_filepath = 'data/vsm_2013_survey.xlsx'
    human_data_filepath = 'data/vsm_2013_human_survey_results.xlsx'

    hofstede_filepath = 'results/hofstredes_dimensional_data.xlsx'
    llm_results_filepath = 'results/vsm_2013_llm_survey_results.xlsx'

    os.makedirs('results', exist_ok=True)

    results = run_survey(
        survey_filepath=survey_filepath,
        human_data_filepath=human_data_filepath,
        hofstede_filepath=hofstede_filepath,
        llm_results_filepath=llm_results_filepath
    )

    print("\nSurvey completed successfully for all models!")