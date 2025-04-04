import os
import random
import re
from typing import Dict, List, Tuple, Any

import anthropic
import google.generativeai as genai
import openai
import pandas as pd
from mistralai import Mistral

from environment import TokenManager
from constants import MODEL_TOKEN_LIMITS, DIMENSION_MAP
from stability.implicit.utils_implicit_stability import (
    get_likert_scale_data, count_tokens, extract_rating_from_string, get_label_to_rating_mapping)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(42)


def get_experiment_instance(model_name: str, model_type: str, **kwargs) -> 'ModelExperiment':
    experiment_classes = {
        "GPT": GPTExperiment,
        "Claude": ClaudeExperiment,
        "LLaMA": LLaMAExperiment,
        "Gemini": GeminiExperiment,
        "Mistral": MistralExperiment
    }
    if model_type not in experiment_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    return experiment_classes[model_type](model_name=model_name, model_type=model_type, **kwargs)


class ModelExperiment:
    def __init__(self, model_name: str, model_type: str, temperature: float, context: Tuple[str, str],
                 likert_scale: int, batching_method: str, buffer_tokens: int, verbose: bool,
                 use_reasoning: bool, batch_size: int):
        self.model_name = model_name
        self.model_type = model_type
        self.temperature = temperature
        self.context_text, self.context_id = context
        self.likert_scale = likert_scale
        self.buffer_tokens = buffer_tokens
        self.verbose = verbose
        self.batching_method = batching_method
        self.use_reasoning = use_reasoning
        self.batch_size = batch_size
        self.detailed_results: List[Dict[str, Any]] = []
        self.token_limit = MODEL_TOKEN_LIMITS.get(self.model_name, 2048)

        if self.batching_method == 'incremental':
            self.conversation_history: List[Dict[str, str]] = []
        elif self.batching_method == 'bulk':
            self.batch_data: List[Tuple[str, str]] = []
            self.batch_mappings: List[Dict[str, Any]] = []
            self.current_batch_tokens = 0
            self.batch_number = 1
        elif self.batching_method == 'none':
            pass
        else:
            raise ValueError(f"Invalid batching_method: {self.batching_method}")

    def generate_response(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def build_incremental_prompt(self, new_comparison: Dict[str, str]) -> str:
        scale_descriptions = get_likert_scale_data(self.likert_scale, ("Cover Letter A", "Cover Letter B"),
                                                   output='description')

        prompt_parts = [
            self.context_text + "\n\n",
            f"The rating is based on a scale from 1 to {self.likert_scale}:\n{scale_descriptions}\n\n"
        ]

        for idx, qa_pair in enumerate(self.conversation_history, 1):
            prompt_parts.append(f"Q{idx}: {qa_pair['question']}\nA{idx}: {qa_pair['answer']}\n\n")

        prompt_parts.append(f"Q{len(self.conversation_history) + 1}: {new_comparison['question']}\n")

        if self.use_reasoning:
            prompt_parts.append(
                "Please reason through your answer in 2-3 sentences before providing the rating.\n"
                f"A{len(self.conversation_history) + 1}:"
            )
        else:
            prompt_parts.append(f"A{len(self.conversation_history) + 1}:")

        return "".join(prompt_parts)

    def default_prompting_batch(self, batch_data: List[Tuple[str, str]]) -> str:
        scale_descriptions = get_likert_scale_data(self.likert_scale, ("Cover Letter A", "Cover Letter B"),
                                                   output='description')

        prompt_parts = [
            self.context_text + "\n\n",
            f"The rating is based on a scale from 1 to {self.likert_scale}:\n{scale_descriptions}\n\n"
        ]

        if self.use_reasoning:
            prompt_parts.append(
                "Compare the following pairs of cover letters and rate them accordingly.\n"
                "For each pair, provide your reasoning in 2-3 sentences, followed by your rating.\n"
                "Use the following format for each pair:\n\n"
                "Pair [Number]\n"
                "Reasoning:\n"
                "[Your reasoning here]\n"
                "Rating:\n"
                "[Rating Number] - [Verbal Label]\n\n"
                "Only include the reasoning and rating for each pair, and nothing else.\n\n"
            )
        else:
            prompt_parts.append(
                "Compare the following pairs of cover letters and rate them accordingly.\n"
                "Please provide the rating for each pair using the format:\n"
                "Pair [Number] Rating: [Rating Number] - [Verbal Label]\n"
                "Only include the ratings and nothing else.\n"
                "For example:\n"
                "Pair 1 Rating: 3 - No Preference\n"
                "Pair 2 Rating: 1 - Strongly prefer Cover Letter A\n\n"
            )

        for idx, (cover_letter_a, cover_letter_b) in enumerate(batch_data, 1):
            prompt_parts.append(
                f"Pair {idx}\n"
                f"Cover Letter A:\n{cover_letter_a}\n\n"
                f"Cover Letter B:\n{cover_letter_b}\n\n"
            )

        return "".join(prompt_parts)

    def build_single_prompt(self, cover_letter_a: str, cover_letter_b: str) -> str:
        scale_descriptions = get_likert_scale_data(self.likert_scale, ("Cover Letter A", "Cover Letter B"),
                                                   output='description')

        prompt_parts = [
            self.context_text + "\n\n",
            f"The rating is based on a scale from 1 to {self.likert_scale}:\n{scale_descriptions}\n\n",
            "Compare the following pair of cover letters and rate them accordingly.\n",
            f"Cover Letter A:\n{cover_letter_a}\n\n"
            f"Cover Letter B:\n{cover_letter_b}\n\n"
        ]

        if self.use_reasoning:
            prompt_parts.append(
                "Please reason through your answer in 2-3 sentences before providing the rating.\n"
                "Use the following format:\n\n"
                "Reasoning:\n"
                "[Your reasoning here]\n"
                "Rating:\n"
                "[Rating Number] - [Verbal Label]\n\n"
                "Only include the reasoning and rating, and nothing else.\n"
            )
        else:
            prompt_parts.append(
                "Please provide the rating using the format:\n"
                "Rating: [Rating Number] - [Verbal Label]\n"
                "Only include the rating and nothing else.\n"
                "For example:\n"
                "Rating: 3 - No Preference\n"
            )

        return "".join(prompt_parts)

    def generate_model_selections(self, dataset: pd.DataFrame, num_trials: int, dimension: str) -> Dict[
        str, List[Dict[str, int]]]:
        combined_model_selections = {dimension: []}

        if self.verbose:
            print("=" * 100)
            print(f"Running experiment for model: {self.model_name}")
            print(f"Dimension: {dimension}")
            print(f"Context: {self.context_id} = {self.context_text}")
            print(f"Batching Method: {self.batching_method}")
            print(f"Likert Scale: {self.likert_scale}")
            print(f"Use Reasoning: {self.use_reasoning}")
            print("=" * 100 + "\n")

        for trial_num in range(1, num_trials + 1):
            if self.verbose:
                print(f"Starting Trial {trial_num}")

            model_selections = {dimension: []}

            if self.batching_method == 'bulk':
                self.batch_data = []
                self.batch_mappings = []
                self.current_batch_tokens = 0
                self.batch_number = 1

            if self.batching_method == 'incremental':
                self.conversation_history = []

            for question_num, (_, row) in enumerate(dataset.iterrows(), 1):
                dim_a, dim_b = DIMENSION_MAP[dimension]
                letter_a_orig = row.get(f'{dim_a}_Cover_Letter')
                letter_b_orig = row.get(f'{dim_b}_Cover_Letter')

                if letter_a_orig is None or letter_b_orig is None:
                    if self.verbose:
                        print(f"  Warning: Missing cover letters for dimension '{dimension}' in row {question_num}.")
                    continue

                choice_a, choice_b, mapping = self._randomize_letters(letter_a_orig, letter_b_orig, dim_a, dim_b)

                comparison_mapping = {
                    'dimension': dimension,
                    'cover_letter_mapping': mapping,
                    'dim_parts': [dim_a, dim_b],
                    'question_num': question_num,
                    'trial_num': trial_num
                }

                if self.batching_method == 'incremental':
                    new_comparison = {
                        'question': f"Compare the following pair of cover letters.\n\n"
                                    f"Cover Letter A:\n{choice_a}\n\n"
                                    f"Cover Letter B:\n{choice_b}\n\n"
                                    "Your rating:"
                    }
                    prompt = self.build_incremental_prompt(new_comparison)
                    total_tokens = count_tokens(self.model_name, prompt) + self.buffer_tokens

                    if total_tokens > self.token_limit:
                        if self.verbose:
                            print(f"Token limit reached. Starting a new conversation.")
                        self.conversation_history = []
                        prompt = self.build_incremental_prompt(new_comparison)
                        total_tokens = count_tokens(self.model_name, prompt) + self.buffer_tokens

                        if total_tokens > self.token_limit:
                            if self.verbose:
                                print(f"Prompt still exceeds token limit even after resetting conversation history.")
                                print(f"Skipping this comparison (Question {question_num}, Trial {trial_num}).")
                            continue

                    model_response = self.generate_response(prompt)
                    self.conversation_history.append({
                        'question': new_comparison['question'],
                        'answer': model_response
                    })
                    self.parse_response(model_response, comparison_mapping, model_selections)

                elif self.batching_method == 'bulk':
                    temp_batch_data = self.batch_data + [(choice_a, choice_b)]
                    temp_prompt = self.default_prompting_batch(temp_batch_data)
                    total_estimated_tokens = count_tokens(self.model_name, temp_prompt) + self.buffer_tokens

                    if total_estimated_tokens > self.token_limit or len(self.batch_data) >= self.batch_size:
                        if self.batch_data:
                            self.process_batch(model_selections)
                            self.batch_data = []
                            self.batch_mappings = []

                        self.batch_data.append((choice_a, choice_b))
                        self.batch_mappings.append(comparison_mapping)
                        if self.verbose:
                            print(f"  Starting new Batch {self.batch_number} with Pair {len(self.batch_data)}.")
                    else:
                        self.batch_data.append((choice_a, choice_b))
                        self.batch_mappings.append(comparison_mapping)
                        if self.verbose:
                            print(f"  Adding Pair {len(self.batch_data)} to Batch {self.batch_number}.")

                elif self.batching_method == 'none':
                    prompt = self.build_single_prompt(choice_a, choice_b)
                    total_tokens = count_tokens(self.model_name, prompt) + self.buffer_tokens

                    if total_tokens > self.token_limit:
                        if self.verbose:
                            print(f"Prompt exceeds token limit. "
                                  f"Skipping this comparison (Question {question_num}, Trial {trial_num}).")
                        continue

                    model_response = self.generate_response(prompt)
                    self.parse_response(model_response, comparison_mapping, model_selections)
                else:
                    raise ValueError(f"Invalid batching_method: {self.batching_method}")

            if self.batching_method == 'bulk' and self.batch_data:
                self.process_batch(model_selections)

            if self.verbose:
                print(f"Completed Trial {trial_num}\n")

            for i, selection in enumerate(model_selections[dimension]):
                if len(combined_model_selections[dimension]) <= i:
                    combined_model_selections[dimension].append(
                        {f'scale_{point}': 0 for point in range(1, self.likert_scale + 1)}
                    )
                for key in selection.keys():
                    combined_model_selections[dimension][i][key] += selection[key]

        return combined_model_selections

    def process_batch(self, model_selections: Dict[str, List[Dict[str, int]]]) -> None:
        if self.batching_method != 'bulk':
            return

        if self.verbose:
            print(f"  Sending Batch {self.batch_number}: {len(self.batch_data)} comparisons.")

        prompt = self.default_prompting_batch(self.batch_data)
        self.current_batch_tokens = count_tokens(self.model_name, prompt)

        model_response = self.generate_response(prompt)

        if self.verbose:
            print(f"  Received response for Batch {self.batch_number}:\n{model_response}\n")

        success = self.parse_batched_responses(model_response, model_selections)

        if not success and self.verbose:
            print(f"  Warning: Not all responses were successfully parsed in Batch {self.batch_number}.")

        self.batch_data = []
        self.batch_mappings = []
        self.current_batch_tokens = 0
        self.batch_number += 1

    def parse_batched_responses(self, batch_response: str, model_selections: Dict[str, List[Dict[str, int]]]) -> bool:
        success = True
        label_to_rating = get_label_to_rating_mapping(self.likert_scale, ("Cover Letter A", "Cover Letter B"))
        pairs = re.split(r'\n(?=Pair\s*\d+[:]?)', batch_response.strip(), flags=re.IGNORECASE)

        for pair_text in pairs:
            pair_number_match = re.match(r'Pair\s*(\d+)', pair_text.strip(), re.IGNORECASE)
            if not pair_number_match:
                if self.verbose:
                    print("  Warning: Could not find pair number in the text.")
                continue

            idx = int(pair_number_match.group(1))
            if idx > len(self.batch_mappings):
                if self.verbose:
                    print(f"  Warning: Pair number {idx} exceeds number of mappings.")
                continue

            mapping = self.batch_mappings[idx - 1]
            dimension = mapping['dimension']
            question_num = mapping['question_num']
            trial_num = mapping['trial_num']

            num_to_add = question_num - len(model_selections[dimension])
            if num_to_add > 0:
                for _ in range(num_to_add):
                    model_selections[dimension].append(
                        {f'scale_{point}': 0 for point in range(1, self.likert_scale + 1)}
                    )

            rating_found = False
            response_int = None

            if self.use_reasoning:
                reasoning_match = re.search(
                    r'Reasoning\s*:\s*(.*?)\s*Rating\s*:', pair_text, re.IGNORECASE | re.DOTALL
                )
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ''
                rating_match = re.search(r'Rating\s*[:\-]?\s*(.*)', pair_text, re.IGNORECASE)
            else:
                reasoning = ''
                rating_match = re.search(r'Rating\s*[:\-]?\s*(.*)', pair_text, re.IGNORECASE)

            if rating_match:
                rating_str = rating_match.group(1).strip()
                response_int = extract_rating_from_string(rating_str, label_to_rating, self.likert_scale)
                rating_found = response_int is not None
            else:
                if self.verbose:
                    print(f"  Warning: No rating found for Pair {idx} in the batch.")
                success = False

            if rating_found and response_int is not None:
                if self.verbose:
                    print(f"    Extracted rating for Pair {idx}: {response_int}")
                if 1 <= response_int <= self.likert_scale:
                    adjusted_response = self.adjust_response(
                        response_int, mapping['cover_letter_mapping'], mapping['dim_parts']
                    )
                    model_selections[dimension][question_num - 1][f'scale_{adjusted_response}'] += 1
            else:
                if self.verbose:
                    print(f"  Warning: No valid rating found for Pair {idx} in the batch.")
                success = False

            self.detailed_results.append({
                'trial_num': trial_num,
                'question_num': question_num,
                'dimension': dimension,
                'cover_letter_mapping': str(mapping['cover_letter_mapping']),
                'rating': response_int if rating_found else None,
                'reasoning': reasoning if self.use_reasoning else None,
                'likert_scale': self.likert_scale,
                'batching_method': self.batching_method,
                'model_name': self.model_name,
                'model_type': self.model_type,
                "context_id": self.context_id,
                'context_text': self.context_text
            })
        return success

    def parse_response(self, model_response: str, mapping: Dict[str, Any],
                       model_selections: Dict[str, List[Dict[str, int]]]) -> None:
        if self.verbose:
            print(
                f"    Parsing response for Question {mapping['question_num']}, "
                f"Trial {mapping['trial_num']} on Dimension {mapping['dimension']}."
            )

        reasoning = ''
        rating_found = False
        response_int = None
        label_to_rating = get_label_to_rating_mapping(self.likert_scale, ("Cover Letter A", "Cover Letter B"))

        if self.use_reasoning:
            reasoning_match = re.search(
                r'Reasoning\s*:\s*(.*?)\s*Rating\s*:', model_response, re.IGNORECASE | re.DOTALL
            )
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
            else:
                reasoning = ''
                if self.verbose:
                    print("      No reasoning found in the response.")

            rating_match = re.search(r'Rating\s*[:\-]?\s*(.*)', model_response, re.IGNORECASE)
            if rating_match:
                rating_str = rating_match.group(1).strip()
                response_int = extract_rating_from_string(rating_str, label_to_rating, self.likert_scale)
                rating_found = response_int is not None
            else:
                if self.verbose:
                    print("      No rating found in the response.")
        else:
            patterns = [
                r'Rating\s*[:\-]?\s*([1-{}])\b'.format(self.likert_scale),
                r'Rating\s*[:\-]?\s*(.*)',
                r'\b([1-{}])\b'.format(self.likert_scale),
                r'(Strongly prefer Cover Letter [AB])',
                r'(Somewhat prefer Cover Letter [AB])',
                r'(Moderately prefer Cover Letter [AB])',
                r'(Slightly prefer Cover Letter [AB])',
                r'(Prefer Cover Letter [AB])',
                r'(No Preference)'
            ]

            for pattern in patterns:
                match = re.search(pattern, model_response, re.IGNORECASE)
                if match:
                    rating_str = match.group(1).strip()
                    response_int = extract_rating_from_string(rating_str, label_to_rating, self.likert_scale)
                    rating_found = response_int is not None
                    if rating_found:
                        break

        self.detailed_results.append({
            'trial_num': mapping['trial_num'],
            'question_num': mapping['question_num'],
            'dimension': mapping['dimension'],
            'cover_letter_mapping': str(mapping['cover_letter_mapping']),
            'rating': response_int if rating_found else None,
            'reasoning': reasoning if self.use_reasoning else None,
            'likert_scale': self.likert_scale,
            'batching_method': self.batching_method,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'context': self.context_text
        })

        if rating_found and response_int is not None:
            if self.verbose:
                print(f"      Extracted rating: {response_int}")
            if 1 <= response_int <= self.likert_scale:
                self._update_model_selections(response_int, mapping, model_selections)
            else:
                if self.verbose:
                    print("      Rating out of range.")
                    print(f"      Model response was: {model_response}")
                self._append_default_selection(mapping, model_selections)
        else:
            if self.verbose:
                print("      No valid rating found in the response.")
                print(f"      Model response was: {model_response}")
            self._append_default_selection(mapping, model_selections)

    def _append_default_selection(self, mapping: Dict[str, Any],
                                  model_selections: Dict[str, List[Dict[str, int]]]) -> None:
        dimension = mapping['dimension']
        question_num = mapping['question_num']

        while len(model_selections[dimension]) < question_num:
            model_selections[dimension].append(
                {f'scale_{point}': 0 for point in range(1, self.likert_scale + 1)}
            )

        if self.verbose:
            print(f"      Appended default selection for Question {question_num}.")

    def _update_model_selections(self, response_int: int, mapping: Dict[str, Any],
                                 model_selections: Dict[str, List[Dict[str, int]]]) -> None:
        adjusted_response = self.adjust_response(
            response_int, mapping['cover_letter_mapping'], mapping['dim_parts']
        )

        dimension = mapping['dimension']
        question_num = mapping['question_num']

        if len(model_selections[dimension]) < question_num:
            model_selections[dimension].append(
                {f'scale_{point}': 0 for point in range(1, self.likert_scale + 1)}
            )

        model_selections[dimension][question_num - 1][f'scale_{adjusted_response}'] += 1

    def adjust_response(self, response_int: int, cover_letter_mapping: Dict[str, str], dim_parts: List[str]) -> int:
        if cover_letter_mapping['A'] == dim_parts[0] and cover_letter_mapping['B'] == dim_parts[1]:
            return response_int
        else:
            return self.likert_scale - response_int + 1

    def _randomize_letters(self, letter_a_orig: str, letter_b_orig: str, dim_a: str, dim_b: str) -> Tuple[
        str, str, Dict[str, str]]:
        if random.choice([True, False]):
            return letter_a_orig, letter_b_orig, {'A': dim_a, 'B': dim_b}
        else:
            return letter_b_orig, letter_a_orig, {'A': dim_b, 'B': dim_a}


class GPTExperiment(ModelExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = openai.OpenAI(api_key=TokenManager.OA_TOKEN)

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()


class ClaudeExperiment(ModelExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = anthropic.Anthropic(api_key=TokenManager.AN_TOKEN)

    def generate_response(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=self.temperature,
        )
        return response.content[0].text.strip()


class LLaMAExperiment(ModelExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = openai.OpenAI(
            api_key=TokenManager.DI_TOKEN,
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=f"meta-llama/{self.model_name}",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response with {self.model_name}: {e}")
            return ""


class GeminiExperiment(ModelExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        genai.configure(api_key=TokenManager.GM_TOKEN)
        self.client = genai.GenerativeModel(f"models/{self.model_name}")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating response with Gemini model {self.model_name}: {e}")
            return ""


class MistralExperiment(ModelExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = Mistral(api_key=TokenManager.MA_TOKEN)

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response with {self.model_name}: {e}")
            return ""
