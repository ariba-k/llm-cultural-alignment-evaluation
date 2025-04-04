import pandas as pd
import os
import re
from typing import Tuple, Optional, List, Dict, Any
from stability.implicit.constants_implicit_stability import Parameters
from constants import DIMENSION_MAP
from environment import MODEL_MAP
from models import get_experiment_instance


def extract_rating_explanation(response_text: str, include_reasoning: bool = True,
                               verbose: bool = False) -> Tuple[Optional[str], Optional[str]]:
    try:
        cleaned_text = response_text.strip()

        patterns = [
            r'Reasoning:[\s\n]+(.*?)[\s\n]+Rating:[\s\n]*(\d+)',
            r'Rating:[\s\n]*(\d+)',
            r'(?:^|\n)\s*(\d+)\s*$'
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned_text, re.DOTALL)
            if match:
                if verbose:
                    print(f"\nDEBUG - Found match with pattern: {pattern}")

                if include_reasoning and 'Reasoning' in pattern:
                    explanation = ' '.join(match.group(1).strip().split())
                    if explanation == "[Your reasoning here]":
                        explanation = None
                    rating = match.group(2)
                elif 'Rating' in pattern:
                    explanation = None
                    rating = match.group(1)
                else:  # Simple number format
                    explanation = None
                    rating = match.group(1)

                rating = ''.join(filter(str.isdigit, rating))
                if not rating.isdigit() or not (1 <= int(rating) <= 5):
                    if verbose:
                        print(f"WARNING: Rating {rating} outside expected range 1-5")
                    continue

                return rating, explanation

        if verbose:
            print(f"WARNING: Could not find pattern in response: {response_text[:100]}...")
        return None, None

    except Exception as e:
        print(f"ERROR: Could not extract rating and explanation: {e}")
        return None, None


def rate_single_cover_letter(experiment: Any, context: str, letter_text: str,
                             include_reasoning: bool, verbose: bool = False) -> Tuple[Optional[str], Optional[str]]:
    prompt = (
        f"{context}Rate the following cover letter using a 5-point scale, where:\n"
        "1. Not likely at all\n"
        "2. Somewhat unlikely\n"
        "3. Neutral\n"
        "4. Somewhat likely\n"
        "5. Very likely\n\n"
    )

    prompt += (
        "Provide a 2-3 sentence explanation of your rating, then the rating in this format:\n"
        "Reasoning: [2-3 sentences explaining your rating]\n"
        "Rating: [Number]\n\n"
    ) if include_reasoning else (
        "Provide only the rating in this format:\n"
        "Rating: [Number]\n\n"
    )

    prompt += f"**Candidate:**\n{letter_text}"

    if verbose:
        print("\n" + "=" * 80 + f"\nPROMPT:\n{prompt}\n" + "=" * 80 + "\n")

    try:
        rating_response = experiment.generate_response(prompt)

        if verbose and rating_response:
            print("\nMODEL RESPONSE:\n" + rating_response + "\n" + "=" * 80 + "\n")

        if rating_response is None:
            print("ERROR: Received None response from model")
            return None, None

        rating, explanation = extract_rating_explanation(rating_response, include_reasoning, verbose)
        return rating, explanation

    except Exception as e:
        print(f"ERROR: Error during processing: {e}")
        return None, None


def extract_batch_responses(response_text: str, include_reasoning: bool = True,
                            verbose: bool = False) -> List[Tuple[str, str, str, str]]:
    try:
        candidates = re.split(r'Candidate [AB]:', response_text)
        candidates = [c.strip() for c in candidates if c.strip()]

        if len(candidates) >= 2:
            rating1, explanation1 = extract_rating_explanation(candidates[0], include_reasoning, verbose)
            rating2, explanation2 = extract_rating_explanation(candidates[1], include_reasoning, verbose)
            return [(rating1, explanation1, rating2, explanation2)]
        return []

    except Exception as e:
        print(f"ERROR: Could not extract batch responses: {e}")
        return []


def rate_cover_letters(experiment: Any, context: str, letter_text1: str, letter_text2: str,
                       include_reasoning: bool, verbose: bool = False) -> Tuple[
    Tuple[Optional[str], Optional[str]],
    Tuple[Optional[str], Optional[str]]
]:
    prompt = (
        f"{context}Rate the following cover letters using a 5-point scale, where:\n"
        "1. Not likely at all\n"
        "2. Somewhat unlikely\n"
        "3. Neutral\n"
        "4. Somewhat likely\n"
        "5. Very likely\n\n"
    )

    if include_reasoning:
        prompt += (
            "For each candidate, provide a 2-3 sentence explanation of your rating, then the rating in this format:\n"
            "Candidate A:\n"
            "Reasoning: [2-3 sentences explaining your rating]\n"
            "Rating: [Number]\n\n"
            "Candidate B:\n"
            "Reasoning: [2-3 sentences explaining your rating]\n"
            "Rating: [Number]\n\n"
        )
    else:
        prompt += (
            "For each candidate, provide only the rating in this format:\n"
            "Candidate A:\n"
            "Rating: [Number]\n\n"
            "Candidate B:\n"
            "Rating: [Number]\n\n"
        )

    prompt += f"**Candidate A:**\n{letter_text1}\n\n"
    prompt += f"**Candidate B:**\n{letter_text2}"

    if verbose:
        print("\n" + "=" * 80 + f"\nPROMPT:\n{prompt}\n" + "=" * 80 + "\n")

    try:
        rating_response = experiment.generate_response(prompt)

        if verbose and rating_response:
            print("\nMODEL RESPONSE:\n" + rating_response + "\n" + "=" * 80 + "\n")

        if rating_response is None:
            print("ERROR: Received None response from model")
            return (None, None), (None, None)

        results = extract_batch_responses(rating_response, include_reasoning, verbose)
        if not results:
            print("ERROR: Could not extract ratings and explanations")
            return (None, None), (None, None)

        rating1, explanation1, rating2, explanation2 = results[0]
        return (rating1, rating2), (explanation1, explanation2)

    except Exception as e:
        print(f"ERROR: Error during processing: {e}")
        return (None, None), (None, None)


def extract_single_responses(response_text: str, include_reasoning: bool = True,
                             verbose: bool = False) -> List[Tuple[Optional[str], Optional[str]]]:
    try:
        # Try simple format first (Letter N: X)
        simple_matches = re.finditer(r'Letter (\d+): (\d+)', response_text)
        ratings = [(match.group(2), None) for match in simple_matches]

        if ratings:
            if verbose:
                print(f"Found {len(ratings)} ratings in simple format")
            return ratings

        # Try formal format
        formal_matches = re.finditer(r'Letter \d+:(.*?)(?=Letter \d+:|$)', response_text, re.DOTALL)
        responses = [match.group(1).strip() for match in formal_matches]

        if verbose:
            print(f"Found {len(responses)} responses in formal format")

        results = []
        for response in responses:
            rating, explanation = extract_rating_explanation(response, include_reasoning, verbose)
            if rating:
                results.append((rating, explanation))

        return results

    except Exception as e:
        print(f"ERROR: Could not extract single responses: {e}")
        return []


def extract_paired_responses(response_text: str, include_reasoning: bool = True,
                             verbose: bool = False) -> List[Tuple[str, Optional[str], str, Optional[str]]]:
    try:
        # Find the first pair marker
        first_pair_match = re.search(r'Pair \d+:', response_text)
        if first_pair_match:
            response_text = response_text[first_pair_match.start():]

        pairs = re.split(r'(?:###\s*)?Pair \d+:', response_text)
        pairs = [p.strip() for p in pairs if p.strip()]

        results = []
        for pair in pairs:
            pair = re.sub(r'\*\*', '', pair)

            # Try original formal format first
            rating_a_match = re.search(r'Candidate A:.*?Rating:\s*(\d+)', pair, re.DOTALL)
            rating_b_match = re.search(r'Candidate B:.*?Rating:\s*(\d+)', pair, re.DOTALL)

            if rating_a_match and rating_b_match:
                rating1 = rating_a_match.group(1)
                rating2 = rating_b_match.group(1)

                explanation1 = None
                explanation2 = None

                if include_reasoning:
                    reasoning_a = re.search(r'Candidate A:.*?Reasoning:\s*(.*?)\s*Rating:', pair, re.DOTALL)
                    reasoning_b = re.search(r'Candidate B:.*?Reasoning:\s*(.*?)\s*Rating:', pair, re.DOTALL)

                    if reasoning_a:
                        explanation1 = reasoning_a.group(1).strip()
                    if reasoning_b:
                        explanation2 = reasoning_b.group(1).strip()

                results.append((rating1, explanation1, rating2, explanation2))
                continue

            # Try simple format: "Candidate A: N\nCandidate B: N"
            simple_match = re.search(r'Candidate A:\s*(\d+)\s*Candidate B:\s*(\d+)', pair.replace('\n', ' '))
            if simple_match:
                rating1, rating2 = simple_match.groups()
                results.append((rating1, None, rating2, None))
                continue

            # Try numbers-only format
            numbers_only = re.findall(r'(?:^|\n)\s*(\d+)\s*(?:\n|$)', pair)
            if len(numbers_only) == 2:
                rating1, rating2 = numbers_only
                results.append((rating1, None, rating2, None))
                continue

        return results

    except Exception as e:
        print(f"ERROR: Could not extract paired responses: {e}")
        return []


def process_batch(
        experiment: Any,
        context: str,
        letters_batch: List[str],
        include_reasoning: bool,
        is_paired: bool = False,
        verbose: bool = False
) -> List[Tuple[Optional[str], Optional[str]]]:
    instructions_text = (
        f"{context}Rate the following cover letter(s) on a 5-point scale:\n"
        "1. Not likely at all\n"
        "2. Somewhat unlikely\n"
        "3. Neutral\n"
        "4. Somewhat likely\n"
        "5. Very likely\n\n"
    )

    if include_reasoning:
        instructions_text += (
            "For each entry, provide a 2-3 sentence explanation of your rating, then the rating in this format:\n"
            "Letter N:\n"
            "Reasoning: [2-3 sentences explaining your rating]\n"
            "Rating: [Number]\n\n"
        )
    else:
        instructions_text += (
            "For each entry, provide only the rating in this format:\n"
            "Letter N:\n"
            "Rating: [Number]\n\n"
        )

    prompt = instructions_text
    if is_paired:
        pair_count = len(letters_batch) // 2
        candidate_template = (
            "{}:\n"
            f"{'Reasoning: [2-3 sentences explaining your rating]' if include_reasoning else ''}\n"
            "Rating: [Number]\n\n"
        )

        for pair_idx in range(pair_count):
            prompt += f"Pair {pair_idx + 1}:\n"
            prompt += candidate_template.format("Candidate A")
            prompt += candidate_template.format("Candidate B")

            letter_text1 = letters_batch[pair_idx * 2]
            letter_text2 = letters_batch[pair_idx * 2 + 1]
            prompt += f"**Candidate A:**\n{letter_text1}\n\n"
            prompt += f"**Candidate B:**\n{letter_text2}\n\n"
    else:
        for idx, letter_text in enumerate(letters_batch, 1):
            prompt += f"Letter {idx}:\n"
            if include_reasoning:
                prompt += "Reasoning: [2-3 sentences explaining your rating]\n"
            prompt += "Rating: [Number]\n\n"
            prompt += f"**Candidate:**\n{letter_text}\n\n"

    if verbose:
        print("\nGENERATED PROMPT:\n" + "-" * 40 + f"\n{prompt}\n" + "=" * 80)

    try:
        batch_response = experiment.generate_response(prompt)

        if verbose and batch_response:
            print("\nMODEL RESPONSE:\n" + "-" * 40 + f"\n{batch_response}\n" + "=" * 80)

        if batch_response is None:
            print("ERROR: Received None response from model")
            return [None] * len(letters_batch)

        if is_paired:
            pairs = extract_paired_responses(batch_response, include_reasoning, verbose)
            results = []
            for rating1, expl1, rating2, expl2 in pairs:
                results.extend([(rating1, expl1), (rating2, expl2)])
        else:
            results = extract_single_responses(batch_response, include_reasoning, verbose)

        while len(results) < len(letters_batch):
            results.append((None, None))

        return results

    except Exception as e:
        print(f"ERROR: Error during batch processing: {e}")
        return [None] * len(letters_batch)


def main(
        model_map: Dict[str, List[str]],
        context: str,
        context_id: str,
        include_reasoning: bool,
        dimensions: List[str],
        n_rows: int,
        results_path: str,
        verbose: bool = False,
        batching_method: str = 'bulk',
        batch_size: int = 30
) -> None:
    filtered_dimensions = {dim: DIMENSION_MAP[dim] for dim in dimensions if dim in DIMENSION_MAP}
    data_path = f"../../{Parameters.data_folder}"
    if results_path:
        os.makedirs(results_path, exist_ok=True)
        print(f"Created results directory at: {results_path}")

    results_dict = {}

    first_dim = list(filtered_dimensions.keys())[0]
    file_path = os.path.join(data_path, f"{first_dim}.xlsx")
    try:
        df = pd.read_excel(file_path, nrows=n_rows)

        if 'Cover_Letter' in df.columns:
            print("\nProcessing single cover letters...")

            for model_type, model_names in model_map.items():
                for model_name in model_names:
                    print(f"\nProcessing single letters for {model_type} model: {model_name}")
                    experiment = get_experiment_instance(model_type, model_name)

                    if batching_method == 'bulk':
                        letters_batch = []
                        batch_indices = []

                        for idx, row in df.iterrows():
                            single_letter = row.get('Cover_Letter')
                            if not pd.isnull(single_letter):
                                letters_batch.append(single_letter)
                                batch_indices.append(idx)

                                if len(letters_batch) >= batch_size:
                                    print(f"\nProcessing batch of {len(letters_batch)} letters")
                                    batch_results = process_batch(
                                        experiment=experiment,
                                        context=context,
                                        letters_batch=letters_batch,
                                        include_reasoning=include_reasoning,
                                        is_paired=False,
                                        verbose=verbose
                                    )

                                    for b_idx, (rating, explanation) in zip(batch_indices, batch_results):
                                        key = (model_name, b_idx)
                                        if key not in results_dict:
                                            results_dict[key] = {
                                                'model_name': model_name,
                                                'likert_scale': 5,
                                                'question_num': b_idx + 1
                                            }
                                        results_dict[key].update({
                                            'Cover_Letter_Rating': rating,
                                            'Cover_Letter_Explanation': explanation
                                        })

                                    letters_batch = []
                                    batch_indices = []

                        if letters_batch:
                            print(f"\nProcessing remaining {len(letters_batch)} letters")
                            batch_results = process_batch(
                                experiment=experiment,
                                context=context,
                                letters_batch=letters_batch,
                                include_reasoning=include_reasoning,
                                is_paired=False,
                                verbose=verbose
                            )

                            for b_idx, (rating, explanation) in zip(batch_indices, batch_results):
                                key = (model_name, b_idx)
                                if key not in results_dict:
                                    results_dict[key] = {
                                        'model_name': model_name,
                                        'likert_scale': 5,
                                        'question_num': b_idx + 1
                                    }
                                results_dict[key].update({
                                    'Cover_Letter_Rating': rating,
                                    'Cover_Letter_Explanation': explanation
                                })
                    else:  # incremental processing
                        for idx, row in df.iterrows():
                            key = (model_name, idx)
                            if key not in results_dict:
                                results_dict[key] = {
                                    'model_name': model_name,
                                    'likert_scale': 5,
                                    'question_num': idx + 1
                                }

                            single_letter = row.get('Cover_Letter')
                            if not pd.isnull(single_letter):
                                print(f"\nProcessing single letter for row {idx + 1}")
                                rating, explanation = rate_single_cover_letter(
                                    experiment=experiment,
                                    context=context,
                                    letter_text=single_letter,
                                    include_reasoning=include_reasoning,
                                    verbose=verbose
                                )

                                results_dict[key].update({
                                    'Cover_Letter_Rating': rating,
                                    'Cover_Letter_Explanation': explanation
                                })
    except Exception as e:
        print(f"ERROR: Error processing single cover letters: {e}")

    for dimension, (pole1, pole2) in filtered_dimensions.items():
        file_path = os.path.join(data_path, f"{dimension}.xlsx")
        try:
            df = pd.read_excel(file_path, nrows=n_rows)
            print(f"Successfully read '{file_path}' for dimension '{dimension}'.")

            for model_type, model_names in model_map.items():
                for model_name in model_names:
                    print(f"\nProcessing {model_type} model: {model_name} for dimension {dimension}")
                    experiment = get_experiment_instance(model_type, model_name)

                    if batching_method == 'bulk':
                        letters_batch = []
                        batch_indices = []

                        for idx, row in df.iterrows():
                            letter_text1 = row.get(f"{pole1}_Cover_Letter", "No text provided.")
                            letter_text2 = row.get(f"{pole2}_Cover_Letter", "No text provided.")

                            if not (pd.isnull(letter_text1) and pd.isnull(letter_text2)):
                                letter_text1 = letter_text1 if not pd.isnull(letter_text1) else "No text provided."
                                letter_text2 = letter_text2 if not pd.isnull(letter_text2) else "No text provided."

                                letters_batch.extend([letter_text1, letter_text2])
                                batch_indices.append(idx)

                                if len(letters_batch) >= batch_size * 2:  # Multiply by 2 for pairs
                                    print(f"\nProcessing batch of {len(batch_indices)} pairs")
                                    batch_results = process_batch(
                                        experiment=experiment,
                                        context=context,
                                        letters_batch=letters_batch,
                                        include_reasoning=include_reasoning,
                                        is_paired=True,
                                        verbose=verbose
                                    )

                                    for b_idx, i in enumerate(batch_indices):
                                        key = (model_name, i)
                                        if key not in results_dict:
                                            results_dict[key] = {
                                                'model_name': model_name,
                                                'likert_scale': 5,
                                                'question_num': i + 1
                                            }

                                        rating1, expl1 = batch_results[b_idx * 2]
                                        rating2, expl2 = batch_results[b_idx * 2 + 1]

                                        results_dict[key].update({
                                            f'{pole1}_Cover_Letter_Rating': rating1,
                                            f'{pole1}_Cover_Letter_Explanation': expl1,
                                            f'{pole2}_Cover_Letter_Rating': rating2,
                                            f'{pole2}_Cover_Letter_Explanation': expl2
                                        })

                                    letters_batch = []
                                    batch_indices = []

                        if letters_batch:
                            print(f"\nProcessing remaining {len(batch_indices)} pairs")
                            batch_results = process_batch(
                                experiment=experiment,
                                context=context,
                                letters_batch=letters_batch,
                                include_reasoning=include_reasoning,
                                is_paired=True,
                                verbose=verbose
                            )

                            for b_idx, i in enumerate(batch_indices):
                                key = (model_name, i)
                                if key not in results_dict:
                                    results_dict[key] = {
                                        'model_name': model_name,
                                        'likert_scale': 5,
                                        'question_num': i + 1
                                    }

                                rating1, expl1 = batch_results[b_idx * 2]
                                rating2, expl2 = batch_results[b_idx * 2 + 1]

                                results_dict[key].update({
                                    f'{pole1}_Cover_Letter_Rating': rating1,
                                    f'{pole1}_Cover_Letter_Explanation': expl1,
                                    f'{pole2}_Cover_Letter_Rating': rating2,
                                    f'{pole2}_Cover_Letter_Explanation': expl2
                                })
                    else:  # incremental processing
                        for idx, row in df.iterrows():
                            key = (model_name, idx)
                            if key not in results_dict:
                                results_dict[key] = {
                                    'model_name': model_name,
                                    'likert_scale': 5,
                                    'question_num': idx + 1
                                }

                            letter_text1 = row.get(f"{pole1}_Cover_Letter", "No text provided.")
                            letter_text2 = row.get(f"{pole2}_Cover_Letter", "No text provided.")

                            if not (pd.isnull(letter_text1) and pd.isnull(letter_text2)):
                                letter_text1 = letter_text1 if not pd.isnull(letter_text1) else "No text provided."
                                letter_text2 = letter_text2 if not pd.isnull(letter_text2) else "No text provided."

                                print(f"\nProcessing paired letters for dimension: {dimension}, row {idx + 1}")
                                (rating1, rating2), (explanation1, explanation2) = rate_cover_letters(
                                    experiment=experiment,
                                    context=context,
                                    letter_text1=letter_text1,
                                    letter_text2=letter_text2,
                                    include_reasoning=include_reasoning,
                                    verbose=verbose
                                )

                                results_dict[key].update({
                                    f'{pole1}_Cover_Letter_Rating': rating1,
                                    f'{pole1}_Cover_Letter_Explanation': explanation1,
                                    f'{pole2}_Cover_Letter_Rating': rating2,
                                    f'{pole2}_Cover_Letter_Explanation': explanation2
                                })
        except Exception as e:
            print(f"ERROR: Error processing dimension '{dimension}': {e}")
            continue

    if results_dict:
        results_df = pd.DataFrame(list(results_dict.values()))
        reasoning_suffix = "reasoning" if include_reasoning else "non_reasoning"
        output_file_path = os.path.join(
            results_path,
            f'context_{context_id.lower().replace(" ", "_")}_'
            f'likert_scale_5_{reasoning_suffix}_{batching_method}.csv'
        )

        try:
            results_df.to_csv(output_file_path, index=False)
            print(f"\nSuccessfully saved all results to '{output_file_path}'")
        except Exception as e:
            print(f"ERROR: Error saving the final CSV file: {e}")
    else:
        print("WARNING: No results to save.")


if __name__ == '__main__':
    dimensions = Parameters.dimensions
    n_rows = 30
    results_path = "results"
    context, context_id = Parameters.original_context
    include_reasoning = False
    batching_method = Parameters.batching_method
    batch_size = 30 if include_reasoning else n_rows

    main(
        model_map=MODEL_MAP,
        context=context,
        context_id=context_id,
        include_reasoning=include_reasoning,
        dimensions=dimensions,
        n_rows=n_rows,
        results_path=results_path,
        verbose=True,
        batching_method=batching_method,
        batch_size=batch_size
    )
