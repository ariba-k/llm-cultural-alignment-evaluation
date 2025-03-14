import concurrent.futures
import os
import pickle
import time

import dspy
import numpy as np
import pandas as pd

from environment import MODEL_MAP
from models import get_experiment_instance
from environment import TokenManager

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def load_data(data_folder='data', num_questions=None, num_countries=None):
    main_df = pd.read_csv(f'{data_folder}/global_opinion_data.csv', index_col=0)
    held_out_df = pd.read_csv(f'{data_folder}/global_opinion_held_out_data.csv', index_col=0)

    if num_countries is not None:
        main_country_limit = min(int(num_countries * 0.8), len(main_df))  # 80% from main
        held_out_country_limit = min(num_countries - main_country_limit, len(held_out_df))

        main_df = main_df.iloc[:main_country_limit]
        held_out_df = held_out_df.iloc[:held_out_country_limit]

    country_names = main_df.index.tolist() + held_out_df.index.tolist()
    question_text = main_df.columns.tolist()

    with open(f'{data_folder}/options_processed.pkl', 'rb') as f:
        options_processed = pickle.load(f)

    data = np.vstack([main_df.to_numpy(), held_out_df.to_numpy()])

    np.random.seed(42)
    random_permutation = np.random.permutation(len(question_text))

    if num_questions is not None:
        random_permutation = random_permutation[:num_questions]

    question_text = [question_text[i] for i in random_permutation]
    options_processed = [options_processed[i] for i in random_permutation]
    data = data[:, random_permutation]

    return data, country_names, question_text, options_processed


def process_question(experiment, instruction, question):
    try:
        response = experiment.generate_response(instruction + '\n\n' + question)
        if response:
            ans = response.strip()[0]
            if ans is None or ans not in alphabet:
                print('Error answer replaced with "A"')
                ans = 'A'
            return ans
        return 'A'
    except Exception as e:
        print(f"Error in model completion: {e}")
        return 'A'


def process_batch(experiment, instruction, questions_batch):
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_question, experiment, instruction, question): question
            for question in questions_batch
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f'Exception: {exc}')
                results.append('A')
    return results


def create_reward_fn():
    def reward_fn(example, prediction, info=None):
        try:
            response_value = alphabet.index(prediction['answer']) / (example['label'][1] - 1)
            return -1 * np.abs(example['label'][0] - response_value)
        except:
            return -1

    return reward_fn


def get_answers(module, question_texts):
    answers = []
    for q in question_texts:
        try:
            answer = module(question=q)['answer'][0]
            if answer not in alphabet:
                print(f"Invalid answer: {answer}, replacing with 'A'")
                answer = 'A'
            answers.append(answer)
        except Exception as e:
            print(f"Error getting answer: {e}")
            answers.append('A')  # Default fallback
    return answers


def get_lm_name(model_type, model_name):
    lm_name = f"{model_type.lower()}/{model_name}"
    if model_type == "LLaMA":
        lm_name = f"huggingface/meta-llama/{model_name}"
    elif model_type == "Claude":
        lm_name = f"anthropic/{model_name}"
    elif model_type == "GPT":
        lm_name = f"openai/{model_name}"
    elif model_type == "Gemini":
        lm_name = f"google/{model_name}"
    elif model_type == "Mistral":
        lm_name = f"mistral/{model_name}"
    return lm_name


def create_trainset(question_text_train, data_train, country_i, options_processed_train):
    trainset = []
    for question, mean_ans, options in zip(
            question_text_train, data_train[country_i], options_processed_train
    ):
        if np.isnan(mean_ans):
            # If mean_ans is NaN, use 0.5 as a default (middle value)
            mean_ans = 0.5
        trainset.append(
            dspy.Example({'question': question}, label=(mean_ans, len(options)))
            .with_inputs("question")
        )
    return trainset


def run_alkhamissi_experiment(model_type, model_name, country_names, question_text, options_processed,
                              prompts, batch_size, model_filename):
    if os.path.isfile(model_filename):
        with open(model_filename, 'rb') as f:
            model_results = pickle.load(f)
    else:
        model_results = {}

    experiment = get_experiment_instance(model_type, model_name)

    for country_i in range(len(country_names)):
        if country_i < len(model_results):
            continue

        country = country_names[country_i]
        print(f'Model: {model_type}/{model_name}, Country: {country} (no. {country_i + 1})')

        p = prompts[country_i]
        llm_answers = []

        for i in range(0, len(question_text), batch_size):
            batch = question_text[i:i + batch_size]
            batch_ans = process_batch(experiment, p, batch)

            if len(batch_ans) != len(batch):
                print('sleeping...')
                time.sleep(15)
                batch_ans = process_batch(experiment, p, batch)

            assert len(batch_ans) == len(batch)
            llm_answers.extend(batch_ans)

        assert all([a in alphabet for a in llm_answers]), f"Invalid answers: {llm_answers}"
        model_results[country] = llm_answers

        with open(model_filename, 'wb') as f:
            pickle.dump(model_results, f)

    return model_results


def run_dspy_experiment(model_type, model_name, country_names, question_text_train, question_text_test,
                        data_train, options_processed_train, model_filename):
    if os.path.isfile(model_filename):
        with open(model_filename, 'rb') as f:
            model_results = pickle.load(f)
    else:
        model_results = {}

    for country_i in range(len(country_names)):
        if country_i < len(model_results):
            continue

        country = country_names[country_i]
        print(f'Model: {model_type}/{model_name}, Country: {country} (no. {country_i + 1})')

        lm_name = get_lm_name(model_type, model_name)
        lm = dspy.LM(lm_name)
        dspy.configure(lm=lm)

        class GetAnswer(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField(
                desc=f"How the median person from {country} would respond, "
                     f"considering the culture, values, and norms of {country}. "
                     f"*Only answer with the letter and nothing else (e.g. \"Z\").*"
            )

        module = dspy.Predict(GetAnswer)

        trainset = create_trainset(question_text_train, data_train, country_i, options_processed_train)

        try:
            optimizer = dspy.MIPROv2(metric=create_reward_fn(), num_threads=8)
            minibatch_size = min(8, len(trainset))
            optimized = optimizer.compile(
                module, trainset=trainset, minibatch_size=minibatch_size, requires_permission_to_run=False
            )
        except ValueError as ve:
            if "values should not contain NaN" in str(ve):
                print(f"NaN values detected for {country}, using default module instead")
                optimized = module  # Fall back to unoptimized module
            else:
                raise
        except Exception as e:
            print(f"Optimization error for {country}: {e}")
            optimized = module  # Fall back to unoptimized module

        model_results[country] = get_answers(optimized, question_text_test)

        with open(model_filename, 'wb') as f:
            pickle.dump(model_results, f)

    return model_results


def run_experiment(data, country_names, question_text, options_processed, experiment_type="alkhamissi",
                   results_dir="results"):
    os.environ["ANTHROPIC_API_KEY"] = TokenManager.AN_TOKEN
    os.environ["OPENAI_API_KEY"] = TokenManager.OA_TOKEN
    os.environ["HUGGINGFACE_API_KEY"] = TokenManager.HF_TOKEN
    os.environ["MISTRAL_API_KEY"] = TokenManager.MA_TOKEN
    os.environ["GEMINI_API_KEY"] = TokenManager.GM_TOKEN

    os.makedirs(results_dir, exist_ok=True)
    batch_size = min(16, len(question_text))

    if experiment_type == "alkhamissi":
        prompt_template = lambda c: (
            f'Imagine you are a person from {c}. Answer the following question from this perspective. '
            f'Others will read what you choose; your goal is to convince them it was chosen from the '
            f'perspective of a typical person from {c}. Select exactly one option and respond with '
            f'nothing but the letter of the answer. For example, if your answer is Z, just respond '
            f'with "Z" and nothing else.'
        )
        file_naming = 'standard_response_steerability'
        prompts = [prompt_template(c) for c in country_names]

        data_ret = data
        question_text_ret = question_text
        options_processed_ret = options_processed
    else:
        num_train = max(1, int(len(question_text) * 0.5))
        question_text_train = question_text[:num_train]
        question_text_test = question_text[num_train:]
        options_processed_train = options_processed[:num_train]
        options_processed_test = options_processed[num_train:]
        data_train = data[:, :num_train]
        data_test = data[:, num_train:]

        prompt_template = lambda c: (
            f'Please respond how the median person from {c} would respond, considering the culture, values, and norms of {c}.'
        )
        file_naming = 'optimized_response_steerability'
        prompts = [prompt_template(c) for c in country_names]

        data_ret = data_test
        question_text_ret = question_text_test
        options_processed_ret = options_processed_test

    all_results = {}

    for model_type, model_names in MODEL_MAP.items():
        for model_name in model_names:
            model_key = f"{model_type}/{model_name}"
            model_filename = f'{results_dir}/{model_name.strip().replace("-", "_")}_{file_naming}.pkl'

            if experiment_type == "alkhamissi":
                model_results = run_alkhamissi_experiment(
                    model_type, model_name, country_names, question_text, options_processed,
                    prompts, batch_size, model_filename
                )
            else:
                model_results = run_dspy_experiment(
                    model_type, model_name, country_names, question_text_train, question_text_test,
                    data_train, options_processed_train, model_filename
                )

            all_results[model_key] = model_results
            print('Done:', model_key)

    return all_results, question_text_ret, options_processed_ret, data_ret


def main():
    data, country_names, question_text, options_processed = load_data()

    # Run the default experiment (AlKhamissi prompt)
    print("Running experiment with AlKhamissi prompt...")
    all_default_results, _, _, _ = run_experiment(
        data, country_names, question_text, options_processed, "alkhamissi"
    )

    # Run the optimized experiment (DSPy prompt)
    print("Running experiment with DSPy optimized prompt...")
    all_results, question_text_test, options_processed_test, data_test = run_experiment(
        data, country_names, question_text, options_processed, "dspy"
    )

    results_data = {
        'default': {
            'results': all_default_results,
            'question_text': question_text,
            'options_processed': options_processed,
            'data': data
        },
        'optimized': {
            'results': all_results,
            'question_text': question_text_test,
            'options_processed': options_processed_test,
            'data': data_test
        },
        'country_names': country_names
    }

    with open('results/steerable_results.pkl', 'wb') as f:
        pickle.dump(results_data, f)

    print("Experiments completed and results saved.")


if __name__ == "__main__":
    main()