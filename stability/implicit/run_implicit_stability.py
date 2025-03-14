import os
from typing import List, Dict, Any

import pandas as pd

from environment import MODEL_MAP
from stability.implicit.constants_implicit_stability import Parameters
from stability.implicit.models_implicit_stability import get_experiment_instance


def run_experiment() -> None:
    num_trials = Parameters.num_trials
    num_rows = Parameters.num_rows
    temperature = Parameters.temperature
    buffer_tokens = Parameters.buffer_tokens
    batching_method = Parameters.batching_method
    verbose = Parameters.verbose
    use_reasoning = Parameters.use_reasoning
    likert_scales = Parameters.likert_scales
    batch_size = Parameters.batch_size
    contexts = Parameters.contexts
    dimensions = Parameters.dimensions
    results_dir = Parameters.results_folder
    model_map = MODEL_MAP

    contexts_dataset = pd.read_excel(Parameters.context_file)

    available_contexts = list(zip(contexts_dataset['Context'], contexts_dataset['Identifier']))
    print(f"Available contexts: {available_contexts}")
    final_contexts = [Parameters.original_context]
    if contexts:
        for context in available_contexts:
            context_text, context_id = context
            if context_id.strip() in contexts:
                final_contexts.append(context)

    contexts = final_contexts
    print(f"Selected contexts: {contexts}")

    if not model_map:
        raise ValueError("No models selected for the experiment based on the provided parameters.")

    for context, context_id in contexts:
        sanitized_context_id = context_id.strip().lower().replace(' ', '_')
        output_filename = f"context_{sanitized_context_id}.csv"
        reasoning_dir = "reasoning" if use_reasoning else "non_reasoning"
        results_path = os.path.join(results_dir, reasoning_dir, batching_method)

        os.makedirs(results_path, exist_ok=True)

        output_filepath = os.path.join(results_path, output_filename)
        print(output_filepath)

        if os.path.exists(output_filepath):
            print(f"Skipping Context '{context_id}': Results file '{output_filename}' already exists.")
            continue

        print(f"Processing Context '{context_id}'...")
        context_results: List[Dict[str, Any]] = []

        for dimension in dimensions:
            dataset_path = f"data/cover_letters/{dimension}.xlsx"

            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset file '{dataset_path}' does not exist. Skipping this dimension.")
                continue

            dataset = pd.read_excel(dataset_path)

            if num_rows is not None and num_rows < len(dataset):
                dataset = dataset.sample(n=num_rows, random_state=42)

            for likert_scale in likert_scales:
                for model_type, model_names in model_map.items():
                    for model_name in model_names:
                        experiment = get_experiment_instance(
                            model_name=model_name,
                            model_type=model_type,
                            temperature=temperature,
                            context=(context, context_id),
                            likert_scale=likert_scale,
                            batching_method=batching_method,
                            buffer_tokens=buffer_tokens,
                            verbose=verbose,
                            use_reasoning=use_reasoning,
                            batch_size=batch_size
                        )

                        experiment.generate_model_selections(dataset, num_trials, dimension)
                        if hasattr(experiment, 'detailed_results'):
                            for result in experiment.detailed_results:
                                context_results.append(result)
                        else:
                            print(f"Warning: Experiment instance for model '{model_name}' lacks 'detailed_results'.")

        if context_results:
            detailed_df = pd.DataFrame(context_results)
            try:
                detailed_df.to_csv(output_filepath, index=False)
                print(f"Context '{context_id}': Results saved to '{output_filename}'.")
            except IOError as e:
                print(f"Failed to save results for Context '{context_id}': {e}")
        else:
            print(f"No results collected for Context '{context_id}'. Skipping file saving.")


if __name__ == "__main__":
    run_experiment()
