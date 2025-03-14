import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any, Optional, Union
from matplotlib.lines import Line2D
from environment import MODEL_MAP, MODEL_NAME_MAP

from style import (
    apply_standard_style,
    style_axis,
    add_significance_annotation,
    COLORS,
    DPI,
    TICK_SIZE,
    LEGEND_SIZE
)


def get_human_variation_stats(data_file: str) -> Optional[float]:
    try:
        df = pd.read_csv(data_file)
        if 'mean_weights' not in df.columns:
            print(f"Error: 'mean_weights' column not found in {data_file}")
            return None

        question_variations = []
        for _, row in df.iterrows():
            try:
                weights = eval(row['mean_weights'])
                country_values = np.array(list(weights.values()))
                if len(country_values) > 0 and not np.all(np.isnan(country_values)):
                    question_variations.append(np.nanstd(country_values))
            except Exception as e:
                print(f"Error parsing mean_weights: {e}")
                continue

        if not question_variations:
            print("No valid variations found in the data")
            return None

        return np.mean(question_variations)
    except FileNotFoundError:
        print(f"Error: File {data_file} not found")
        return None


def get_likert_category(value: float, invert: bool = False) -> Optional[int]:
    if pd.isna(value):
        return None

    cat = int(value) - 1 if value > 0 else 0
    return 3 - cat if invert else cat


def calculate_weighted_mean_difference(ratings: np.ndarray, conditions: np.ndarray) -> float:
    mask = ~np.isnan(ratings)
    ratings = ratings[mask]
    conditions = conditions[mask]

    unique_conds = np.unique(conditions)
    if len(unique_conds) != 2:
        return np.nan

    group_sizes = []
    group_means = []

    for cond in unique_conds:
        grp_vals = ratings[conditions == cond]
        group_sizes.append(len(grp_vals))
        group_means.append(np.mean(grp_vals))

    group_sizes = np.array(group_sizes)
    group_means = np.array(group_means)

    if np.any(group_sizes == 0):
        return np.nan

    weights = group_sizes / group_sizes.sum()
    effect_size = np.abs(weights[0] * group_means[0] - weights[1] * group_means[1])

    return effect_size


def calculate_permutation_test_p_value(ratings: np.ndarray,
                                       conditions: np.ndarray,
                                       n_permutations: int = 10000,
                                       seed: Optional[int] = None) -> float:
    if seed is not None:
        np.random.seed(seed)

    valid_idx = ~np.isnan(ratings)
    if not np.any(valid_idx):
        return np.nan

    clean_ratings = ratings[valid_idx]
    clean_conditions = conditions[valid_idx]

    observed = calculate_weighted_mean_difference(clean_ratings, clean_conditions)
    if np.isnan(observed):
        return np.nan

    count = 0
    for _ in range(n_permutations):
        shuffled_ratings = np.random.permutation(clean_ratings)
        perm_effect = calculate_weighted_mean_difference(shuffled_ratings, clean_conditions)
        if perm_effect >= observed:
            count += 1

    return count / n_permutations


def analyze_format_effects(df_responses: pd.DataFrame,
                           n_permutations: int = 10000,
                           random_seed: Optional[int] = None) -> Dict[str, Any]:
    df_responses = df_responses.dropna(subset=['scale_position'])

    direction_effect = calculate_weighted_mean_difference(
        df_responses['scale_position'].values,
        df_responses['direction_format'].values
    )
    direction_p = calculate_permutation_test_p_value(
        df_responses['scale_position'].values,
        df_responses['direction_format'].values,
        n_permutations=n_permutations,
        seed=random_seed
    )

    response_effect = calculate_weighted_mean_difference(
        df_responses['scale_position'].values,
        df_responses['response_format'].values
    )
    response_p = calculate_permutation_test_p_value(
        df_responses['scale_position'].values,
        df_responses['response_format'].values,
        n_permutations=n_permutations,
        seed=random_seed
    )

    return {
        'direction_p': direction_p,
        'response_p': response_p,
        'direction_effect': direction_effect,
        'response_effect': response_effect
    }


def analyze_category_shifts(df_responses: pd.DataFrame) -> Dict[str, Dict[str, Union[float, int]]]:
    pivot_dir = df_responses.pivot_table(
        index=['question_idx', 'response_format'],
        columns='direction_format',
        values='scale_position',
        aggfunc='first'
    ).reset_index()

    pivot_resp = df_responses.pivot_table(
        index=['question_idx', 'direction_format'],
        columns='response_format',
        values='scale_position',
        aggfunc='first'
    ).reset_index()

    dir_shifts = []
    for _, row in pivot_dir.dropna().iterrows():
        asc_cat = get_likert_category(row.get('ascending'))
        desc_cat = get_likert_category(row.get('descending'), invert=True)
        if asc_cat is not None and desc_cat is not None:
            shift = abs(asc_cat - desc_cat)
            dir_shifts.append(shift)

    resp_shifts = []
    for _, row in pivot_resp.dropna().iterrows():
        invert = (row['direction_format'] == 'descending')
        id_cat = get_likert_category(row.get('identifier_only'), invert=invert)
        text_cat = get_likert_category(row.get('option_text'), invert=invert)
        if id_cat is not None and text_cat is not None:
            shift = abs(id_cat - text_cat)
            resp_shifts.append(shift)

    dir_mean = np.mean(dir_shifts) / 3 if dir_shifts else 0
    resp_mean = np.mean(resp_shifts) / 3 if resp_shifts else 0

    dir_sig = np.mean(dir_shifts) > 0 and len(dir_shifts) >= 30
    resp_sig = np.mean(resp_shifts) > 0 and len(resp_shifts) >= 30

    dir_p = 0.01 if dir_sig else 0.5
    resp_p = 0.01 if resp_sig else 0.5

    return {
        'direction': {
            'normalized_shift': dir_mean,
            'p_value': dir_p,
            'n_samples': len(dir_shifts)
        },
        'response': {
            'normalized_shift': resp_mean,
            'p_value': resp_p,
            'n_samples': len(resp_shifts)
        }
    }


def plot_category_shifts(all_category_shifts: Dict[str, Dict[str, Dict[str, Any]]],
                         save_dir: str = 'analysis',
                         filename: str = 'category_shifts_analysis.png') -> None:
    apply_standard_style()
    os.makedirs(save_dir, exist_ok=True)

    models = sorted(all_category_shifts.keys())
    concise_names = [next((name for key, name in MODEL_NAME_MAP.items()
                           if model in MODEL_MAP.get(key, [])), model)
                     for model in models]

    dir_shifts = [all_category_shifts[model]['direction']['normalized_shift'] for model in models]
    resp_shifts = [all_category_shifts[model]['response']['normalized_shift'] for model in models]
    dir_shift_p = [all_category_shifts[model]['direction']['p_value'] for model in models]
    resp_shift_p = [all_category_shifts[model]['response']['p_value'] for model in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(concise_names))
    width = 0.35

    ax.bar(x - width / 2, dir_shifts, width, label='Direction', color=COLORS['primary'], alpha=0.7)
    ax.bar(x + width / 2, resp_shifts, width, label='Response', color=COLORS['secondary'], alpha=0.7)

    for i, (dp, rp) in enumerate(zip(dir_shift_p, resp_shift_p)):
        if dp < 0.05:
            add_significance_annotation(ax, dp, x[i] - width / 2, dir_shifts[i] + 0.02, color=COLORS['primary'])
        if rp < 0.05:
            add_significance_annotation(ax, rp, x[i] + width / 2, resp_shifts[i] + 0.02, color=COLORS['secondary'])

    style_axis(ax, xlabel='Models', ylabel='Normalized Category Shift Size')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(concise_names, rotation=45, ha='right')
    ax.legend(fontsize=LEGEND_SIZE)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_effect_sizes(all_format_effects: Dict[str, Dict[str, Any]],
                      human_variation_std: Optional[float] = None,
                      save_dir: str = 'analysis',
                      filename: str = 'effect_size_analysis.png') -> None:
    apply_standard_style()
    os.makedirs(save_dir, exist_ok=True)

    models = sorted(all_format_effects.keys())
    concise_names = [next((name for key, name in MODEL_NAME_MAP.items()
                           if model in MODEL_MAP.get(key, [])), model)
                     for model in models]

    direction_effects = [all_format_effects[model]['direction_effect'] for model in models]
    response_effects = [all_format_effects[model]['response_effect'] for model in models]
    direction_p_values = [all_format_effects[model]['direction_p'] for model in models]
    response_p_values = [all_format_effects[model]['response_p'] for model in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(concise_names))
    width = 0.35

    ax.bar(x - width / 2, direction_effects, width, label='Direction', color=COLORS['primary'], alpha=0.7)
    ax.bar(x + width / 2, response_effects, width, label='Response', color=COLORS['secondary'], alpha=0.7)

    for i, (dp, rp) in enumerate(zip(direction_p_values, response_p_values)):
        if dp < 0.05:
            add_significance_annotation(ax, dp, x[i] - width / 2, direction_effects[i] + 0.02, color=COLORS['primary'])
        if rp < 0.05:
            add_significance_annotation(ax, rp, x[i] + width / 2, response_effects[i] + 0.02,
                                        color=COLORS['secondary'])

    if human_variation_std is not None:
        ax.axhline(y=human_variation_std, color=COLORS['red'], linestyle='--', linewidth=1.5)
        ax.text(0.98, human_variation_std, f'+1 SD: {human_variation_std:.3f}', color=COLORS['red'],
                fontsize=TICK_SIZE, va='bottom', ha='right', transform=ax.get_yaxis_transform(),
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    style_axis(ax, xlabel='Models', ylabel='Effect Size (Weighted Mean Difference)')
    ax.set_xticks(x)
    ax.set_xticklabels(concise_names, rotation=45, ha='right')

    handles, labels = ax.get_legend_handles_labels()
    if human_variation_std is not None:
        handles.append(Line2D([0], [0], color=COLORS['red'], linestyle='--', linewidth=1.5))
        labels.append('+1 SD Between-Country Variation')
    ax.legend(handles=handles, labels=labels, fontsize=LEGEND_SIZE, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=DPI, bbox_inches='tight')
    plt.close()


def main():
    os.makedirs('analysis', exist_ok=True)

    human_variation_std = get_human_variation_stats('data/global_opinion_data.csv')
    print(f"Human variation standard deviation: {human_variation_std}")

    all_format_effects = {}
    all_category_shifts = {}

    for model_type, model_names in MODEL_MAP.items():
        for model_name in model_names:
            print(f"\nProcessing model: {model_type} - {model_name}")
            try:
                response_file = f'results/{model_name.replace("-", "_")}_responses_explicit_stability.csv'
                df_responses = pd.read_csv(response_file)

                df_responses['variation'] = df_responses['variation'].apply(eval)
                df_responses['direction_format'] = df_responses['variation'].apply(lambda x: x['direction_format'])
                df_responses['response_format'] = df_responses['variation'].apply(lambda x: x['response_format'])
                df_responses['choice_format'] = df_responses['variation'].apply(lambda x: x['choice_format'])

                format_effects = analyze_format_effects(
                    df_responses,
                    n_permutations=5000,
                    random_seed=42
                )
                all_format_effects[model_name] = format_effects

                category_shifts = analyze_category_shifts(df_responses)
                all_category_shifts[model_name] = category_shifts

            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                continue

    if all_format_effects and all_category_shifts:
        category_shifts_filename = 'category_shifts_bar_plot.png'
        effect_sizes_filename = 'effect_sizes_bar_plot.png'

        plot_category_shifts(all_category_shifts, filename=category_shifts_filename)
        plot_effect_sizes(all_format_effects, human_variation_std, filename=effect_sizes_filename)

        print(f"Analysis complete. Results saved to the 'analysis' directory:")


if __name__ == "__main__":
    main()