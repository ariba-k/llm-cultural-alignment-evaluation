import os
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from stability.implicit.constants_implicit_stability import DIMENSION_NAME_MAP
from style import (apply_standard_style, style_axis, add_significance_annotation,
                   COLORS, DPI, TICK_SIZE, LABEL_SIZE, LEGEND_SIZE)


def load_data(reasoning_path: str, no_reasoning_path: str, likert_scale: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_reasoning = pd.read_csv(reasoning_path)
    df_no_reasoning = pd.read_csv(no_reasoning_path)

    df_reasoning = df_reasoning[df_reasoning['likert_scale'] == likert_scale]
    df_no_reasoning = df_no_reasoning[df_no_reasoning['likert_scale'] == likert_scale]

    return df_reasoning, df_no_reasoning


def calculate_reason_effect_size(ratings: np.ndarray, conditions: np.ndarray) -> float:
    mask = ~np.isnan(ratings)
    ratings = ratings[mask]
    conditions = conditions[mask]

    unique_conditions = np.unique(conditions)
    sizes = np.array([np.sum(conditions == c) for c in unique_conditions])
    means = np.array([np.mean(ratings[conditions == c]) for c in unique_conditions])

    weights = sizes / np.sum(sizes)
    effect_size = abs(weights[0] * means[0] - weights[1] * means[1])

    return effect_size


def calculate_reason_p_value(ratings: np.ndarray,
                             conditions: np.ndarray,
                             n_permutations: int = 10000,
                             seed: Optional[int] = None) -> float:
    if seed is not None:
        np.random.seed(seed)

    observed_effect = calculate_reason_effect_size(ratings, conditions)
    if np.isnan(observed_effect):
        return np.nan

    count = 0
    for _ in range(n_permutations):
        shuffled_ratings = np.random.permutation(ratings)
        perm_effect = calculate_reason_effect_size(shuffled_ratings, conditions)
        if perm_effect >= observed_effect:
            count += 1

    return count / n_permutations


def create_reason_statistical_analysis(df_reasoning: pd.DataFrame,
                                       df_no_reasoning: pd.DataFrame,
                                       stats_path: str) -> pd.DataFrame:
    dimensions = sorted(df_reasoning['dimension'].unique())
    results = []

    for dimension in dimensions:
        ratings_with = df_reasoning[df_reasoning['dimension'] == dimension]['rating'].values
        ratings_without = df_no_reasoning[df_no_reasoning['dimension'] == dimension]['rating'].values

        ratings = np.concatenate([ratings_with, ratings_without])
        conditions = np.concatenate([
            np.full(len(ratings_with), 'with_reasoning'),
            np.full(len(ratings_without), 'without_reasoning')
        ])

        effect_size = calculate_reason_effect_size(ratings, conditions)
        p_value = calculate_reason_p_value(ratings, conditions)

        results.append({
            'dimension': DIMENSION_NAME_MAP.get(dimension, dimension),
            'effect_size': round(effect_size, 3) if not np.isnan(effect_size) else np.nan,
            'p_value': round(p_value, 3) if not np.isnan(p_value) else np.nan
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(stats_path, index=False)

    return results_df


def create_reason_box_plot(df_reasoning: pd.DataFrame,
                                        df_no_reasoning: pd.DataFrame,
                                        results_df: pd.DataFrame,
                                        plot_path: str) -> None:
    apply_standard_style()

    df_reasoning_norm = df_reasoning.copy()
    df_no_reasoning_norm = df_no_reasoning.copy()
    df_reasoning_norm['rating'] = df_reasoning_norm['rating'] / 5
    df_no_reasoning_norm['rating'] = df_no_reasoning_norm['rating'] / 5

    dimensions = sorted(df_reasoning_norm['dimension'].unique())
    positions = np.arange(len(dimensions)) * 3

    data_by_condition = {
        'With Reasoning': [],
        'Without Reasoning': []
    }

    for dim in dimensions:
        reasoning_ratings = df_reasoning_norm[df_reasoning_norm['dimension'] == dim]['rating'].dropna()
        no_reasoning_ratings = df_no_reasoning_norm[df_no_reasoning_norm['dimension'] == dim]['rating'].dropna()
        data_by_condition['With Reasoning'].append(reasoning_ratings)
        data_by_condition['Without Reasoning'].append(no_reasoning_ratings)

    fig, ax = plt.subplots(figsize=(10, 4))
    style_axis(ax)

    box_width = 0.8
    box_props = {
        'With Reasoning': {'positions': positions - 0.5, 'color': COLORS['primary']},
        'Without Reasoning': {'positions': positions + 0.5, 'color': COLORS['secondary']}
    }

    boxplots = {}
    for condition, props in box_props.items():
        boxplots[condition] = ax.boxplot(
            data_by_condition[condition],
            positions=props['positions'],
            widths=box_width,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            meanprops={'color': 'black', 'linestyle': '-', 'linewidth': 1},
            medianprops={'visible': False},
            boxprops={'facecolor': props['color'], 'alpha': 0.7},
            flierprops={'marker': 'o', 'markerfacecolor': props['color'], 'markersize': 4, 'alpha': 0.7}
        )

    for i, dim in enumerate(dimensions):
        mapped_dim = DIMENSION_NAME_MAP.get(dim, dim)
        result = results_df[results_df['dimension'] == mapped_dim]
        if not result.empty and result.iloc[0]['p_value'] < 0.05:
            max_values = [
                data[i].max() if not data[i].empty else -np.inf
                for data in data_by_condition.values()
            ]
            max_y = max(max_values)
            annotation_y = min(max_y + 0.1, 1.15)
            add_significance_annotation(ax, result.iloc[0]['p_value'], positions[i], annotation_y)

    ax.set_xlabel('Dimension', fontsize=LABEL_SIZE)
    ax.set_ylabel('Normalized Rating', fontsize=LABEL_SIZE)
    ax.set_xticks(positions)
    ax.set_xticklabels([DIMENSION_NAME_MAP.get(dim, dim) for dim in dimensions],
                       rotation=45, ha='right', fontsize=TICK_SIZE)

    legend_handles = [
        Patch(facecolor=color, edgecolor='black', label=condition, alpha=0.9)
        for condition, color in zip(
            ['With Reasoning', 'Without Reasoning'],
            [COLORS['primary'], COLORS['secondary']]
        )
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=LEGEND_SIZE)

    ax.set_ylim(0.0, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    plt.tight_layout()
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def main() -> None:
    output_dir = 'analysis'
    os.makedirs(output_dir, exist_ok=True)

    stats_path = os.path.join(output_dir, 'reasoning_statistical_analysis.csv')
    plot_path = os.path.join(output_dir, 'reasoning_box_plot.png')

    reasoning_path = "../../results/reasoning/bulk/context_hiring_manager.csv"
    no_reasoning_path = "../../results/non_reasoning/bulk/context_hiring_manager.csv"

    df_reasoning, df_no_reasoning = load_data(reasoning_path, no_reasoning_path)
    results_df = create_reason_statistical_analysis(df_reasoning, df_no_reasoning, stats_path)
    create_reason_box_plot(df_reasoning, df_no_reasoning, results_df, plot_path)
    print("Analysis complete. Results saved to the 'analysis' directory.")


if __name__ == "__main__":
    main()
