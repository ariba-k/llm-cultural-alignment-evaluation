import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from constants import DIMENSION_NAME_MAP
from matplotlib.patches import Patch

from style import (apply_standard_style, style_axis, add_significance_annotation,
                   COLORS, DPI, TICK_SIZE, LABEL_SIZE, LEGEND_SIZE)
from stability.implicit.constants_implicit_stability import Parameters


def load_data(results_dir: str, likert_scale: int = 5) -> pd.DataFrame:
    # Include all contexts plus the original context
    all_contexts = Parameters.contexts + [Parameters.original_context[1]]

    paths = {}
    for context in all_contexts:
        context_file = f"context_{context.strip().lower().replace(' ', '_')}.csv"
        paths[context] = os.path.join(results_dir, context_file)

    dfs = []
    for context, path in paths.items():
        df = pd.read_csv(path)
        df = df[df["likert_scale"] == likert_scale].copy()
        df["context_source"] = context
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def calculate_context_effect_size(ratings: np.ndarray, contexts: np.ndarray) -> float:
    mask = ~np.isnan(ratings)
    ratings = ratings[mask]
    contexts = contexts[mask]

    unique_contexts = np.unique(contexts)
    context_sizes = np.array([np.sum(contexts == c) for c in unique_contexts])
    context_means = np.array([np.mean(ratings[contexts == c]) for c in unique_contexts])

    weights = context_sizes / np.sum(context_sizes)
    weighted_mean = np.sum(context_means * weights)
    weighted_variance = np.sum(weights * (context_means - weighted_mean) ** 2)

    return np.sqrt(weighted_variance)


def calculate_context_p_value(ratings: np.ndarray,
                              contexts: np.ndarray,
                              n_permutations: int = 10000,
                              seed: Optional[int] = None) -> float:
    if seed is not None:
        np.random.seed(seed)

    observed_effect = calculate_context_effect_size(ratings, contexts)
    if np.isnan(observed_effect):
        return np.nan

    count = 0
    for _ in range(n_permutations):
        shuffled_ratings = np.random.permutation(ratings)
        perm_effect = calculate_context_effect_size(shuffled_ratings, contexts)
        if perm_effect >= observed_effect:
            count += 1

    return count / n_permutations


def create_context_statistical_analysis(df: pd.DataFrame, stats_path: str) -> pd.DataFrame:
    dimensions = sorted(df['dimension'].unique())
    results = []

    for dimension in dimensions:
        dim_data = df[df['dimension'] == dimension]
        ratings = dim_data['rating'].values
        contexts = dim_data['context_source'].values

        effect_size = calculate_context_effect_size(ratings, contexts)
        p_value = calculate_context_p_value(ratings, contexts)

        results.append({
            'dimension': DIMENSION_NAME_MAP.get(dimension, dimension),
            'effect_size': round(effect_size, 3) if not np.isnan(effect_size) else np.nan,
            'p_value': round(p_value, 3) if not np.isnan(p_value) else np.nan
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(stats_path, index=False)

    return results_df


def create_context_box_plot(df: pd.DataFrame, stats_df: pd.DataFrame, plot_path: str) -> None:
    apply_standard_style()

    df = df.copy()
    df['rating'] = df['rating'] / 5  # Normalize ratings

    dimensions = sorted(df['dimension'].unique())
    positions = np.arange(len(dimensions)) * 4

    # Include all contexts for visualization
    all_contexts = Parameters.contexts + [Parameters.original_context[1]]
    context_data = {context: [] for context in all_contexts}

    for dim in dimensions:
        dim_data = df[df['dimension'] == dim]
        for context in all_contexts:
            context_data[context].append(
                dim_data[dim_data['context_source'] == context]['rating'].dropna()
            )

    fig, ax = plt.subplots(figsize=(10, 4))
    style_axis(ax)

    box_width = 0.6

    num_contexts = len(all_contexts)
    offset_range = np.linspace(-0.7 * (num_contexts - 1) / 2, 0.7 * (num_contexts - 1) / 2, num_contexts)

    color_keys = list(COLORS.keys())
    context_colors = {
        context: COLORS[color_keys[i % len(color_keys)]]
        for i, context in enumerate(all_contexts)
    }

    box_props = {}
    for i, context in enumerate(all_contexts):
        box_props[context] = {
            'positions': positions + offset_range[i],
            'color': context_colors[context]
        }

    boxplots = {}
    for context, props in box_props.items():
        boxplots[context] = ax.boxplot(
            context_data[context],
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
        result = stats_df[stats_df['dimension'] == mapped_dim]
        if not result.empty and result.iloc[0]['p_value'] < 0.05:
            max_values = [
                data[i].max() if not data[i].empty else -np.inf
                for data in context_data.values()
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
        Patch(facecolor=context_colors[context], edgecolor='black', label=context, alpha=0.9)
        for context in all_contexts
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=LEGEND_SIZE)

    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    plt.tight_layout()
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def main() -> None:
    output_dir = "analysis"
    os.makedirs(output_dir, exist_ok=True)

    stats_path = os.path.join(output_dir, 'context_statistical_analysis.csv')
    plot_path = os.path.join(output_dir, 'context_box_plot.png')

    results_dir = "../../results/non_reasoning/bulk"
    df_combined = load_data(results_dir, likert_scale=5)
    stats_df = create_context_statistical_analysis(df_combined, stats_path)
    create_context_box_plot(df_combined, stats_df, plot_path)
    print("Analysis complete. Results saved to the 'analysis' directory.")


if __name__ == "__main__":
    main()