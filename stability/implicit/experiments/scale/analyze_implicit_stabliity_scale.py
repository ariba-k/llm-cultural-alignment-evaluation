import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from constants import DIMENSION_NAME_MAP
from style import (apply_standard_style, style_axis, add_significance_annotation,
                   COLORS, TICK_SIZE, LABEL_SIZE, LEGEND_SIZE, DPI)


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])

    df['likert_scale'] = pd.to_numeric(df['likert_scale'], errors='coerce')
    df = df.dropna(subset=['likert_scale'])
    df['likert_scale'] = df['likert_scale'].astype(int)

    valid_ratings = df.apply(lambda row: 1 <= row['rating'] <= row['likert_scale'], axis=1)
    df = df[valid_ratings]

    df['normalized_rating'] = df.apply(lambda row: (row['rating'] - 1) / (row['likert_scale'] - 1), axis=1)

    return df


def calculate_scale_effect_size(ratings: np.ndarray, scales: np.ndarray) -> float:
    mask = ~np.isnan(ratings)
    ratings = ratings[mask]
    scales = scales[mask]

    unique_scales = np.unique(scales)
    scale_sizes = np.array([np.sum(scales == s) for s in unique_scales])
    scale_means = np.array([np.mean(ratings[scales == s]) for s in unique_scales])

    weights = scale_sizes / np.sum(scale_sizes)
    weighted_mean = np.sum(scale_means * weights)
    weighted_variance = np.sum(weights * (scale_means - weighted_mean) ** 2)

    return np.sqrt(weighted_variance)


def calculate_scale_p_value(ratings: np.ndarray,
                            scales: np.ndarray,
                            n_permutations: int = 10000,
                            seed: Optional[int] = None) -> float:
    if seed is not None:
        np.random.seed(seed)

    observed_effect = calculate_scale_effect_size(ratings, scales)
    if np.isnan(observed_effect):
        return np.nan

    count = 0
    for _ in range(n_permutations):
        perm_ratings = np.random.permutation(ratings)
        perm_effect = calculate_scale_effect_size(perm_ratings, scales)
        if perm_effect >= observed_effect:
            count += 1

    return count / n_permutations


def create_scale_statistical_analysis(df: pd.DataFrame, stats_path: str) -> pd.DataFrame:
    dimensions = sorted(df['dimension'].unique())
    results = []

    for dimension in dimensions:
        dim_data = df[df['dimension'] == dimension]
        ratings = dim_data['normalized_rating'].values
        scales = dim_data['likert_scale'].values

        effect_size = calculate_scale_effect_size(ratings, scales)
        p_value = calculate_scale_p_value(ratings, scales)

        results.append({
            'dimension': DIMENSION_NAME_MAP.get(dimension, dimension),
            'effect_size': round(effect_size, 3) if not np.isnan(effect_size) else np.nan,
            'p_value': round(p_value, 3) if not np.isnan(p_value) else np.nan
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(stats_path, index=False)

    return results_df


def create_scale_box_plot(df: pd.DataFrame, stats_df: pd.DataFrame, plot_path: str) -> None:
    apply_standard_style()

    dimensions = sorted(df['dimension'].unique())
    scales = sorted(df['likert_scale'].unique())
    positions = np.arange(len(dimensions)) * 4

    scale_data = {scale: [] for scale in scales}
    for dim in dimensions:
        dim_data = df[df['dimension'] == dim]
        for scale in scales:
            scale_data[scale].append(
                dim_data[dim_data['likert_scale'] == scale]['normalized_rating'].dropna()
            )

    fig, ax = plt.subplots(figsize=(10, 4))
    style_axis(ax)

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]
    offsets = [-0.7, 0, 0.7]

    for scale, offset, color in zip(scales, offsets, colors):
        ax.boxplot(
            scale_data[scale],
            positions=positions + offset,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            meanprops={'color': 'black', 'linestyle': '-', 'linewidth': 1},
            medianprops={'visible': False},
            boxprops={'facecolor': color, 'alpha': 0.7},
            flierprops={'marker': 'o', 'markerfacecolor': color, 'markersize': 4, 'alpha': 0.7}
        )

    for i, dim in enumerate(dimensions):
        mapped_dim = DIMENSION_NAME_MAP.get(dim, dim)
        result = stats_df[stats_df['dimension'] == mapped_dim]
        if not result.empty and result.iloc[0]['p_value'] < 0.05:
            max_values = [
                data[i].max() if not data[i].empty else -np.inf
                for data in scale_data.values()
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
        Patch(facecolor=color, edgecolor='black', label=f'{scale}-point scale', alpha=0.9)
        for scale, color in zip(scales, colors)
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=LEGEND_SIZE)

    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

    plt.tight_layout()
    plt.savefig(plot_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def main() -> None:
    output_dir = "analysis"
    os.makedirs(output_dir, exist_ok=True)

    stats_path = os.path.join(output_dir, 'likert_scale_statistical_analysis.csv')
    plot_path = os.path.join(output_dir, 'likert_scale_box_plot.png')

    likert_path = "../../results/non_reasoning/bulk/context_hiring_manager.csv"

    df = load_data(likert_path)
    stats_df = create_scale_statistical_analysis(df, stats_path)
    create_scale_box_plot(df, stats_df, plot_path)
    print("Analysis complete. Results saved to the 'analysis' directory.")


if __name__ == "__main__":
    main()
