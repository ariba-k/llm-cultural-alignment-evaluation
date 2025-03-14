import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
from matplotlib.patches import Patch
from stability.implicit.constants_implicit_stability import DIMENSION_MAP, DIMENSION_NAME_MAP
from style import (apply_standard_style, style_axis, add_significance_annotation,
                   COLORS, DPI, TICK_SIZE, LABEL_SIZE, LEGEND_SIZE)


def load_data(comparative_path: str, absolute_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comp_df = pd.read_csv(comparative_path)
    abs_df = pd.read_csv(absolute_path)

    comp_df = comp_df[comp_df['likert_scale'] == 5]
    abs_df = abs_df[abs_df['likert_scale'] == 5]

    rating_columns = [col for col in abs_df.columns if 'Rating' in col]
    abs_df[rating_columns] = abs_df[rating_columns].fillna(3)

    comp_df['centered_rating'] = pd.to_numeric(comp_df['rating'], errors='coerce') - 3
    comp_df['normalized_rating'] = comp_df['centered_rating'] / 2
    comp_df.dropna(subset=['normalized_rating'], inplace=True)

    abs_melted = melt_and_merge_data(abs_df)

    return comp_df, abs_melted


def melt_and_merge_data(abs_df: pd.DataFrame) -> pd.DataFrame:
    """
        1) Compute _diff columns in abs_df for each dimension: (High - Low).
        2) Melt abs_df from wide -> long, so each dimension becomes its own row.
        3) Return the melted abs_df with columns:
           [model_name, question_num, dimension, abs_diff_value, (other columns...)].
    """
    for dim_code, (high_dim, low_dim) in DIMENSION_MAP.items():
        high_col = f"{high_dim}_Cover_Letter_Rating"
        low_col = f"{low_dim}_Cover_Letter_Rating"

        if high_col not in abs_df.columns or low_col not in abs_df.columns:
            continue

        abs_df[f"{dim_code}_norm_diff"] = (abs_df[high_col] - abs_df[low_col]) / 4

    norm_diff_cols = [c for c in abs_df.columns if c.endswith("_norm_diff")]
    abs_melted = abs_df.melt(
        id_vars=["model_name", "likert_scale", "question_num"],
        value_vars=norm_diff_cols,
        var_name="dimension",
        value_name="normalized_diff"
    )

    abs_melted["dimension"] = abs_melted["dimension"].str.replace("_norm_diff", "", regex=False)

    return abs_melted


def calculate_comparative_effect_size(ratings: np.ndarray, conditions: np.ndarray) -> float:
    mask = ~np.isnan(ratings)
    ratings = ratings[mask]
    conditions = conditions[mask]

    unique_conditions = np.unique(conditions)
    sizes = np.array([np.sum(conditions == c) for c in unique_conditions])
    means = np.array([np.mean(ratings[conditions == c]) for c in unique_conditions])

    weights = sizes / np.sum(sizes)
    effect_size = abs(weights[0] * means[0] - weights[1] * means[1])
    return effect_size


def calculate_comparative_p_value(ratings: np.ndarray, conditions: np.ndarray,
                                  n_permutations: int = 10000, seed: Optional[int] = None) -> float:
    if seed is not None:
        np.random.seed(seed)

    observed_effect = calculate_comparative_effect_size(ratings, conditions)
    if np.isnan(observed_effect):
        return np.nan

    count = 0
    for i in range(n_permutations):
        shuffled_ratings = np.random.permutation(ratings)
        perm_effect = calculate_comparative_effect_size(shuffled_ratings, conditions)
        if perm_effect >= observed_effect:
            count += 1

    p_value = count / n_permutations
    return p_value


def create_comparative_statistical_analysis(comp_df: pd.DataFrame, abs_df: pd.DataFrame,
                                stats_path: Path) -> pd.DataFrame:
    dimensions = sorted(comp_df['dimension'].unique())
    results = []

    for dimension in dimensions:
        comp_ratings = comp_df[comp_df['dimension'] == dimension]['normalized_rating'].values
        abs_ratings = abs_df[abs_df['dimension'] == dimension]['normalized_diff'].values

        ratings = np.concatenate([comp_ratings, abs_ratings])
        conditions = np.concatenate([
            np.full(len(comp_ratings), 'comparative'),
            np.full(len(abs_ratings), 'absolute')
        ])

        effect_size = calculate_comparative_effect_size(ratings, conditions)
        p_value = calculate_comparative_p_value(ratings, conditions)

        results.append({
            'dimension': DIMENSION_NAME_MAP.get(dimension, dimension),
            'effect_size': round(effect_size, 3) if not np.isnan(effect_size) else np.nan,
            'p_value': round(p_value, 3) if not np.isnan(p_value) else np.nan
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(stats_path, index=False)

    return results_df


def create_comparative_box_plot(comp_df: pd.DataFrame, abs_df: pd.DataFrame,
                               results_df: pd.DataFrame, output_path: Path) -> None:
    apply_standard_style()

    dimensions = sorted(comp_df['dimension'].unique())
    positions = np.arange(len(dimensions)) * 3

    comp_data = []
    abs_data = []
    for dim in dimensions:
        comp_ratings = comp_df[comp_df['dimension'] == dim]['normalized_rating'].dropna()
        abs_ratings = abs_df[abs_df['dimension'] == dim]['normalized_diff'].dropna()
        comp_data.append(comp_ratings)
        abs_data.append(abs_ratings)

    fig, ax = plt.subplots(figsize=(10, 4))
    style_axis(ax)

    bp1 = ax.boxplot(comp_data,
                     positions=positions - 0.5,
                     widths=0.8,
                     patch_artist=True,
                     showmeans=True,
                     meanline=True,
                     meanprops=dict(color='black', linestyle='-', linewidth=1),
                     medianprops=dict(visible=False),
                     boxprops=dict(facecolor=COLORS['primary'], alpha=0.7),
                     flierprops=dict(marker='o', markerfacecolor=COLORS['primary'],
                                     markersize=4, alpha=0.7))

    bp2 = ax.boxplot(abs_data,
                     positions=positions + 0.5,
                     widths=0.8,
                     patch_artist=True,
                     showmeans=True,
                     meanline=True,
                     meanprops=dict(color='black', linestyle='-', linewidth=1),
                     medianprops=dict(visible=False),
                     boxprops=dict(facecolor=COLORS['secondary'], alpha=0.7),
                     flierprops=dict(marker='o', markerfacecolor=COLORS['secondary'],
                                     markersize=4, alpha=0.7))

    for i, dim in enumerate(dimensions):
        mapped_dim = DIMENSION_NAME_MAP.get(dim, dim)
        result = results_df[results_df['dimension'] == mapped_dim]
        if not result.empty and result.iloc[0]['p_value'] < 0.05:
            comp_max = comp_data[i].max() if not comp_data[i].empty else -np.inf
            abs_max = abs_data[i].max() if not abs_data[i].empty else -np.inf
            max_y = max(comp_max, abs_max)
            annotation_y = min(max_y + 0.1, 1.15)
            add_significance_annotation(ax, result.iloc[0]['p_value'], positions[i], annotation_y)

    ax.set_xlabel('Dimension', fontsize=LABEL_SIZE)
    ax.set_ylabel('Normalized Preference Score\n(Negative = Low, Positive = High)',
                  fontsize=LABEL_SIZE)
    ax.set_xticks(positions)
    ax.set_xticklabels([DIMENSION_NAME_MAP.get(dim, dim) for dim in dimensions],
                       rotation=45, ha='right', fontsize=TICK_SIZE)

    legend_handles = [
        Patch(facecolor=COLORS['primary'], edgecolor='black', label='Comparative Rating', alpha=0.9),
        Patch(facecolor=COLORS['secondary'], edgecolor='black', label='Absolute Rating', alpha=0.9)
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=LEGEND_SIZE)

    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def main():
    comparative_path = '../../results/non_reasoning/bulk/context_hiring_manager.csv'
    absolute_path = 'results/context_hiring_manager_likert_scale_5_non_reasoning_bulk.csv'
    output_dir = Path('analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    comp_df, abs_df = load_data(comparative_path, absolute_path)

    stats_path = output_dir / 'comparative_statistical_analysis.csv'
    results_df = create_comparative_statistical_analysis(comp_df, abs_df, stats_path)

    plot_path = output_dir / 'comparative_box_plot.png'
    create_comparative_box_plot(comp_df, abs_df, results_df, plot_path)
    print("Analysis complete. Results saved to the 'analysis' directory.")


if __name__ == "__main__":
    main()