import os
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from stability.implicit.constants_implicit_stability import DIMENSION_NAME_MAP
from style import (apply_standard_style, style_axis, add_significance_annotation,
                   COLORS, DPI, LEGEND_SIZE, LABEL_SIZE, TICK_SIZE)


def load_data(data_file: str, dimensions: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        df = pd.read_excel(data_file)
    except Exception as e:
        raise Exception(f"Error loading data file: {e}")

    df_countries = df[df['type'] == 'human'].copy()
    df_llm = df[df['type'] == 'llm'].copy()

    df_countries.set_index('country', inplace=True)
    df_llm.set_index('country', inplace=True)

    df_countries = df_countries.dropna(subset=dimensions)
    df_llm = df_llm.dropna(subset=dimensions)

    return df_countries, df_llm


def scale_data(df: pd.DataFrame, dims: List[str]) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(df[dims])


def perform_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X)
    return kmeans.labels_


def find_optimal_clusters(df: pd.DataFrame, dims: List[str], min_k: int = 2, max_k: int = 10,
                          random_state: int = 42) -> Tuple[int, float, Dict[int, float]]:
    X = scale_data(df, dims)
    n_samples = X.shape[0]
    max_k = min(max_k, n_samples - 1) if n_samples > min_k else min_k

    best_k = min_k
    best_score = -1
    silhouette_dict = {}

    for k in range(min_k, max_k + 1):
        if k >= n_samples:
            continue

        labels = perform_kmeans(X, k, random_state)
        if len(np.unique(labels)) < 2:
            continue

        score = silhouette_score(X, labels)
        silhouette_dict[k] = score
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score, silhouette_dict


def create_clusters(df: pd.DataFrame, dims: List[str], n_clusters: int, random_state: int = 42) -> np.ndarray:
    X = scale_data(df, dims)
    return perform_kmeans(X, n_clusters, random_state)


def permutation_test(labels_a: np.ndarray, labels_b: np.ndarray, num_permutations: int = 10000,
                     random_state: int = 42) -> Tuple[float, float]:
    np.random.seed(random_state)
    observed_ari = adjusted_rand_score(labels_a, labels_b)

    permuted_labels = np.array([np.random.permutation(labels_b) for _ in range(num_permutations)])
    ari_scores = np.array([adjusted_rand_score(labels_a, pl) for pl in permuted_labels])
    count = np.sum(ari_scores >= observed_ari)
    p_value = max(count / num_permutations, 1 / num_permutations)
    return observed_ari, p_value


def compute_ari(ref_labels: np.ndarray, labels: np.ndarray, run_permutation_tests: bool = True,
                num_permutations: int = 10000) -> Tuple[float, Optional[float]]:
    ari = adjusted_rand_score(ref_labels, labels)
    p_value = None
    if run_permutation_tests:
        _, p_value = permutation_test(ref_labels, labels, num_permutations=num_permutations)
    return ari, p_value


def perform_ari_analysis(df: pd.DataFrame, dims: List[str], use_optimal_k: bool, k_range: Tuple[int, int],
                         run_permutation_tests: bool, num_permutations: int) -> Tuple[pd.DataFrame, int]:
    if use_optimal_k:
        best_k, _, _ = find_optimal_clusters(df, dims, min_k=k_range[0], max_k=k_range[1])
    else:
        best_k = 3

    df['Collective_Cluster'] = create_clusters(df, dims, n_clusters=best_k)

    ari_results = []
    for r in range(1, len(dims)):
        for subset in combinations(dims, r):
            subset_dims = list(subset)
            labels = create_clusters(df, subset_dims, n_clusters=best_k)
            ref_labels = df['Collective_Cluster'].values

            ari, p_value = compute_ari(ref_labels, labels, run_permutation_tests, num_permutations)

            ari_results.append({
                'Dimensions': ",".join(subset_dims),
                'Num_Dimensions': r,
                'Observed_ARI': ari,
                'P_value': p_value
            })

    df_ari = pd.DataFrame(ari_results)
    return df_ari, best_k


def analyze_dimension_impact(df_ari: pd.DataFrame, dims: List[str]) -> pd.DataFrame:
    results = []
    for D in dims:
        for r in sorted(df_ari['Num_Dimensions'].unique()):
            subsets_r = df_ari[df_ari['Num_Dimensions'] == r]
            group_a = subsets_r[subsets_r['Dimensions'].str.contains(rf'\b{D}\b')]['Observed_ARI']
            group_b = subsets_r[~subsets_r['Dimensions'].str.contains(rf'\b{D}\b')]['Observed_ARI']

            if len(group_a) > 0 and len(group_b) > 0:
                mean_a = group_a.mean()
                mean_b = group_b.mean()
                diff = mean_a - mean_b

                p_values = subsets_r[subsets_r['Dimensions'].str.contains(rf'\b{D}\b')]['P_value']
                median_p = p_values.median() if not p_values.empty else None

                results.append({
                    'Dimension': D,
                    'Num_Dimensions': r,
                    'Mean_ARI_Including_D': mean_a,
                    'Mean_ARI_Excluding_D': mean_b,
                    'MeanDiff': diff,
                    'P_value': median_p
                })

    return pd.DataFrame(results)


def plot_ari_box_plot(df_ari_countries: pd.DataFrame, df_ari_llm: pd.DataFrame,
                      output_dir: str, filename: str) -> None:
    apply_standard_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    df_ari_countries_box = df_ari_countries.copy()
    df_ari_countries_box['Group'] = 'Countries'

    df_ari_llm_box = df_ari_llm.copy()
    df_ari_llm_box['Group'] = 'LLM'

    df_combined = pd.concat([df_ari_countries_box, df_ari_llm_box], ignore_index=True)
    df_combined['Num_Dimensions'] = df_combined['Num_Dimensions'].astype(str)

    style_axis(ax, xlabel='Number of Dimensions', ylabel='Adjusted Rand Index (ARI)')

    sns.boxplot(
        x='Num_Dimensions',
        y='Observed_ARI',
        hue='Group',
        data=df_combined,
        palette={'Countries': COLORS['primary'], 'LLM': COLORS['secondary']},
        showfliers=True,
        showmeans=True,
        ax=ax
    )

    ax.axhline(y=0, color=COLORS['red'], linestyle='--', label='Random Chance')
    ax.set_ylim(-0.25, 1.05)

    medians = df_combined.groupby(['Group', 'Num_Dimensions'])['Observed_ARI'].median().reset_index()
    sorted_dimensions = sorted(df_combined['Num_Dimensions'].unique(), key=lambda x: int(x))
    dimension_to_x = {dim: idx for idx, dim in enumerate(sorted_dimensions)}

    median_positions = {'Countries': [], 'LLM': []}

    for group in ['Countries', 'LLM']:
        group_medians = medians[medians['Group'] == group].sort_values('Num_Dimensions')
        for _, row in group_medians.iterrows():
            x = dimension_to_x[row['Num_Dimensions']]
            median_positions[group].append((x, row['Observed_ARI']))

    for group, color in zip(['Countries', 'LLM'], [COLORS['primary'], COLORS['secondary']]):
        xs, ys = zip(*median_positions[group])
        if group == 'Countries':
            xs = np.array(xs) - 0.2
        else:
            xs = np.array(xs) + 0.2
        ax.plot(xs, ys, '-', color=color, alpha=0.7, linewidth=1.5, zorder=3)
        ax.scatter(xs, ys, color=color, s=50, zorder=4, edgecolor='black', linewidth=0.7)

    for dim in sorted(df_ari_countries['Num_Dimensions'].unique(), key=lambda x: int(x)):
        x_pos = dimension_to_x[str(dim)]

        country_pvals = df_ari_countries[df_ari_countries['Num_Dimensions'] == dim]['P_value'].dropna()
        if not country_pvals.empty:
            median_p = country_pvals.median()
            y_max = df_combined[
                (df_combined['Num_Dimensions'] == str(dim)) &
                (df_combined['Group'] == 'Countries')
                ]['Observed_ARI'].max()
            add_significance_annotation(ax, median_p, x_pos - 0.2, y_max + 0.02, color=COLORS['primary'])

        llm_pvals = df_ari_llm[df_ari_llm['Num_Dimensions'] == dim]['P_value'].dropna()
        if not llm_pvals.empty:
            median_p = llm_pvals.median()
            y_max = df_combined[
                (df_combined['Num_Dimensions'] == str(dim)) &
                (df_combined['Group'] == 'LLM')
                ]['Observed_ARI'].max()
            add_significance_annotation(ax, median_p, x_pos + 0.2, y_max + 0.02, color=COLORS['secondary'])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_dimension_impact_ari_bar_plot(dimension_impact_df_countries: pd.DataFrame,
                                       dimension_impact_df_llm: pd.DataFrame,
                                       output_dir: str, filename: str) -> None:
    apply_standard_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    df_countries = dimension_impact_df_countries.copy()
    df_llm = dimension_impact_df_llm.copy()

    df_countries['Dimension'] = df_countries['Dimension'].map(DIMENSION_NAME_MAP)
    df_llm['Dimension'] = df_llm['Dimension'].map(DIMENSION_NAME_MAP)

    agg_countries = df_countries.groupby('Dimension')['MeanDiff'].mean().reset_index()
    agg_countries['Group'] = 'Countries'
    agg_llm = df_llm.groupby('Dimension')['MeanDiff'].mean().reset_index()
    agg_llm['Group'] = 'LLM'

    aggregated = pd.concat([agg_countries, agg_llm], ignore_index=True)

    sns.barplot(x='MeanDiff',
                y='Dimension',
                hue='Group',
                data=aggregated,
                palette={'Countries': COLORS['primary'], 'LLM': COLORS['secondary']},
                ax=ax,
                dodge=True)

    for i, (dim, group) in enumerate(zip(aggregated['Dimension'], aggregated['Group'])):
        dim_code = [k for k, v in DIMENSION_NAME_MAP.items() if v == dim][0]

        if group == 'Countries':
            pvals = dimension_impact_df_countries[
                dimension_impact_df_countries['Dimension'] == dim_code
                ]['P_value'].dropna()
            color = COLORS['primary']
        elif group == 'LLM':
            pvals = dimension_impact_df_llm[
                dimension_impact_df_llm['Dimension'] == dim_code
                ]['P_value'].dropna()
            color = COLORS['secondary']
        else:
            pvals = pd.Series()
            color = 'black'

        if not pvals.empty:
            median_p = pvals.median()
            bar = ax.patches[i]
            if bar.get_width() >= 0:
                x_pos = bar.get_width() + 0.005
            else:
                x_pos = bar.get_width() - 0.02
            y_pos = bar.get_y() + bar.get_height() / 2
            add_significance_annotation(ax, median_p, x_pos, y_pos, color=color)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0] * 1.2, current_xlim[1] * 1.2)

    ax.set_xlabel('Average Mean Cluster Similarity (ARI) Difference', fontsize=LABEL_SIZE)
    ax.set_ylabel('Dimension', fontsize=LABEL_SIZE)

    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=DPI, bbox_inches='tight')
    plt.close()


def main(
        data_file: str,
        run_permutation_tests: bool = True,
        num_permutations: int = 10000,
        output_dir: str = 'analysis',
        use_optimal_k: bool = False,
        k_range: Tuple[int, int] = (2, 10)
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    dimensions = list(DIMENSION_NAME_MAP.keys())

    # Define filenames in main
    box_plot_filename = 'ari_box_plot.png'
    bar_plot_filename = 'ari_dimension_impact_bar_plot.png'

    df_countries, df_llm = load_data(data_file, dimensions)

    df_ari_countries, best_k_countries = perform_ari_analysis(
        df_countries, dimensions, use_optimal_k, k_range,
        run_permutation_tests, num_permutations
    )

    df_ari_llm, best_k_llm = perform_ari_analysis(
        df_llm, dimensions, use_optimal_k, k_range,
        run_permutation_tests, num_permutations
    )

    df_ari_countries.to_csv(os.path.join(output_dir, 'incremental_dimension_ari_results_countries.csv'), index=False)
    df_ari_llm.to_csv(os.path.join(output_dir, 'incremental_dimension_ari_results_llm.csv'), index=False)

    dimension_impact_df_countries = analyze_dimension_impact(df_ari_countries, dimensions)
    dimension_impact_df_llm = analyze_dimension_impact(df_ari_llm, dimensions)

    dimension_impact_df_countries.to_csv(os.path.join(output_dir, 'dimension_impact_results_countries.csv'),
                                         index=False)
    dimension_impact_df_llm.to_csv(os.path.join(output_dir, 'dimension_impact_results_llm.csv'), index=False)

    plot_ari_box_plot(df_ari_countries, df_ari_llm, output_dir, box_plot_filename)
    plot_dimension_impact_ari_bar_plot(dimension_impact_df_countries, dimension_impact_df_llm, output_dir,
                                       bar_plot_filename)


if __name__ == '__main__':
    output_dir = 'analysis'
    os.makedirs(output_dir, exist_ok=True)
    results_filepath = 'results/hofstredes_dimensional_data.xlsx'

    try:
        main(
            data_file=results_filepath,
            run_permutation_tests=True,
            num_permutations=50,
            output_dir=output_dir,
            use_optimal_k=True,
            k_range=(2, 10)
        )
    except Exception as e:
        print(f"An error occurred during analysis: {e}")