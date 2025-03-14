import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from environment import MODEL_MAP, MODEL_NAME_MAP
from style import (
    apply_standard_style, style_axis, COLORS, DPI, LEGEND_SIZE
)

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def process_results(results, options_processed, data):
    vectors = []
    for _, model_results in results.items():
        for country, answers in model_results.items():
            answer_vec = []
            for a, o in zip(answers, options_processed):
                num_answers = len(o)
                try:
                    answer_num = alphabet.index(a)
                    answer_vec.append(answer_num / (num_answers - 1))
                except:
                    answer_vec.append(0.5)
            vectors.append(answer_vec)

    return np.vstack([data, np.array(vectors)])


def calculate_distances(all_data, country_names):
    num_countries = len(country_names)
    num_examples = all_data.shape[0]

    def mean_dist(vecs):
        dists = np.sqrt(np.sum((vecs[:, np.newaxis, :] - vecs[np.newaxis, :, :]) ** 2, axis=2))
        return np.sum(dists) / (dists.shape[0] * (dists.shape[0] - 1))

    test_stat = mean_dist(all_data[:num_countries])

    get_rand_idxs = lambda: np.random.choice(
        np.arange(num_countries, num_examples), size=num_countries, replace=False
    )
    ref_stats = [mean_dist(all_data[get_rand_idxs()]) for _ in range(10000)]

    percentile = 100 * np.searchsorted(np.sort(ref_stats), test_stat, side="right") / len(ref_stats)
    ratio = mean_dist(all_data[num_countries:]) / test_stat

    all_ratios = []
    model_index = 0
    for model_type, model_names in MODEL_MAP.items():
        for _ in model_names:
            start_idx = num_countries * (model_index + 1)
            end_idx = num_countries * (model_index + 2)
            all_ratios.append(mean_dist(all_data[start_idx:end_idx]) / test_stat)
            model_index += 1

    print(f'Ratio: {ratio}, all_ratios: {all_ratios}, p value: {percentile}')
    return test_stat, percentile, ratio, all_ratios


def plot_tsne_scatter_plot(tsne_results_default, tsne_results, country_names, file_name,
                           title1="AlKhamissi et al. (2024) Prompt",
                           title2="DSPy Optimized Prompt",
                           analysis_dir="analysis"):

    os.makedirs(analysis_dir, exist_ok=True)
    shapes = ['o', 's', 'D', 'v', '^', '<', '>', 'p', '*', 'h', 'H', 'X', 'd', 'P', '8']
    num_countries = len(country_names)

    model_colors = [COLORS['blue'], COLORS['secondary'], COLORS['tertiary'], COLORS['red'], 'purple', 'brown']

    models_and_colors = [('Human', model_colors[0])]
    for i, model_type in enumerate(sorted(MODEL_NAME_MAP.keys())):
        models_and_colors.append((MODEL_NAME_MAP[model_type], model_colors[i + 1]))

    apply_standard_style()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), dpi=DPI)

    for i in range(len(models_and_colors)):
        for j in range(num_countries):
            for k, tsne_results_set in enumerate([tsne_results_default, tsne_results]):
                axes[k].scatter(
                    tsne_results_set[i * num_countries + j, 0],
                    tsne_results_set[i * num_countries + j, 1],
                    c=models_and_colors[i][1],
                    marker=shapes[j],
                    s=60,
                    alpha=0.7
                )

    model_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=c, markersize=8, label=l)
        for l, c in models_and_colors
    ]

    country_handles = [
        plt.Line2D([0], [0], marker=shape, color='w',
                   markerfacecolor='gray', markersize=8, label=country)
        for shape, country in zip(shapes, country_names)
    ]

    for i, title in enumerate([title1, title2]):
        style_axis(axes[i], title=title, show_grid=False)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.subplots_adjust(right=0.85, wspace=0.1, left=0.05)

    first_legend = axes[1].legend(
        handles=model_handles,
        loc='center left',
        bbox_to_anchor=(1.0, 0.8),
        fontsize=LEGEND_SIZE,
        frameon=True
    )

    axes[1].add_artist(first_legend)
    axes[1].legend(
        handles=country_handles,
        loc='center left',
        bbox_to_anchor=(1.0, 0.35),
        fontsize=LEGEND_SIZE,
        frameon=True
    )

    plt.savefig(
        f'{analysis_dir}/{file_name}',
        format='png',
        bbox_inches="tight",
        dpi=DPI,
        pad_inches=0.5
    )

    plt.show()


def main():
    try:
        with open('results/steerable_results.pkl', 'rb') as f:
            results_data = pickle.load(f)

        country_names = results_data['country_names']

        # Process default (AlKhamissi) results
        all_default_results = results_data['default']['results']
        options_processed = results_data['default']['options_processed']
        data = results_data['default']['data']
        all_data_default = process_results(all_default_results, options_processed, data)

        # Process optimized (DSPy) results
        all_results = results_data['optimized']['results']
        options_processed_test = results_data['optimized']['options_processed']
        data_test = results_data['optimized']['data']
        all_data = process_results(all_results, options_processed_test, data_test)

    except FileNotFoundError:
        print("Results file not found. Please run experiments first.")
        return

    print("Calculating distance metrics for default prompt:")
    calculate_distances(all_data_default, country_names)

    print("Calculating distance metrics for optimized prompt:")
    calculate_distances(all_data, country_names)

    print("Running t-SNE for visualization...")
    tsne_default = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=5000)
    tsne_results_default = tsne_default.fit_transform(all_data_default)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=5000)
    tsne_results = tsne.fit_transform(all_data)

    file_name = 'steerability_tsne_scatter_plot.png'
    print(f"Creating visualization: {file_name}")
    plot_tsne_scatter_plot(tsne_results_default, tsne_results, country_names, file_name)

    print("Analysis completed.")


if __name__ == "__main__":
    main()