import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import (
    DPI, COLORS, LABEL_SIZE, apply_standard_style, style_axis
)


def load_and_process_data(file_path: str, max_scale: int) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['likert_response'] = df['likert_response'].astype(str)
    df = df[~df['likert_response'].str.contains('INVALID', case=False, na=False)]
    df['likert_value'] = pd.to_numeric(df['likert_response'], errors='coerce')
    df = df.dropna(subset=['likert_value'])
    df = df[(df['likert_value'] >= 1) & (df['likert_value'] <= max_scale)]
    df['likert_value'] = df['likert_value'].astype(int)
    return df


def get_country_preferences(df: pd.DataFrame, max_scale: int) -> pd.DataFrame:
    countries = sorted(list(set(df['country_A'].unique()) | set(df['country_B'].unique())))
    preference_data: Dict[str, List[int]] = {country: [] for country in countries}

    countryA_prefs = df.copy()
    countryA_prefs['preference'] = (max_scale + 1) - countryA_prefs['likert_value']

    for country in countries:
        country_A_data = countryA_prefs[countryA_prefs['country_A'] == country]
        if len(country_A_data) > 0:
            preference_data[country].extend(country_A_data['preference'].tolist())

    countryB_prefs = df.copy()
    for country in countries:
        country_B_data = countryB_prefs[countryB_prefs['country_B'] == country]
        if len(country_B_data) > 0:
            preference_data[country].extend(country_B_data['likert_value'].tolist())

    avg_preferences: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for country, scores in preference_data.items():
        if scores:
            avg_preferences[country] = np.mean(scores)
            counts[country] = len(scores)
        else:
            avg_preferences[country] = 0
            counts[country] = 0

    avg_df = pd.DataFrame([
        {'country': country,
         'avg_preference': avg_preferences[country],
         'count': counts[country]}
        for country in countries
    ])

    if max_scale == 5:
        avg_df['normalized_score'] = (avg_df['avg_preference'] - 1) / 4
    else:
        avg_df['normalized_score'] = (avg_df['avg_preference'] - 1) / 3

    return avg_df


def create_preference_ranking_bar_plot() -> None:
    RESULTS_DIR = 'results'
    STANDARD_FILE = os.path.join(RESULTS_DIR, 'likert_standard_results.csv')
    FORCED_CHOICE_FILE = os.path.join(RESULTS_DIR, 'likert_forced_choice_results.csv')
    OUTPUT_DIR = 'analysis'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    apply_standard_style()

    standard_df = load_and_process_data(STANDARD_FILE, 5)
    forced_choice_df = load_and_process_data(FORCED_CHOICE_FILE, 4)

    standard_prefs = get_country_preferences(standard_df, 5)
    forced_choice_prefs = get_country_preferences(forced_choice_df, 4)

    forced_choice_prefs = forced_choice_prefs.sort_values('normalized_score', ascending=False)
    country_order = forced_choice_prefs['country'].tolist()

    standard_prefs['order'] = standard_prefs['country'].map({country: i for i, country in enumerate(country_order)})
    standard_prefs = standard_prefs.sort_values('order')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    style_axis(ax1, xlabel="Country", ylabel="Normalized Preference Score (0-1)", title="Standard 5-Point Scale")
    style_axis(ax2, xlabel="Country", title="Forced Choice 4-Point Scale")

    bars1 = ax1.bar(standard_prefs['country'], standard_prefs['normalized_score'],
                    color=COLORS['primary'], alpha=0.7)
    if len(bars1) > 0:
        bars1[0].set_color(COLORS['tertiary'])
        bars1[-1].set_color(COLORS['red'])

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=LABEL_SIZE)

    ax1.set_ylim(0, 1.05)
    ax1.set_xticklabels(standard_prefs['country'], rotation=45, ha='right')

    bars2 = ax2.bar(forced_choice_prefs['country'], forced_choice_prefs['normalized_score'],
                    color=COLORS['primary'], alpha=0.7)
    if len(bars2) > 0:
        bars2[0].set_color(COLORS['tertiary'])
        bars2[-1].set_color(COLORS['red'])

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=LABEL_SIZE)

    ax2.set_ylim(0, 1.05)
    ax2.set_xticklabels(forced_choice_prefs['country'], rotation=45, ha='right')

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'country_preference_ranking_bar_plot.png')
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved country preference ranking to {output_path}")

    comparison_data = pd.merge(
        standard_prefs[['country', 'normalized_score']],
        forced_choice_prefs[['country', 'normalized_score']],
        on='country',
        suffixes=('_standard', '_forced')
    )
    comparison_data = comparison_data.sort_values('normalized_score_forced', ascending=False)

    csv_path = os.path.join(OUTPUT_DIR, 'country_preference_comparison.csv')
    comparison_data.to_csv(csv_path, index=False)
    print(f"Saved comparison data to {csv_path}")


if __name__ == "__main__":
    create_preference_ranking_bar_plot()