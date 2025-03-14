import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

STYLE = 'seaborn-v0_8'
DPI = 300
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'gray': '#95a5a6',
    'red': '#e74c3c',
    'blue': '#3498db',
    'black': '#000000'
}
FONT_FAMILY = 'TIMES NEW ROMAN'
TITLE_SIZE = 12
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10


def apply_standard_style():
    plt.style.use(STYLE)
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': TICK_SIZE,
        'axes.titlesize': TITLE_SIZE,
        'axes.labelsize': LABEL_SIZE,
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': LEGEND_SIZE,
        'figure.dpi': DPI
    })


def style_axis(ax, title=None, xlabel=None, ylabel=None, show_grid=True):
    if title:
        ax.set_title(title, pad=20)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Grid settings
    ax.grid(False)
    if show_grid:
        ax.yaxis.grid(True, linestyle='--', alpha=0.2)


def add_significance_annotation(ax, p_value, x_pos, y_pos, color='black'):
    stars = ''
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'

    if stars:
        text = ax.text(x_pos, y_pos, stars,
                       ha='center', va='bottom',
                       fontsize=LABEL_SIZE,
                       fontweight='bold',
                       color=color)

        # Add black outline if color is not black
        if color.lower() != 'black':
            text.set_path_effects([
                path_effects.Stroke(linewidth=0.5, foreground='black'),  # Black outline
                path_effects.Normal()  # Fill with original color
            ])
