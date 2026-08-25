"""Shared matplotlib style settings for publication-ready figures."""
import matplotlib.pyplot as plt


def set_publication_style(font="Arial", dpi=600):
    """Apply a consistent, journal-ready matplotlib style."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font, "DejaVu Sans", "Liberation Sans"],
        "figure.dpi": 150,
        "savefig.dpi": dpi,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "svg.fonttype": "none",
    })
