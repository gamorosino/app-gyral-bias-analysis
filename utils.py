import numpy as np
from matplotlib import pyplot as plt

import numpy as np

# def nature_style_plot(
#     ax,
#     ymin=None,
#     ymax=None,
#     n_yticks=3,
#     spine_width=2,
#     tick_length=6,
#     tick_width=2,
#     fontsize=16,
#     y_decimals=3,
#     add_origin_padding=True, 
#     pad_fraction=0.02
# ):
#     """
#     Apply Nature-style formatting to matplotlib axes.

#     Includes:
#     - left/bottom spines only
#     - outward ticks
#     - optional sparse y-ticks
#     - trimmed spines for L-style axis ending
#     """
#     # Remove top/right spines
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     # Style spines
#     ax.spines["left"].set_linewidth(spine_width)
#     ax.spines["bottom"].set_linewidth(spine_width)

#     # Tick style
#     ax.tick_params(
#         axis="both",
#         direction="out",
#         length=tick_length,
#         width=tick_width,
#         labelsize=fontsize
#     )

#     ax.xaxis.set_ticks_position("bottom")
#     ax.yaxis.set_ticks_position("left")

#     # Y limits
#     if ymin is not None and ymax is not None:
#         ax.set_ylim(ymin, ymax)

#     # Optional y-tick control
#     if n_yticks is not None:
#         ymin_current, ymax_current = ax.get_ylim()

#         if n_yticks == 3:
#             mid = (ymin_current + ymax_current) / 2
#             ticks = [ymin_current, mid, ymax_current]
#         elif n_yticks == 2:
#             ticks = [ymin_current, ymax_current]
#         else:
#             ticks = np.linspace(ymin_current, ymax_current, n_yticks)

#         ax.set_yticks(ticks)
#         ax.set_yticklabels([f"{t:.{y_decimals}f}" for t in ticks])

#     # --- KEY: trim spines to tick range ---
#     ymin_current, ymax_current = ax.get_ylim()
#     ax.spines["left"].set_bounds(ymin_current, ymax_current)

#     xticks = ax.get_xticks()
#     if len(xticks) > 0:
#         ax.spines["bottom"].set_bounds(xticks[0], xticks[-1])

#     if add_origin_padding:
#         xmin, xmax = ax.get_xlim()
#         ymin, ymax = ax.get_ylim()

#         xpad = pad_fraction * (xmax - xmin)
#         ypad = pad_fraction * (ymax - ymin)

#         ax.set_xlim(xmin - xpad, xmax)
#         ax.set_ylim(ymin - ypad, ymax)

#     return ax

def nature_style_plot(
    ax,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    xticks=None,
    yticks=None,
    n_xticks=None,
    n_yticks=3,
    spine_width=2,
    tick_length=6,
    tick_width=2,
    fontsize=16,
    x_decimals=0,
    y_decimals=3,
    add_origin_padding=True,
    pad_fraction=0.02,
    format_xticklabels=True,
    format_yticklabels=True
):
    """
    Apply Nature-style formatting to matplotlib axes.

    Priority:
    1. explicit xticks / yticks
    2. n_xticks / n_yticks
    3. keep existing ticks
    """

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Style spines
    ax.spines["left"].set_linewidth(spine_width)
    ax.spines["bottom"].set_linewidth(spine_width)

    # Tick style
    ax.tick_params(
        axis="both",
        direction="out",
        length=tick_length,
        width=tick_width,
        labelsize=fontsize
    )

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # Set limits
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)

    # Optional padding
    if add_origin_padding:
        xmin_current, xmax_current = ax.get_xlim()
        ymin_current, ymax_current = ax.get_ylim()

        xpad = pad_fraction * (xmax_current - xmin_current)
        ypad = pad_fraction * (ymax_current - ymin_current)

        ax.set_xlim(xmin_current - xpad, xmax_current)
        ax.set_ylim(ymin_current - ypad, ymax_current)

    # ----- X ticks -----
    if xticks is not None:
        ax.set_xticks(xticks)
        if format_xticklabels:
            ax.set_xticklabels([f"{t:.{x_decimals}f}" for t in xticks])
    elif n_xticks is not None:
        xmin_current, xmax_current = ax.get_xlim()
        if n_xticks == 3:
            ticks = [xmin_current, (xmin_current + xmax_current) / 2, xmax_current]
        elif n_xticks == 2:
            ticks = [xmin_current, xmax_current]
        else:
            ticks = np.linspace(xmin_current, xmax_current, n_xticks)

        ax.set_xticks(ticks)
        if format_xticklabels:
            ax.set_xticklabels([f"{t:.{x_decimals}f}" for t in ticks])

    # ----- Y ticks -----
    if yticks is not None:
        ax.set_yticks(yticks)
        if format_yticklabels:
            ax.set_yticklabels([f"{t:.{y_decimals}f}" for t in yticks])
    elif n_yticks is not None:
        ymin_current, ymax_current = ax.get_ylim()
        if n_yticks == 3:
            ticks = [ymin_current, (ymin_current + ymax_current) / 2, ymax_current]
        elif n_yticks == 2:
            ticks = [ymin_current, ymax_current]
        else:
            ticks = np.linspace(ymin_current, ymax_current, n_yticks)

        ax.set_yticks(ticks)
        if format_yticklabels:
            ax.set_yticklabels([f"{t:.{y_decimals}f}" for t in ticks])

    # Trim spines to final ticks
    xticks_final = ax.get_xticks()
    yticks_final = ax.get_yticks()

    if len(xticks_final) > 0:
        ax.spines["bottom"].set_bounds(xticks_final[0], xticks_final[-1])

    if len(yticks_final) > 0:
        ax.spines["left"].set_bounds(yticks_final[0], yticks_final[-1])

    return ax



def cortical_area_equal_bins(max_ecc=90, n_bins=10, area='V1'):
    """
    Equal cortical *surface area* bins for visual areas V1/V2/V3,
    using Dougherty et al. (2003) magnification parameters.

    Parameters from Eq. 3 of https://doi.org/10.1167/3.10.1:
        V1: A=29.2 mm, E2=3.67°
        V2: A=22.8 mm, E2=2.54°
        V3: A=19.4 mm, E2=2.69°
    """
    params = {
        'V1': (29.2, 3.67),
        'V2': (22.8, 2.54),
        'V3': (19.4, 2.69)
    }
    A, E2 = params[area]

    # cumulative cortical area A_c(E) ~ A^2 * log(E + E2)
    def F(E):
        return A**2 * np.log(E + E2)

    F0 = F(0)
    Fmax = F(max_ecc)

    # equal area steps
    steps = np.linspace(F0, Fmax, n_bins + 1)

    # invert:
    # F = A^2 * log(E + E2)
    # => E = exp(F/A^2) - E2
    bins = np.exp(steps / A**2) - E2
    return bins


# Dougherty et al. (2003) parameters for Eq. 3 of https://doi.org/10.1167/3.10.1
PARAMS = {
    'V1': (29.2, 3.67),
    'V2': (22.8, 2.54),
    'V3': (19.4, 2.69),
}

import numpy as np

# Dougherty et al. (2003) parameters (Eq. 3 magnification)
PARAMS = {
    'V1': (29.2, 3.67),
    'V2': (22.8, 2.54),
    'V3': (19.4, 2.69),
}


def equal_cortical_area_bins(max_ecc=90, n_bins=20,  min_width=None, area='V1'):
    """
    Compute eccentricity bin edges such that each bin spans equal cortical surface area
    according to Dougherty et al. (2003) magnification model (Eq. 3).

    Parameters:
    -----------
    max_ecc : float
        Maximum eccentricity (e.g., 90 degrees)
    n_bins : int
        Desired number of equal-area bins (before enforcing min-width)
    area : str
        'V1', 'V2', or 'V3' (selects A and E2 parameters)
    min_width : float or None
        Minimum eccentricity width allowed per bin (e.g. 1 degree).
        If None → no constraint. If float → bins smaller than min_width
        will be merged upward until all are ≥ min_width.

    Returns:
    --------
    bins : numpy array
        Monotonic vector of eccentricity bin edges satisfying all constraints.
    """

    A, E2 = PARAMS[area]

    # cumulative cortical area as a function of eccentricity:
    # F(E) = 2π A² * log(E + E2)
    def F(E):
        return 2 * np.pi * A**2 * np.log(E + E2)

    # desired equal-area edges (unconstrained)
    F_min = F(0)
    F_max = F(max_ecc)
    steps = np.linspace(F_min, F_max, n_bins + 1)

    # invert F(E) to get E(F):
    # E = exp(F/(2πA²)) - E2
    bins = np.exp(steps / (2 * np.pi * A**2)) - E2

    # If no min_width constraint → done
    if min_width is None:
        return bins

    # Enforce minimum width
    # ---------------------------------------------------
    # We walk through the bin edges and merge bins that are too small.
    merged = [bins[0]]
    for i in range(1, len(bins)):
        if (bins[i] - merged[-1]) < min_width:
            # skip this edge (merge with next)
            continue
        merged.append(bins[i])

    # ensure last bin ends exactly at max_ecc
    if merged[-1] < max_ecc:
        merged[-1] = max_ecc

    return np.array(merged)


def cortical_length_equal_bins(max_ecc=90, n_bins=10, A=17.3, E2=0.75):
    """
    Equal cortical DISTANCE bins, not area.
    """

    def X(E):
        return A * np.log((E + E2) / E2)

    X_max = X(max_ecc)
    steps = np.linspace(0, X_max, n_bins + 1)

    # invert X(E)
    # E = E2*(exp(X/A)-1)
    bins = E2 * (np.exp(steps / A) - 1)
    return bins

def save_figure(path, dpi=300):
    """Save plot in PNG and vector formats."""
    plt.savefig(path.with_suffix(".png"), dpi=dpi)
    plt.savefig(path.with_suffix(".pdf"))
    plt.savefig(path.with_suffix(".svg"))
