#!/usr/bin/python3
r"""
Based on https://jwalton.info/Embed-Publication-Matplotlib-Latex/

To find out the actual \textwidth:

    % your document class here
    \documentclass[10pt]{report}
    \begin{document}

    % gives the width of the current document in pts
    \showthe\textwidth

    % If you’re plotting in a document typeset in columns, you may use \showthe\columnwidth in a similar manner.

    \end{document}


The graphicx package may then be used to insert this figure into LaTeX:

    % your document class here
    \documentclass[10pt]{report}
    % package necessary to inset pdf as image
    \usepackage{graphicx}

    \begin{document}

    \begin{figure}[!htbp]
    	\centering
    	\includegraphics{/path/to/directory/example_1.pdf}
    	\caption{Our first figure.}
        \label{fig:our-first-figure}
    \end{figure}

    \end{document}

"""

def set_size(width, fraction=1, subplot=[1, 1]):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Example usage:
        fig, ax = plt.subplots(5, 2, figsize=set_size(width, subplot=[5, 2]))
    to prepare a plot with 5 rows, 2 columns.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy
    subplot: list
            [rows, columns] of subplots in plot

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches

    """
    if width == 'thesis':
        width_pt = 433.62  # adjusted for my actual thesis template
    elif width == 'beamer':
        width_pt = 307.28987
    elif width == 'pnas':
        width_pt = 246.09686
    else:
        width_pt = width

    # Width of figure
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplot[0] / subplot[1])
    # fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def setup_mpl_for_latex():
    import matplotlib as mpl

    nice_settings = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": [],
    # Use 10pt font in plots, to match 10pt font in document
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "axes.labelsize": 8,  # was 10
    "axes.titlesize": 8,  # was 10
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.4,  # spine thickness
    "axes.edgecolor": 'grey',
    "axes.spines.right": False,
    "axes.spines.top": False,
    }

    mpl.rcParams.update(nice_settings)


""" A simple example of creating a figure with text rendered in LaTeX. """
#
# import numpy as np
# import matplotlib.pyplot as plt
# # from my_plot import set_size
#
# # Using seaborn's style
# plt.style.use('seaborn')
# width = 345
#
# setup_mpl_for_latex()
#
# x = np.linspace(0, 2*np.pi, 100)
# # Initialise figure instance
# fig, ax = plt.subplots(1, 1, figsize=set_size(width))
#
# # Plot
# ax.plot(x, np.sin(x))
# ax.set_xlim(0, 2*np.pi)
# ax.set_xlabel(r'$\theta$')
# ax.set_ylabel(r'$\sin{(\theta)}$')
#
# # Save and remove excess whitespace
# plt.savefig('example_1.pdf', format='pdf', bbox_inches='tight')
