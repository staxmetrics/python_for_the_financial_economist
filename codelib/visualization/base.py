"""
Contains fundamental plots which can be used for specific implementations.
"""

from typing import NoReturn, Union, List

import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from codelib.visualization.layout import default_colors
from matplotlib.ticker import FuncFormatter
from numpy import ndarray
import seaborn as sns

color_list = ['#1a9641', '#ffffbf', '#d73027']  # red, yellow, green
default_color_map = LinearSegmentedColormap.from_list(name="default", colors=color_list, N=50)

__all__ = ['waterfall_chart', 'risk_waterfall_chart', 'fan_chart', 'correlation_plot', 'correlation_scatter_plot']


"""
Fan chart
"""


def fan_chart(x: ndarray, y: ndarray, **kwargs) -> NoReturn:

    """
    Plots a fan chart.

    If the number of rows of `y` is divisible by 2, the middle row of `y` is plotted as a line in the middle

    Parameters
    ----------
    x: ndarray
        Vector representing the "x-values" of the plot
    y: ndarray
        Matrix of data to plot. Number of columns equal to the length of `x`. Number of rows / 2 is equal to the number
        different colored areas in the plot. It is assumed that values in the first row is smaller than the values in the
        second row and so on.
    **kwargs
        Other keyword-only arguments

    Returns
    -------
        None

    Examples
    --------
    .. plot::
        :include-source:

            import numpy as np
            from codelib.visualization.base import fan_chart
            data = np.array([np.random.normal(size=1000) * s for s in np.arange(0, 1, 0.1)])
            percentiles = np.percentile(data, [10, 20, 50, 80, 90], axis=1)
            fan_chart(np.arange(1, 11, 1), percentiles, labels=['80% CI', '60% CI', 'median'])
            plt.show()

    """

    # defaults
    color_perc = "blue"
    color_median = "red"
    xlabel = None
    ylabel = None
    title = None
    labels = None
    initialize_fig = True

    if 'color' in kwargs:
        color_perc = kwargs['color']
    if 'color_median' in kwargs:
        color_median = kwargs['color_median']
    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    if 'ylabel' in kwargs:
        ylabel = kwargs['ylabel']
    if 'title' in kwargs:
        title = kwargs['title']
    if 'labels' in kwargs:
        labels = True
        labels_to_plot = kwargs['labels']
    if "fig" in kwargs:
        fig = kwargs["fig"]
    if "ax" in kwargs:
        ax = kwargs["ax"]
        initialize_fig = False

    number_of_rows = y.shape[0]
    number_to_plot = number_of_rows // 2

    if labels is None:
        labels_to_plot = ["" for i in range(number_to_plot + number_of_rows % 2)]

    if initialize_fig:
        fig, ax = plt.subplots()

    for i in range(number_to_plot):

        # for plotting below
        values1 = y[i, :]
        values2 = y[i + 1, :]

        # for plotting above
        values3 = y[-2 - i, :]
        values4 = y[-1 - i, :]

        # calculate alpha
        alpha = 0.95 * (i + 1) / number_to_plot

        ax.fill_between(x, values1, values2, alpha=alpha, color=color_perc, label=labels_to_plot[i])
        ax.fill_between(x, values3, values4, alpha=alpha, color=color_perc)

    # plot center value with specific color
    if number_of_rows % 2 == 1:
        ax.plot(x, y[number_to_plot], color=color_median, label=labels_to_plot[-1])

    # add title
    plt.title(title)
    # add label to x axis
    plt.xlabel(xlabel)
    # add label to y axis
    plt.ylabel(ylabel)
    # legend
    if labels:
        ax.legend()


"""
Waterfall charts
"""


def risk_waterfall_chart(individual_risks: Union[np.ndarray, List[float]],
                         total_risk: float,
                         names: Union[List[str], None] = None, **kwargs):

    """
    Plots a waterfall chart with risk decomposition of a portfolio/asset.

    Parameters
    ----------
    individual_risks: ndarray or list
        Vector or list with individual risks
    total_risk: float
        Total risk of portfolio/asset
    names: List
        List of names
    **kwargs
        Other keyword-only arguments

    Returns
    -------
        None

    Examples
    --------
    .. plot::
        :include-source:

        from corelib.plotting import risk_waterfall_chart
        risk_waterfall_chart(names=["Asset 1", "Asset 2", "Asset 3", "Asset 4"],
            individual_risks=[0.1, 0.2, 0.1, 0.4],
            total_risk=0.45)
        plt.show()

    """
    # my_params = mpl.rcParams
    # sns.set(style="white")

    risk_color = default_colors['orange']
    diversification_color = default_colors['green']
    total_risk_color = default_colors['red']
    
    xlabel = ""
    ylabel = ""
    title = ""
    total_risk_label = "Total risk"
    diversification_label = "Diversification"
    formatting = "{:,.1f}"
    add_legend = False

    if 'risk_color' in kwargs:
        risk_color = kwargs['risk_color']
    if 'diversification_color' in kwargs:
        diversification_color = kwargs['diversification_color']
    if 'total_risk_color' in kwargs:
        total_risk_color = kwargs['total_risk_color']
    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    if 'ylabel' in kwargs:
        ylabel = kwargs['ylabel']
    if 'title' in kwargs:
        title = kwargs['title']
    if 'total_risk_label' in kwargs:
        total_risk_label = kwargs['total_risk_label']
    if 'diversification_label' in kwargs:
        diversification_label = kwargs['diversification_label']
    if 'formatting' in kwargs:
        formatting = kwargs['formatting']
    if 'add_legend' in kwargs:
        add_legend = kwargs['add_legend']

    if isinstance(individual_risks, ndarray):
        individual_risks = list(individual_risks)
    if names is None:
        names = ["label" for element in individual_risks]

    # calculated diversification
    diversification = total_risk - np.sum(individual_risks)

    # add diversification and total risk to lists with individual risk and names
    individual_risks.append(diversification)
    names.append(diversification_label)

    fig, ax = waterfall_chart(names, individual_risks, positive_color=risk_color, negative_color=diversification_color,
                              total_color=total_risk_color, xlabel=xlabel, ylabel=ylabel, total_label=total_risk_label,
                              formatting=formatting)

    # add title
    plt.title(title)

    # add legend
    if add_legend:
        patch1 = mpatches.Patch(color=risk_color,
                                label='$\\vert \\beta_F \\vert \\sigma_F$ [factor risk] / $\\sigma_E$ [idio. risk]')
        patch2 = mpatches.Patch(color=diversification_color, label='Diversification')
        patch3 = mpatches.Patch(color=total_risk_color, label='$\\sigma_A$ [total risk]')

        plt.legend(loc='upper center', handles=[patch1, patch2, patch3], bbox_to_anchor=(0.5, -0.22), ncol=3)

    return fig, ax


def waterfall_chart(labels: List[str], values: Union[np.ndarray, List[float]], **kwargs):

    """
    Plots a waterfall chart. Positive values are added and negative values are subtracted.
    The last bar is the grand total.

    Parameters
    ----------
    labels: List
        List with string elements representing labels
    values: float
        Values to include
    **kwargs
        Other keyword-only arguments

    Returns
    -------
        None

    Examples
    --------
    .. plot::
        :include-source:

        from corelib.plotting import waterfall_chart
        waterfall_chart(labels = ["Candy", "Econometrics Books", "Wine"],
        values=[100, 500, 1000],
        ylabel="Money used pr. week")
    """

    positive_color = default_colors['orange']
    negative_color = default_colors['green']
    total_color = default_colors['red']

    xlabel = ""
    ylabel = ""
    title = ""
    threshold = None
    total_label = "Total"
    sort_values = False
    rotation_value = 30
    blank_color = (0, 0, 0, 0)
    formatting = "{:,.1f}"

    pos_text_color = positive_color
    neg_text_color = negative_color

    if 'positive_color' in kwargs:
        positive_color = kwargs['positive_color']
    if 'negative_color' in kwargs:
        negative_color = kwargs['negative_color']
    if 'total_color' in kwargs:
        total_color = kwargs['total_color']
    if 'xlabel' in kwargs:
        xlabel = kwargs['xlabel']
    if 'ylabel' in kwargs:
        ylabel = kwargs['ylabel']
    if 'title' in kwargs:
        title = kwargs['title']
    if 'total_label' in kwargs:
        total_label = kwargs['total_label']
    if 'formatting' in kwargs:
        formatting = kwargs['formatting']

    index = np.array(labels)
    data = np.array(values)

    # sorted by absolute value
    if sort_values:
        abs_data = abs(data)
        data_order = np.argsort(abs_data)[::-1]
        data = data[data_order]
        index = index[data_order]

    # group contributors less than the threshold into 'other'
    if threshold:

        abs_data = abs(data)
        threshold_v = abs_data.max() * threshold

        if threshold_v > abs_data.min():
            index = np.append(index[abs_data >= threshold_v], labels)
            data = np.append(data[abs_data >= threshold_v], sum(data[abs_data < threshold_v]))

    changes = {'amount': data}

    # define format formatter
    def money(x, pos):

        """
        The two args are the value and tick position'

        Parameters
        ----------
        x
        pos

        Returns
        -------

        """

        return formatting.format(x)

    formatter = FuncFormatter(money)

    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(formatter)

    # Store data and create a blank series to use for the waterfall
    trans = pd.DataFrame(data=changes, index=index)
    blank = trans.amount.cumsum().shift(1).fillna(0)

    trans['positive'] = trans['amount'] > 0

    # Get the net total number for the final element in the waterfall
    total = trans.sum().amount
    trans.loc[total_label] = total
    blank.loc[total_label] = total

    # The steps graphically show the levels as well as used for label placement
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan

    # When plotting the last element, we want to show the full bar,
    # Set the blank to 0
    blank.loc[total_label] = 0

    # define bar colors for net bar
    trans.loc[trans['positive'] > 1, 'positive'] = 99
    trans.loc[trans['positive'] < 0, 'positive'] = 99
    trans.loc[(trans['positive'] > 0) & (trans['positive'] < 1), 'positive'] = 99

    trans['color'] = trans['positive']

    trans.loc[trans['positive'] == 1, 'color'] = positive_color
    trans.loc[trans['positive'] == 0, 'color'] = negative_color
    trans.loc[trans['positive'] == 99, 'color'] = total_color

    my_colors = list(trans.color)

    # Plot and label
    my_plot = plt.bar(range(0, len(trans.index)), blank, width=0.5, color=blank_color)
    plt.bar(range(0, len(trans.index)), trans.amount, width=0.6,
            bottom=blank, color=my_colors)

    # connecting lines - figure out later
    # my_plot = lines.Line2D(step.index, step.values, color = "gray")
    # my_plot = lines.Line2D((3,3), (4,4))

    # axis labels
    plt.xlabel("\n" + xlabel)
    plt.ylabel(ylabel + "\n")

    # Get the y-axis position for the labels
    y_height = trans.amount.cumsum().shift(1).fillna(0)

    temp = list(trans.amount)

    # create dynamic chart range
    for i in range(len(temp)):
        if (i > 0) & (i < (len(temp) - 1)):
            temp[i] = temp[i] + temp[i - 1]

    trans['temp'] = temp

    plot_max = trans['temp'].max()
    plot_min = trans['temp'].min()

    # Make sure the plot doesn't accidentally focus only on the changes in the data
    if all(i >= 0 for i in temp):
        plot_min = 0
    if all(i < 0 for i in temp):
        plot_max = 0

    if abs(plot_max) >= abs(plot_min):
        maxmax = abs(plot_max)
    else:
        maxmax = abs(plot_min)

    pos_offset = maxmax / 40

    plot_offset = maxmax / 15  # needs to me cumulative sum dynamic

    # Start label loop
    loop = 0
    for index, row in trans.iterrows():
        # For the last item in the list, we don't want to double count
        if row['amount'] == total:
            y = y_height[loop]
        else:
            y = y_height[loop] + row['amount']
        # Determine if we want a neg or pos offset
        if row['amount'] > 0:
            y += (pos_offset * 2)
            if index == total_label:
                plt.annotate(formatting.format(row['amount']), (loop, y), ha="center", color=total_color, fontsize=9)
            else:
                plt.annotate(formatting.format(row['amount']), (loop, y), ha="center", color=pos_text_color, fontsize=9)
        else:
            y -= (pos_offset * 4)
            if index == total_label:
                plt.annotate(formatting.format(row['amount']), (loop, y), ha="center", color=total_color, fontsize=9)
            else:
                plt.annotate(formatting.format(row['amount']), (loop, y), ha="center", color=neg_text_color, fontsize=9)
        loop += 1

    # Scale up the y axis so there is room for the labels
    plt.ylim(plot_min - round(3.6 * plot_offset, 7), plot_max + round(3.6 * plot_offset, 7))

    # Rotate the labels
    plt.xticks(range(0, len(trans)), trans.index, rotation=rotation_value)

    # xlim
    plt.xlim([-0.5, len(trans.index) - 0.5])

    # add zero line and title
    plt.axhline(0, color='black', linewidth=0.6, linestyle="dashed")
    plt.title(title)
    plt.tight_layout()

    return fig, ax


"""
Correlation plots
"""


def correlation_plot(correlation_matrix: np.ndarray, names: Union[List[str], None] = None, **kwargs) -> None:

    """
    Plots a correlation matrix using seaborn heatmap

    Parameters
    ----------
    correlation_matrix: ndarray
        Matrix with entries being correlations
    names: List
        List of names representing the variables names. Ordering is the same as for the correlation matrix
    **kwargs
        Other keyword-only arguments

    Returns
    -------
        None

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from corelib.plotting import correlation_plot
        data = np.random.normal(size=(10, 1000))
        corr = np.corrcoef(data)
        correlation_plot(corr)
        plt.show()

    """
    # my_params = mpl.rcParams
    # sns.set(style="white")

    mask_upper_diagonal = True
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    vmax = 1.0
    vmin = -1.0
    center = 0.0
    cbar_kws = {"shrink": .75}
    title = None
    include_diagonal = False
    include_values = False
    fmt = "d"
    size_scale = False

    if 'mask' in kwargs:
        mask_upper_diagonal = kwargs['mask']
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    if 'vmin' in kwargs:
        vmin = kwargs['vmin']
    if 'center' in kwargs:
        center = kwargs['center']
    if 'vmax' in kwargs:
        vmax = kwargs['vmax']
    if 'title' in kwargs:
        title = kwargs['title']
    if 'cbar_kws' in kwargs:
        cbar_kws = kwargs['cbar_kws']
    if 'include_diagonal' in kwargs:
        include_diagonal = kwargs['include_diagonal']
    if 'include_values' in kwargs:
        include_values = kwargs['include_values']
    if 'size_scale' in kwargs:
        size_scale = kwargs['size_scale']

    # create data frame with correlation matrix
    df_correlation = pd.DataFrame(correlation_matrix, columns=names, index=names)

    mask = None
    if mask_upper_diagonal:
        k = 0
        if include_diagonal:
            k = 1
        mask = np.triu(np.ones_like(df_correlation, dtype=bool), k=k)

#    fig, ax = plt.subplots()

    sns.heatmap(df_correlation, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                square=True, linewidths=.5, cbar_kws=cbar_kws, annot=include_values)

    # add title
    plt.title(title)

    # restore default values
    # mpl.rcParams.update(my_params)


def correlation_scatter_plot(correlation_matrix: np.ndarray, names: Union[List[str], None] = None, **kwargs) -> None:

    """
    Plots a correlation matrix using matplotlib scatterplot

    Parameters
    ----------
    correlation_matrix: ndarray
        Matrix with entries being correlations
    names: List
        List of names representing the variables names. Ordering is the same as for the correlation matrix
    **kwargs
        Other keyword-only arguments

    Returns
    -------
        None

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from corelib.plotting import correlation_scatter_plot
        data = np.random.normal(size=(10, 1000))
        corr = np.corrcoef(data)
        correlation_scatter_plot(corr)
        plt.show()

    """

    mask_upper_diagonal = True
    cmap = default_color_map
    vmax = 1.0
    vmin = -1.0
    center = 0.0
    cbar_kws = {"shrink": .75}
    title = None
    include_diagonal = False
    include_values = False
    fmt = "d"

    if 'mask' in kwargs:
        mask_upper_diagonal = kwargs['mask']
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    if 'vmin' in kwargs:
        vmin = kwargs['vmin']
    if 'center' in kwargs:
        center = kwargs['center']
    if 'vmax' in kwargs:
        vmax = kwargs['vmax']
    if 'title' in kwargs:
        title = kwargs['title']
    if 'cbar_kws' in kwargs:
        cbar_kws = kwargs['cbar_kws']
    if 'include_diagonal' in kwargs:
        include_diagonal = kwargs['include_diagonal']
    if 'include_values' in kwargs:
        include_values = kwargs['include_values']

    # the number of variables
    num_variables = correlation_matrix.shape[0]

    # variables to use for scatter plot
    x = np.repeat(np.arange(num_variables), num_variables).flatten()
    y = np.repeat(np.atleast_2d(np.arange(num_variables)), num_variables, axis=0).flatten()
    c = correlation_matrix.flatten()

    mask_idx = None
    if mask_upper_diagonal:
        k = 0
        if include_diagonal:
            k = 1

        mask_idx = np.triu(np.ones_like(correlation_matrix, dtype=np.bool), k=k).flatten()

        x = x[mask_idx]
        y = y[mask_idx]
        c = c[mask_idx]

    sc = plt.scatter(x, y, s=(np.abs(c) + 0.5) * 100,
                     c=c,
                     cmap=cmap,
                     vmin=-1,
                     vmax=1)

    # fix limits
    plt.xlim([0 - 0.5, num_variables - 1 + 0.5])
    plt.ylim([0 - 0.5, num_variables - 1 + 0.5])

    # add variables names
    if names is None:
        names = np.arange(num_variables)

    plt.xticks(np.arange(num_variables), names, rotation='vertical')
    plt.yticks(np.arange(num_variables), names, rotation='horizontal')

    # add color bar
    plt.colorbar(sc)

    # add title
    plt.title(title)
