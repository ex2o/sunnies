import matplotlib.pyplot as plt
import scipy.linalg
import numpy

COLORS = ["orange", "blue", "green", "purple"]

def nice_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#EEEEEE')
    ax.xaxis.grid(False)
    return ax

def violinplot(values, positions, labels=None, multi=True):
    if not multi:
        assert len(values)==len(positions)
    # --- Axis formatting
    _, ax = plt.subplots()
    ax = nice_axes(ax)

    if multi:
        new_positions = [3*_p for _p in positions] # More x-axis space
        widths = [0.3 for _ in positions]

        for _n, _values in enumerate(values):
            _col = COLORS[_n]
            _positions = [1+_p + 0.4*_n for _p in new_positions]
            _vplot = ax.violinplot(_values,
                        positions=_positions,
                        widths=widths,
                        #quantiles=[[0.05, 0.95] for _ in range(5)],
                        showextrema=False,
                        showmeans=True,
                        )

            if labels is not None:
                ax.plot([0], linestyle='-', label=labels[_n], c=_col)

            _vplot["cmeans"].set_color("black")
            for _pc in _vplot['bodies']:
                _pc.set_facecolor(_col)
                _pc.set_edgecolor("black")
                _pc.set_alpha(0.8)

        ax.set_xticks([_p+1.5 for _p in new_positions])
        ax.set_xticklabels([str(_p+1) for _p in positions])
        plt.legend(loc="lower right")
        #plt.legend(loc="upper right")
        plt.ylim([-2, 2])
        plt.ylabel("Normalised Shapley value")

    else:
        ax.violinplot(values, positions=positions)
        plt.ylabel("Shapley value")

    plt.xlabel("Feature")

def boxplot(values, positions, labels=None, multi=True):
    """
    If values is a list of numbers, you'll get a plot containing one box.
    If values contains lists of lists of numbers, you'll get a plot containing
    one box per list.
    If values is a list of lists which contain numbers and multi=True, you'll
    get a plot containing one box per sublist and list.
    """
    if not multi:
        assert len(values)==len(positions)
    # --- Axis formatting
    _, ax = plt.subplots()
    ax = nice_axes(ax)

    if multi:
        new_positions = [2*_p for _p in positions] # More x-axis space
        widths = [0.1 for _ in positions]

        for _n, _values in enumerate(values):
            _col = COLORS[_n]
            _positions = [1+_p + 0.2*_n for _p in new_positions]
            _bplot = ax.boxplot(_values,
                        positions=_positions,
                        showfliers=False,
                        widths=widths,
                        patch_artist=True,
                        medianprops=dict(color="black"),
                        )

            if labels is not None:
                ax.plot([0], linestyle='-', label=labels[_n], c=_col)

            [_bplot["boxes"][_p].set_facecolor(_col) for _p in positions]
        ax.set_xticks([_p+1.5 for _p in new_positions])
        ax.set_xticklabels([str(_p+1) for _p in positions])
        plt.legend()
        plt.ylabel("Normalised Shapley value")

    else:
        ax.boxplot(values, positions=positions, showfliers=False)
        plt.ylabel("Shapley value")

    plt.xlabel("Feature")
    plt.draw()

def barplot_all(xs, values):
    _, ax = plt.subplots()
    ax = nice_axes(ax)

    plt.bar(xs, values)
    plt.draw()

def least_squares_plot(x_data, y_data):
    _, ax = plt.subplots()
    ax = nice_axes(ax)

    A = numpy.vstack([x_data, numpy.ones(len(x_data))]).T
    solution = numpy.linalg.lstsq(A, y_data)
    m, c = solution[0]
    resid = solution[1][0]

    r2 = 1 - resid / (y_data.size * y_data.var())

    x = numpy.linspace(-1, 1, 100)
    plt.plot(x, m*x+c, 'r', label=r"$R^2=${0}".format(round(r2, 4)))
    plt.scatter(x_data, y_data, alpha=0.8)
    plt.xlabel(r"$X_4$", fontsize=16)
    plt.ylabel(r"$Y$", fontsize=16)
    plt.legend(fontsize=16)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.show()

def plot_data(X, Y):
    _, ax = plt.subplots()
    ax = nice_axes(ax)

    if len(X.shape) == 1:
        plt.scatter(X, Y, alpha=0.5)
    else:
        for _i in range(X.shape[1]):
            plt.scatter(X[:, _i], Y, label="X{0}".format(_i), alpha=0.5)
        plt.legend(loc="upper right")

    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.draw()

