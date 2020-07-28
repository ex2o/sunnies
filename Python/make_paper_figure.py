import os
import numpy
import matplotlib.pyplot as plt

# --- Our stuff
from plot import violinplot
from shapley import calc_n_shapley_values

CF_DICT = {
        "r2" : r"$R^2$",
        "dcor" : "Distance correlation",
        "aidc" : "Affine invariant dist. corr",
        "hsic" : "Hilbert-Schmidt indep.cr.",
        "xgb" : "XGBoost Regressor",
        }

def normalise(x):
    return (x - numpy.mean(x))/(numpy.std(x))

def make_paper_violin_plot():
    """
    Create violin plot as it appears in the paper
    """

    n_samples = 1000
    n_feats = 5
    n_iter = 1000
    players = list(range(n_feats))
    data_type = "random"
    data_dir = os.path.join("result_data", "{0}".format(data_type))
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    # --- Plot shapley decompositions for all cfs per player
    cfs = ["r2", "hsic", "dcor", "aidc"]
    all_cf_shaps_per_player = []
    for _cf in cfs:
        _all_player_shaps = []
        _cf_shaps = calc_n_shapley_values(n_feats, n_samples, n_iter,
                data_type, _cf, overwrite=False, data_dir=data_dir)

        # --- Group shapley decompositions per player. Normalised.
        all_cf_shaps_per_player.append([normalise(numpy.array(_cf_shaps))[:,_player] for _player in players])
        print("Done with {0}.".format(_cf))
    # ---

    cf_labels = [CF_DICT.get(_cf, 0) for _cf in cfs]
    violinplot(all_cf_shaps_per_player, players, labels=cf_labels, multi=True)
    plt.gca().grid(False)
    plt.savefig('violin_demo.png', transparent=True)
    plt.show()


if __name__ == "__main__":

    make_paper_violin_plot()
