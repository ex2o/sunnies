"""
Shapley values with various model-agnostic measures of dependence as
utility functions.
"""

import numpy
import os

# --- Our stuff
import synthetic_data as data
import shapley_helpers as sh


def calc_shapley_values(x, y, cf_name="dcor"):
    """
    Returns the shapley values for features x and labels y, given a
    characteristic function (default dcor)
    """
    players = list(range(x.shape[1]))
    shapley_values = []
    cf_dict = sh.make_cf_dict(x, y, players, cf_name=cf_name)
    for _player in players:
        shapley_values.append(sh.calc_shap(x, y, _player, cf_dict))
    return shapley_values

def calc_n_shapley_values(n_feats, n_samples, n_iter, data_type, cf_name, overwrite=False, data_dir="result_data"):
    """
    Returns a nested list of shapley values (per player) per iteration;
    [[v1... vn], [v1...vn], [v1...vn], ...]
    I.e. the length of the list is equal to n_iter
    """
    players = list(range(n_feats))

    filename = f"{data_dir}/{n_feats}_feats_{n_samples}_samples_{n_iter}_iter_{cf_name}.npy"
    if not overwrite and os.path.exists(filename):
        return numpy.load(filename)

    all_shaps = []
    for _i in range(n_iter):
        x, y = data.make_data(n_feats, n_samples, data_type)

        _shapley_values = calc_shapley_values(x, y, cf_name)
        all_shaps.append(_shapley_values)

    numpy.save(filename, all_shaps)

    return all_shaps
