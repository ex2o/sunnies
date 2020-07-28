"""
Helper functions for shalpey.py
Package dependencies:
pip install dcor (from pypi.org/project/dcor/)
"""

import sys
from itertools import combinations
import dcor
import numpy
import scipy
from HSIC import dHSIC
from xgb_regressor import make_xgb_dict

def AIDC(x, y):
    cov_y = numpy.cov(y)
    cov_x = numpy.cov(x.T)

    if cov_x.shape is ():
        inv_cov_x = 1.0/cov_x
        x_trans = numpy.dot(x, numpy.sqrt(inv_cov_x))
    else:
        inv_cov_x = numpy.linalg.inv(cov_x)
        x_trans = numpy.dot(x, scipy.linalg.sqrtm(inv_cov_x))

    inv_cov_y = 1/cov_y
    y_trans = numpy.dot(y, numpy.sqrt(inv_cov_y))
    return dcor.distance_correlation(y_trans, x_trans)

def CF(x, y, team, cf_name):
    """
    Available characteristic functions:
        dcor: Distance correlation between y and x
    """
    x = x[:, team]

    if len(team)==0:
        return 0.0

    if cf_name.lower() == "dcor":
        return dcor.distance_correlation(y, x)

    elif cf_name.lower() == "r2":
        det_C_xy = numpy.linalg.det(numpy.corrcoef(x.T, y))
        if len(team)==1:
            det_C_x = 1
        else:
            det_C_x = numpy.linalg.det(numpy.corrcoef(x.T))

        # ------------------------------------
        # FOr debugging R2 in Julia
        #print(f"team={team}")
        #print(1 - det_C_xy/det_C_x)
        # ------------------------------------

        return (1 - det_C_xy/det_C_x)

    elif cf_name.lower() == "aidc":
        return dcor.distance_correlation_af_inv(y, x)
        #return AIDC(x, y)

    elif cf_name.lower() == "hsic":
        return dHSIC(x, y)

    else:
        raise NameError("I don't know the characteristic function {0}".format(cf_name))
        return 0

def make_cf_dict(x, y, players, cf_name):
    """
    Creates dictionary with values of the characteristic function for each
    combination of the players.
    """
    cf_dict = {}
    num_players = len(players)
    team_sizes = list(range(num_players+1))

    if cf_name is "xgb":
        return make_xgb_dict(x, y)

    for _size in team_sizes:
        value_s = 0
        teams_of_size_s = list(combinations(players, _size)) #NB: returns tuples
        for _team in teams_of_size_s:
            cf_dict[_team] = CF(x, y, _team, cf_name)

    return cf_dict

def calc_shap(x, y, v, cf_dict):
    """
    Calculate the Shapley value for player indexed v,
    given features x and labels/targets y, using as
    caracteristic function the pre-computed values in cf_dict.
    """
    players = list(range(x.shape[1]))

    if v in players:
        players.remove(v)

    num_players = len(players)
    team_sizes = list(range(num_players+1))
    value = 0
    v_tuple = (v,)

    for _size in team_sizes:
        value_s = 0
        teams_of_size_s = list(combinations(players, _size))
        for _team in teams_of_size_s:
            #value_in_team = cf(x, y, _team + v_tuple) - cf(x, y, _team)
            value_in_team = (cf_dict[tuple(sorted(_team+v_tuple))] - cf_dict[_team])

            #this sometimes gets negative when using cf=r^2
            #print(value_in_team)
            value_s += value_in_team
        average_value_s = value_s/len(teams_of_size_s)
        value += average_value_s
    average_value = value/len(team_sizes)

    return average_value
