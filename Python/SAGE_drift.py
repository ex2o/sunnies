"""
Commands used to set up the environment:

 conda create -n shapley python=3.8
 conda activate shapley
 conda install pandas
 conda install -c intel scikit-learn
 conda install -c conda-forge shap
 conda install git
 conda install pip
 pip3 install xgboost
 pip install git+git://github.com/iancovert/sage.git

"""

# %%
import sage
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import shap
from collections import namedtuple
import pickle

Results = namedtuple('Results', 'shap sage shaploss')
ResultsX = namedtuple('ResultsX', 'value X')
ResultsXy = namedtuple('ResultsXy', 'value X y')

"""
Algorithm:

1. Generate Y = X1 + X2 + (1+t/10)X3 + (1-t/10)X4 + ? with t = 0

2. Split data 50/50 train/test

3. Train simple xgboost model using the same parameters as in R

4. For each t in 0:10:

    _1. Generate data from Y = X1 + X2 + (1+t/10)X3 + (1-t/10)X4 + ?
   
    _2. Compute the SHAP, SHAPloss and SAGE values
        SHAP: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Fitting%20a%20Linear%20Simulation%20with%20XGBoost.html
        SHAPloss: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Explaining%20the%20Loss%20of%20a%20Model.html
        SAGE: https://github.com/iancovert/sage
             

5. Plot SHAPloss (e.g., average) and SAGE, versus t

    Consider:
    Use shaploss value to monitor model
    https://towardsdatascience.com/use-shap-loss-values-to-debug-monitor-your-model-83f7808af40f
    Though you may need to check the plotting functions used in SHAP
"""

# %% 1
n,d = 1000, 4
t_max = 10

def dat_t(n,d,t,t_max):
    X = np.random.normal(0,2,size=(n,d))
    T = np.array([1+t/t_max, 1-t/t_max])
    B = np.concatenate( (np.repeat(1,d-2),T) )
    y = X @ B
    return X,y

# %% 2
X_tr, y_tr = dat_t(n,d,0,t_max)
X_te, y_te = dat_t(n,d,0,t_max)

# %% 3
model = XGBRegressor()
model.fit(X_tr, y_tr)
model.score(X_te,y_te) # around 95%

# %% 4.1 define functions
def compute_sage(X, y, model):
    imputer = sage.MarginalImputer(model, X[:512])
    estimator = sage.PermutationEstimator(imputer, 'mse')
    sage_values = estimator(X, y)
    return sage_values

def compute_shap(X, model):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

def compute_shaploss(X, y, model):
    explainer = shap.TreeExplainer(
        model, X, feature_dependence="independent", model_output="logloss")
    shaploss_values = explainer.shap_values(X,y)
    return shaploss_values

# %% 4.2 compute values
shap_list, sage_list, shaploss_list = [], [], []

for t in range(0,t_max):
    X_te, y_te = dat_t(n,d,t,t_max)
    shap_list.append(ResultsX(compute_shap(X_te, model), X_te))
    sage_list.append(ResultsXy(compute_sage(X_te, y_te, model), X_te, y_te))
    shaploss_list.append(ResultsXy(compute_shaploss(X_te, y_te, model), X_te, y_te))

results = Results(shap_list, sage_list, shaploss_list)

# %% 4.3 save results
filename = 'drift_results.obj'
file = open(filename, 'wb')
pickle.dump(results, file)
file.close()

# %% Load example
filename = 'drift_results.obj'
file = open(filename, 'rb')
results = pickle.load(file)
file.close()

# %% 5.1 View mean results
t = 8
shap_res_t = np.mean(results.shap[t].value, axis=0)
shaploss_res_t = np.mean(results.shaploss[t].value, axis=0)
sage_res_t = results.sage[t].value.values
print("SHAP:    ", np.round(shap_res_t, 3))
print("SHAPloss:", np.round(shaploss_res_t, 3))
print("SAGE:    ", np.round(sage_res_t, 3))

# %% 5.2 Plot local results
shap.summary_plot(results.shap[t].value, shap_list[t].X)
shap.summary_plot(results.shaploss[t].value, shaploss_list[t].X)

# %% 5.3 plot mean results
shap_means = np.mean(results.shap[0].value, axis=0)
shaploss_means = np.mean(results.shaploss[0].value, axis=0)
sage_means = results.sage[0].value.values
for t in range(1,t_max):
    shap_means = np.vstack((shap_means, np.mean(results.shap[t].value, axis=0)))
    shaploss_means = np.vstack((shaploss_means, np.mean(results.shaploss[t].value, axis=0)))
    sage_means = np.vstack((sage_means, results.sage[t].value.values)) 


# %%
plt.plot(shap_means)

# %%
plt.plot(shaploss_means)

# %%
plt.plot(sage_means)


# %%
f = plt.figure()

features = [r"$X_1$", r"$X_2$", r"$X_3$", r"$X_4$"]
markers = ['.', 'v', 's', '*']
mcolors = ["red", "green", "blue", "purple"]
lstyles = ['--', '-', '-.']
args_plt = {'c':'black', 'alpha':0.7} 
shaploss_plt =  {'ls':lstyles[1], 'label':'SHAPloss'}
sage_plt = {'ls':lstyles[0], 'label':"SAGE"}
args_sct = {'marker':'.', 'c':'red'}
xs = list(range(t_max))



for _i in range(0,d):   
    if _i == 0:
        plt.plot(shaploss_means[:,_i], **args_plt, **shaploss_plt)
        plt.plot(sage_means[:,_i], **args_plt, **sage_plt)
        shaploss_plt.pop('label'); sage_plt.pop('label')

    args_sct['marker'] = markers[_i]
    args_sct['c'] = mcolors[_i]

    plt.plot(shaploss_means[:,_i], **args_plt, **shaploss_plt)
    plt.scatter(xs, shaploss_means[:,_i], **args_sct, label=features[_i])
    
    plt.plot(sage_means[:,_i], **args_plt, **sage_plt)
    plt.scatter(xs, sage_means[:,_i], **args_sct)

plt.legend(bbox_to_anchor=(1,1), loc="upper left")
plt.ylim([-20,15])
plt.xlabel('time')
plt.ylabel('Shapley value')
plt.show()

# %%
f.savefig("SAGE_drift_plot.pdf", bbox_inches = 'tight')
