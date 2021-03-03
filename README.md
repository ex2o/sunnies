# Sunnies :sunglasses:

**S**hapley values that **u**ncover **n**onli**n**ear dependenc**ies**

Herein lie code and results for the paper [Explaining the data or explaining a model? Shapley values that uncover non-linear dependencies](https://arxiv.org/abs/2007.06011) by [Daniel Fryer](https://danielvfryer.com), [Inga Strumke](https://strumke.com) and [Hien Nguyen](https://hiendn.github.io).

## Guide

-   For the **R script** that generates the majority of the results, see [here](R/All_scripts_for_paper.R). The other R files are dependencies of that script.
-   For the **Python script** that produces SAGE and SHAP results for the dataset drift example, see [here](Python/SAGE_drift.py).
-   The `shapley` function [here](R/Shapley_helpers.R) can be used to calculate Shapley values given a data set and utility function (e.g., choosing a measure of non-linear dependence as the utility function will produce Sunnies values). Some utility functions can be found [here](R/utility_functions.R).
-   Run the **Python script** [here](Python/make_paper_figure.py) to generate the violin plot in Figure 2.
-   Other Python files in the Python directory are dependencies of the violin plot script, and can also be used to calculate Shapley and Sunnies values for any data generating process or data set.

## Resources

* shapr https://norskregnesentral.github.io/shapr/articles/understanding_shapr.html#advanced
* SHAP Python explaining loss https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Explaining%20the%20Loss%20of%20a%20Model.html
* SHAP Python tree based models https://shap.readthedocs.io/en/latest/tabular_examples.html#tree-based-models