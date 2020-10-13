# Sunnies :sunglasses:
**S**hapley values that **u**ncover **n**onli**n**ear dependenc**ies**

Herein lie code and results for the paper [Explaining the data or explaining a model? Shapley values that uncover non-linear dependencies](https://arxiv.org/abs/2007.06011) by [Daniel Fryer](https://danielvfryer.com), [Inga Strumke](https://strumke.com) and [Hien Nguyen](https://hiendn.github.io). 

## Guide
* For the **R script** that generates all results other than the violin plot, see [here](R/All_scripts_for_paper.R). The other R files are dependencies of that script. All simulated data are generated in that script, while non-simulated data is in the folder [RL_data](R/RL_data).
* The `shapley` function [here](R/Shapley_helpers.R) can be used to calculate Shapley values given a data set and utility function (e.g., choosing a measure of non-linear dependence as the utility function will produce Sunnies values). Some utility functions can be found [here](R/utility_functions.R).
* Run the **Python script** [here](Python/make_paper_figure.py) to generate the violin plot in Figure 2. 
* Other Python files in the Python directory are dependencies of the violin plot script, and can also be used to calculate Shapley and Sunnies values for any data generating process or data set.
