source("utility_functions.R")
source("shapley_helpers.R")


### This function is very similar to shapley_sim1_N
# but it just calculates the utility of each feature as singleton.
singletons_sim_N <- function(n, d, N, data_gen, overwrite = F,
                                   loc_only = F, save = T, ...) {
  # Directory and file string part
  fun_name <- as.character(substitute(data_gen))
  dir <- "results/"
  loc <- paste0(dir, "singletons_sim_n",n,"_N",N,"_d",d,"_", fun_name,".Rds")
  if (loc_only) return(loc)
  if (save && !overwrite && file.exists(loc)) {
    stop("file exists, perhaps choose overwrite = T")}
  # Simulation part
  results <- list()
  utilities <- c("R2","DC","BCDC","AIDC","HSIC")
  for (i in 1:length(utilities)) {
    results[[utilities[i]]] <- shapley_sim1(
      get(utilities[i]), N, n , d, data_gen, ...)
  }
  # Save and return part
  if (save) {
    saveRDS(results, loc)
    return(loc)
  } else {
    return(results) 
  }
}

# helper for singletons_sim_N
singletons_sim <- function(utility, N, n, d, data_gen, ...) {
  results <- matrix(0, nrow = N, ncol = d)
  for ( i in 1:N ) {
    dat <- data_gen(d, n, ...)
    y <- dat[,1, drop=F]; X <- dat[,-1, drop=F]
    results[i,] <- singleton_utilities(y, X, utility)
  }
  results
}

# Helper for singletons_sim_N
singleton_utilities <- function(y, X, utility) {
  apply(X, FUN = function(xj){utility(y,xj)}, MARGIN = 2)
}

 
### Parameter names
# n: sample size (used 1000 for saved data)
# d: number of features (used 4 for saved data)
# N: number of iterations simulating data (used 1000 for saved data)
# data_gen: the data generating function
# overwrite: whether to overwrite file if it exists (TRUE). 
#            Ignored if loc_only = T or save = F
# loc_only: whether to only return the generated 
#           file name and not to run sim (TRUE)
# save: whether to save the results (TRUE) or just return the results (FALSE)
# ...: arguments passed to data_gen
#
#### Calls shapley_sim1 N times with each dependence measure and
# saves the results with a standard naming convention that captures
# the parameter values.
# Returns a character vector representing the saved file location.
shapley_sim1_N <- function(n, d, N, data_gen, overwrite = F, 
                           loc_only = F, save = T, ...) {
  # Directory and file string part
  fun_name <- as.character(substitute(data_gen))
  dir <- "results/"
  loc <- paste0(dir, "sim1_n",n,"_N",N,"_d",d,"_", fun_name,".Rds")
  if (loc_only) return(loc)
  if (save && !overwrite && file.exists(loc)) {
    stop("file exists, perhaps choose overwrite = T")}
  # Simulation part
  results <- list()
  utilities <- c("R2","DC","BCDC","AIDC","HSIC")
  for (i in 1:length(utilities)) {
    results[[utilities[i]]] <- shapley_sim1(
      get(utilities[i]), N, n , d, data_gen, ...)
  }
  # Save and return part
  if (save) {
    saveRDS(results, loc)
    return(loc)
  } else {
    return(results) 
  }
}


### shapley_sim1 calls the shapley function for N
# samples of the simulated data given by data_gen
## Parameter names
# n:        sample size
# d:        number of features
# N:        number of samples
# data_gen: a data generating function (see simulated_datasets.R)
# ... :     arguments passed to data_gen
shapley_sim1 <- function(utility, N, n, d, data_gen, ...) {
  results <- matrix(0, nrow = N, ncol = d)
  for ( i in 1:N ) {
    dat <- data_gen(d, n, ...)
    y <- dat[,1, drop=F]; X <- dat[,-1, drop=F]
    CF_i <- estimate_CF(X, utility, y = y)
    results[i,] <- shapley(CF = CF_i)
  }
  results
}