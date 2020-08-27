library(devtools)
## Uncomment to install shapr
# devtools::install_github("NorskRegnesentral/shapr")
library(shapr)
library(SHAPforxgboost)
library(speedglm) # speedlm
library(Rfast) # dcor bcdcor
library(dHSIC) # dHSIC
library(expm) # sqrtm
library(xgboost)
library(dplyr)
library(tidyr)
library(ggplot2)
library(latex2exp)
library(reticulate)
library(reshape2)
library(naniar)
source("simulated_datasets.R")
source("shapley_helpers.R")
source("utility_functions.R")
source("comparison_helpers.R")
source("Shapley_helpers.R")
source("Applications_helpers.R")


### R2 LINEAR EXAMPLE -------------------------------------------------------
n <- 1e4
dat <- dat_unif_squared(n = n) %>% 
  data.frame()
N <- 100
k <- 1000
coeffs <- matrix(NA, nrow = N, ncol = 2)
R2 <- vector(mode = "numeric", length = N)
for (i in 1:N) {
  s <- sample(1:n, k)
  model <- dat[s,] %>% 
    data.frame() %>% 
    lm(y ~ x1+x2+x3+x4, data = .) %>% 
    summary()
  R2[i] <- model$r.sq
  coeffs[i,] <- model$coefficients[c(1,5),1]
}
coeffs <- data.frame(coeffs)
names(coeffs) <- c("intercept", "slope")
mean_coeffs <- as.data.frame(t(apply(coeffs, MARGIN = 2, FUN = mean)))
#pdf(file="R2_plot.pdf", width=5, height=4)
ggplot(dat) +
  geom_point(aes(y=y,x=x4), alpha = 0.05, shape = 8) +
  theme_minimal() +
  geom_abline(data=coeffs, aes(slope=slope, intercept=intercept), alpha=0.05,
              colour="darkred", size=2) +
  xlab(TeX("$X_4")) +
  geom_abline(data = mean_coeffs, 
              aes(slope=slope, intercept=intercept),
              alpha=0.5, colour="indianred", size=0.5)
#dev.off()
quantile(R2, probs = c(0.025, 0.975))
mean(R2)


### LDA Example 1 -----------------------------------------------------------
# The repetitions producing violin plots is in python, but here is similar:
result1 <- run_evaluations(dat_unif_squared, utility = DC, n = 1e3, plots = T)

### LDA Examples 2 and 3  ----------------------------------------------------
# (which should be the same example)
# This example should be done exactly by hand, but here is a simulation.
# When it is calculated exactly, the shapley values will be equal to each other:
result2 <- run_evaluations(dat_catcat_XOR, utility = DC, n = 1e3, plots = T)

### EXAMPLE 1 -----------------------------------------------------------
# One feature becomes more important, one less important.
n <- 1e4; m <- 10; d <- 4; N <- 100; s <- 1000
shaps_lab <- matrix(0, nrow = m+1, ncol = d) 
shaps_res <- matrix(0, nrow = m+1, ncol = d)
mse <- vector(mode = "numeric", length = m+1)
datt <- dat_t(n, d, t = 0, max_t = m)
sdatt <- split_dat(datt)
xgb <- basic_xgb_fit(sdatt) # Model is only fit once
cat("Number of features: ", xgb$nfeatures)
cdN_all <- array(NA, dim = c(6,d,N,m))
for (t in 0:m) {
  print(paste0("round ",t))
  xgbtt <- basic_xgb_test(xgb, sdatt) # Model is tested each time
  mse[t+1] <- xgbtt$test_mse
  cdNt <- compare_DARRP_N(sdatt, xgbtt, features = 1:4, 
                         feature_names = paste0("x",1:4),
                         sample_size = s, N = N,
                         valid = F, all_labs = F)
  cdN_all[,,,t] <- cdNt
  datt <- dat_t(n, d, t = t, max_t = m)
  sdatt <- split_dat(datt)
}

cdN_all <- readRDS("results/run1_cdN_drift.Rds")
mse <- readRDS("results/run1_cdN_drift_mse.Rds")
pdf(file="Drift_ADL_ADR.pdf",width=5,height=4)
plot_compare_DARRP_N_drift(cdN_all, shap_index = c(1,5))
dev.off()
# ------------
pdf(file="Drift_ADL_ADR_lines.pdf",width=5,height=4)
plot_compare_DARRP_N_drift2(cdN_all, shap_index = c(1,5))
dev.off()


### EXAMPLE 2 -----------------------------------------------------------
n <- 1e4; d <- 5
X <- matrix(rnorm(n*(d-2),0,1), nrow = n, ncol = (d-2))
X4 <- sample(0:1, replace = T, n)
X5 <- sample(0:1, replace = T, n)
y <- rowSums(X[,-(d-2)]) + 5*(X4 & X5)*X[,(d-2)] + rnorm(n,0,0.1)
dat <- cbind(y,X,X4,X5)
sdat <- split_dat(dat, df = T)
lmodel <- lm(y ~ x1 + x2 + x3 + x4 + x5, dat = sdat$df_yx_train)
lmodelt <- basic_lmodel_test(lmodel, sdat)

cdN <- readRDS("results/run1_cdN_linear1.Rds")
plot_compare_DARRP_N(cdN, main = "test run", all_labs = F)
lmodel2 <- lm(y ~ x1 + x2 + x3 + x4 + x5 + x3:x4:x5, dat = sdat$df_yx_train)
lmodel2t <- basic_lmodel_test(lmodel2, sdat)

colpal <- c("#CC79A7", "#0072B2", "#D55E00")
cdN2 <- readRDS("results/run1_cdN_linear2.Rds")
pdf(file="DARRP_interact.pdf",width=5,height=4)
plot_compare_DARRP_N_interact_ADL_ADP(cdN)
dev.off()
pdf(file="DARRP_interact2.pdf",width=5,height=4)
plot_compare_DARRP_N_interact_ADL_ADP(cdN2)
dev.off()
pdf(file="DARRP_interact_all.pdf",width=5,height=4)
plot_compare_DARRP_N_interact_all(cdN, colpal=colpal)
dev.off()
pdf(file="DARRP_interact_all2.pdf",width=5,height=4)
plot_compare_DARRP_N_interact_all(cdN2, colpal=colpal)
dev.off()


### Application 1 -----------------------------------------------------------
# The NHANES I dataset from [14]
Xh <- read.csv("RL_data/X_data_with_header.csv")
y <- read.csv("RL_data/y_data.csv", header = F)
names(y) <- "logRR"
Xh <- apply(Xh, FUN = function(x){x[is.nan(x)] <- NA; x}, MARGIN = 2)
Xh <- as_tibble(Xh)
missv <- miss_var_summary(Xh)
high_miss <- missv[1:15,][[1]]; high_miss
Xh2 <- select(Xh, -one_of(high_miss))
X_dr <- remove_all_missing(Xh2, ncols = 3)
y_dr <- y[attr(X_dr, "keep"),]
dat <- cbind(y_dr, X_dr)
interesting <- c("age", "physical_activity", "systolic_blood_pressure")
#interesting2 <- c("cholesterol", "bmi", "systolic_blood_pressure")
#interesting3 <- c("cholesterol", "bmi", "physical_activity", "systolic_blood_pressure", "age")
fts <- which(colnames(dat) %in% interesting) - 1
fnams <- c("age", "PA", "SBP")
sdat4way <- split_dat_gender_4way(dat)
xgb <- nhanes_xgb_fit(sdat4way, nround=5000)
cdN4way <- compare_DARRRP_N_gender_4way(
  sdat4way, xgb, sample_size = 1000, N = 100,
  features = fts, feature_names = fnams)
#saveRDS(cdN4way3, "run1_cdN4way3.Rds")
#cdN4way <- readRDS("results/run1_cdN4way.Rds")
pdf(file="DARRP_4way.pdf",width=5,height=4)
plot_compare_DARRP_N_4way(cdN4way$cdN)
dev.off()
