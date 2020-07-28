source("utility_functions.R")
source("shapley_helpers.R")
source("simulated_datasets.R")

#devtools::install_github("NorskRegnesentral/shapr")
library(shapr)
library(xgboost)
library(SHAPforxgboost)

split_dat <- function(dat, df = F) {
  n <- nrow(dat)
  train <- sample(1:n, floor(n/2))
  x_train <- dat[train,-1]
  colnames(x_train) <- paste0("x",1:ncol(x_train))
  x_test <- dat[-train,-1]
  colnames(x_test) <- paste0("x",1:ncol(x_train))
  y_train <- dat[train,1, drop = F]
  colnames(y_train) <- "y"
  y_test <- dat[-train, 1, drop = F]
  colnames(y_train) <- "y"
  if (df) {
    df_yx_train <- data.frame(y = y_train, x_train) 
    df_yx_test <- data.frame(y = y_test, x_test)
    return(list(dat = dat, 
                x_train = x_train,
                y_train = y_train,
                x_test = x_test,
                y_test = y_test,
                df_yx_train = df_yx_train,
                df_yx_test = df_yx_test))
  }
  return(list(dat = dat, 
              x_train = x_train,
              y_train = y_train,
              x_test = x_test,
              y_test = y_test))
}

# Splits the data proportionally using gender
split_dat_gender <- function(dat, p1f, p2f) {
  n <- nrow(dat)
  male_index <- (dat[["sex_isFemale"]] == F)
  nm <- sum(male_index)
  n1mf <- calc_n1m_n1f(n, nm, p1f, p2f)
  X_m <- dat[male_index,-1]
  X_f <- dat[!male_index,-1]
  y_m <- dat[male_index,1]
  y_f <- dat[!male_index,1]
  if (n1mf$n1f == 0) {
    x_train <- X_m
    x_test <- X_f
    y_train <- y_m
    y_test <- y_f
  } else if (n1mf$n1m == 0) {
    x_train <- X_f
    x_test <- X_m
    y_train <- y_f
    y_test <- y_m
  } else {
    train_m <- sample(1:nrow(X_m), n1mf$n1m)
    train_f <- sample(1:nrow(X_f), n1mf$n1f)
    x_train <- rbind(X_m[train_m,], X_f[train_f,]) 
    x_test <- rbind(X_m[-train_m,], X_f[-train_f,])
    y_train <- c(y_m[train_m], y_f[train_f])
    y_test <- c(y_m[-train_m], y_f[-train_f])
  }
  return(list(y_train = as.matrix(y_train), 
              y_test = as.matrix(y_test), 
              x_train = as.matrix(x_train), 
              x_test = as.matrix(x_test)))
}

split_dat_gender_3way <- function(dat) {
  n <- nrow(dat)
  male_index <- which(dat[["sex_isFemale"]] == F)
  nm <- length(male_index)
  X_m <- dat[male_index,-1]
  X_f <- dat[-male_index,-1]
  y_m <- dat[male_index,1]
  y_f <- dat[-male_index,1]
  
  train_m <- sample(1:nrow(X_m), as.integer(nm/2))
  train_f <- sample(1:nrow(X_f), as.integer(nm/2))
  x_train <- rbind(X_m[train_m,], X_f[train_f,])
  x_test <- X_m[-train_m,]
  x_valid <- X_f[-train_f,]
  y_train <- c(y_m[train_m], y_f[train_f])
  y_test <- y_m[-train_m]
  y_valid <- y_f[-train_f]
  return(list(y_train = as.matrix(y_train), 
              y_test = as.matrix(y_test),
              y_valid = as.matrix(y_valid),
              x_train = as.matrix(x_train), 
              x_test = as.matrix(x_test),
              x_valid = as.matrix(x_valid)))
}

# split_dat_gender_4way <- function(dat) {
#   n <- nrow(dat)
#   male_index <- which(dat[["sex_isFemale"]] == F)
#   nm <- length(male_index)
#   X_m <- dat[male_index,-1]
#   X_f <- dat[-male_index,-1]
#   y_m <- dat[male_index,1]
#   y_f <- dat[-male_index,1]
#   
#   train_m <- sample(1:nrow(X_m), as.integer(nm/2))
#   train_f <- sample(1:nrow(X_f), as.integer(nm/2))
#   x_train <- rbind(X_m[train_m,], X_f[train_f,])
#   x_test <- X_m[-train_m,]
#   x_valid <- X_f[-train_f,]
#   y_train <- c(y_m[train_m], y_f[train_f])
#   y_test <- y_m[-train_m]
#   y_valid <- y_f[-train_f]
#   return(list(y_train = as.matrix(y_train), 
#               y_test = as.matrix(y_test),
#               y_valid = as.matrix(y_valid),
#               x_train = as.matrix(x_train), 
#               x_test = as.matrix(x_test),
#               x_valid = as.matrix(x_valid)))
# }

# Calculates number of males n1m in the training set, 
# and number of females n1f in the training set, 
# where it is assumed that we do not want to discard anybody.
# n := number of people overall
# nm := number of males overall
# p1f := proportion of females to males in the training set
# p2f := proportion of females to males in the test set
calc_n1m_n1f <- function(n, nm, p1f, p2f) {
  K1 <- p1f/(1-p1f)
  K2 <- p2f/(1-p2f)
  if (K1 == K2) {return(list(n1m = as.integer(nm/2), 
                             n1f = as.integer((n-nm)/2)))}
  if (p1f == 0) {return(list(n1m = nm, n1f = 0))}
  if (p1f == 1) {return(list(n1m = 0, n1f = n - nm))}
  n1m <- (n - nm*(K2+1))/(K1-K2)
  list(n1m = as.integer(n1m), n1f = as.integer(K1*n1m))
}
#n <- 14264
#nm <- 5765
#p1f <- 0.5
#p2f <- 0.9
#calc_n1m(n, nm, p1f, p2f)

# If we want to increase size of training set we need to drop females.
# Calculate number of discarded females df as a function of
# number of people in the training set, n1
# where the total number of people n and males nm are fixed:
# n1 is number of people in the training set.
# n is total number of people
# nm is total number of males
# p1f is the proportion of females in the training set
# p2f is the proportion of females in the test set
# df is the number of discarded females
n1_calc_df <- function(n1, p1f, p2f, n, nm) {
  K1 <- p1f/(1-p1f)
  K2 <- p2f/(1-p2f)
  df <- n - n1*(K1-K2)/(K1+1) - nm*(K2+1)
  return(df*(df > 0))
}
#n1 <- 10000:14000; n <- 10000
#plot(n1, n1_calc_df(n1, 0.5, 0.9, n, 5765), type = 'l')


compare_label_shapleys <- function(sdat, features, 
                                   feature_names,
                                   legend = c("train","test")) {
  s1 <- shapley(sdat$y_train, sdat$x_train[,features], utility = DC)
  s2 <- shapley(sdat$y_test, sdat$x_test[,features], utility = DC)
  s <- rbind(s1, s2)
  colnames(s) <- feature_names
  barplot(s,
          xlab = "Feature", ylab = "Attribution",
          col = c("black","gray"), beside = T)
  legend(x = "topright", legend = legend, 
         col = c("black","gray"), pch = c(15,15))
}

compare_DARRP <- function(sdat, modelt, features, 
                          feature_names, 
                          utility = DC,
                          sample_size = 1e3,
                          valid = F, all_labs = T) {
  
  if (valid & is.null(modelt$pred_valid)) {stop(paste0("Make sure",
    "modelt holds validation set results when valid = TRUE"))}
  samp_train <- 1:nrow(sdat$x_train)
  samp_test <- 1:nrow(sdat$x_test)
  samp_valid <- if(valid) {1:nrow(sdat$x_valid)} else {NULL}
  if (!is.na(sample_size)) {
    samp_train <- sample(samp_train, sample_size)
    samp_test <- sample(samp_test, sample_size)
    samp_valid <- if(valid) {sample(samp_valid, sample_size)} else {NULL}
  }
  X1 <- sdat$x_test[samp_test,features,drop=F]
  X2 <- sdat$x_train[samp_train,features,drop=F]
  X3 <- if(valid) {sdat$x_valid[samp_valid,features,drop=F]} else {NULL}
  y1 <- as.matrix(sdat$y_test)[samp_test,,drop = F]
  y2 <- as.matrix(sdat$y_train)[samp_train,,drop = F]
  y3 <- if(valid) {as.matrix(sdat$y_valid)[samp_valid,,drop = F]} else {NULL}
  p1 <- as.matrix(modelt$pred_test)[samp_test,,drop = F]
  p2 <- as.matrix(modelt$pred_train)[samp_train,,drop = F]
  p3 <- if(valid) {as.matrix(modelt$pred_valid)[samp_valid,,drop = F]} else {NULL}
  r1 <- as.matrix(modelt$residuals_test)[samp_test,,drop = F]
  r2 <- as.matrix(modelt$residuals_train)[samp_train,,drop = F]
  r3 <- if(valid) {as.matrix(modelt$residuals_valid)[samp_valid,,drop = F]} else {NULL}
  y <- rbind(y1,y2,y3)
  X <- rbind(X1,X2,X3)
  
  s0 <- if(all_labs) {shapley(y, X, utility = utility)} else {NULL}
  s1 <- shapley(y1, X1, utility = utility)
  s2 <- shapley(y2, X2, utility = utility)
  s3 <- if (valid) {shapley(y3, X3, utility = utility)} else {NULL}
  s4 <- shapley(p1, X1, utility = utility)
  s5 <- shapley(p2, X2, utility = utility)
  s6 <- if (valid) {shapley(p3, X3, utility = utility)} else {NULL}
  s7 <- shapley(r1, X1, utility = utility)
  s8 <- shapley(r2, X2, utility = utility)
  s9 <- if (valid) {shapley(r3, X3, utility = utility)} else {NULL}
  s <- rbind(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9)
  colnames(s) <- feature_names

  return(s)
}

compare_DARRP2 <- function(sdat4way, modelt, features, 
                          feature_names, 
                          utility = DC,
                          sample_size = 1e3) {
  
  samp_test <- 1:nrow(sdat4way$x_test)
  samp_males <- 1:nrow(sdat4way$x_males)
  samp_females <- 1:nrow(sdat4way$x_females)
  if (!is.na(sample_size)) {
    samp_test <- sample(samp_test, sample_size)
    samp_males <- sample(samp_males, sample_size)
    samp_females <- sample(samp_females, sample_size)
  }
  X1 <- sdat4way$x_test[samp_test,features,drop=F]
  X2 <- sdat4way$x_males[samp_males,features,drop=F]
  X3 <- sdat4way$x_females[samp_females,features,drop=F]
  y1 <- as.matrix(sdat4way$y_test)[samp_test,,drop = F]
  y2 <- as.matrix(sdat4way$y_males)[samp_males,,drop = F]
  y3 <- as.matrix(sdat4way$y_females)[samp_females,,drop = F]
  p1 <- as.matrix(modelt$pred_test)[samp_test,,drop = F]
  p2 <- as.matrix(modelt$pred_males)[samp_males,,drop = F]
  p3 <- as.matrix(modelt$pred_females)[samp_females,,drop = F]
  r1 <- as.matrix(modelt$residuals_test)[samp_test,,drop = F]
  r2 <- as.matrix(modelt$residuals_males)[samp_males,,drop = F]
  r3 <- as.matrix(modelt$residuals_females)[samp_females,,drop = F]
  
  s1 <- shapley(y1, X1, utility = utility)
  s2 <- shapley(y2, X2, utility = utility)
  s3 <- shapley(y3, X3, utility = utility)
  s4 <- shapley(p1, X1, utility = utility)
  s5 <- shapley(p2, X2, utility = utility)
  s6 <- shapley(p3, X3, utility = utility)
  s7 <- shapley(r1, X1, utility = utility)
  s8 <- shapley(r2, X2, utility = utility)
  s9 <- shapley(r3, X3, utility = utility)
  s <- rbind(s1,s2,s3,s4,s5,s6,s7,s8,s9)
  colnames(s) <- feature_names
  
  return(s)
}

compare_DARRP_N <- function(sdat, modelt, features, 
                            feature_names, 
                            utility = DC,
                            sample_size = 1e3, N = 100,
                            valid = F, all_labs = T) {
  d <- length(features)
  cd <- array(dim = c(6 + valid*3 + all_labs, d, N))
  for (i in 1:N) {
    cdi <- compare_DARRP(sdat, modelt, features, 
                         feature_names, utility = DC,
                         sample_size = sample_size,
                         valid = valid, all_labs = all_labs)
    cd[,,i] <- cdi
    
  }
  dimnames(cd) <- list(NULL, feature_names, NULL)
  return(cd)
}

compare_DARRP_N2 <- function(sdat4way, modelt, features, 
                            feature_names, 
                            utility = DC,
                            sample_size = 1e3, 
                            N = 100) {
  d <- length(features)
  cd <- array(dim = c(9, d, N))
  for (i in 1:N) {
    cdi <- compare_DARRP2(sdat4way, modelt, features, 
                         feature_names, utility = DC,
                         sample_size = sample_size)
    cd[,,i] <- cdi
    
  }
  dimnames(cd) <- list(S = paste0("S",1:9), feature = feature_names, i = 1:N)
  return(cd)
}

plot_compare_DARRP_N_drift2 <- function(
  cdN_all, p = c(0.025,0.975), d = dim(cdN_all)[2],
  feature_names = paste0("X",1:d), shap_index = c(1,5), #only two allowed
  shap_type = c("ADL", "ADR"), y_name = "Shapley value",
  leg_labs = unname(c(TeX("$X_1"),TeX("$X_2"),TeX("$X_3"),TeX("$X_4")))) {
  
  m <- dim(cdN_all)[4]
  N <- dim(cdN_all)[3]
  d <- dim(cdN_all)[2]
  s <- dim(cdN_all)[1]
  dimnames(cdN_all) <- list(
    S = 1:s, feature = paste0("X",1:d), i = 1:N, t = 0:(m-1))
  cdN_lines <- as_tibble(reshape2::melt(cdN_all)) %>%
    dplyr::filter(S %in% shap_index)
  cdN_lines$S <- rep(shap_type, nrow(cdN_lines)/2)
  cd2 <- matrix(ncol = d, nrow = 0)
  for (t in 1:m) {
    cdt <- cdN_all[shap_index,,,t] %>%
      apply(MARGIN = c(1,2), FUN = function(x){
        c(mean(x), quantile(x, probs = p))
      })
    for (i in 1:length(shap_index)) {cd2 <- rbind(cd2, cdt[,i,])}
  }
  colnames(cd2) <- feature_names
  cd2 <- cd2 %>%
    data.frame() %>%
    cbind(S = rep(shap_type,each=3),
          CI = 1:3, time = rep(0:(m-1), each = 6)) %>%
    pivot_longer(all_of(feature_names), names_to = "feature") %>%
    pivot_wider(names_from = CI, values_from = value, names_prefix = "CI")
  
  plt <- ggplot(data=cd2, aes(x=time, y=CI1, fill=feature,
                              shape=feature)) +
    geom_point() + geom_line() +
    geom_ribbon(aes(ymin=CI2, ymax=CI3), alpha=0.1) +
    scale_x_continuous(breaks = 0:m) +
    scale_y_continuous(name = y_name) +
    scale_shape_discrete(labels=leg_labs) +
    scale_fill_discrete(labels=leg_labs) +
    theme_set(theme_minimal())  +
    facet_grid(S ~ .) +
    theme(strip.text.y.right = element_text(angle = 0)) +
    geom_line(data = cdN_lines,
              mapping = aes(y=value, x=t, group=interaction(i,feature),
                            colour=interaction(i,feature)),
              alpha=0.02, colour="grey20"); plt
  return(plt)
}

plot_compare_DARRP_N_interact_all <- function(
  cdN_all, p = c(0.025,0.975),
  d = dim(cdN_all)[2],
  feature_names = paste0("X",1:d),
  y_name = "Shapley value",
  leg_labs = unname(TeX(paste0("$X_",1:d))),
  colpal=c("#CC79A7", "#0072B2", "#D55E00")) {
  
  s <- dim(cdN_all)[1]
  d <- dim(cdN_all)[2]
  N <- dim(cdN_all)[3]
  cd2 <- matrix(ncol = s, nrow = 0)
  cdN <- matrix(ncol = d, nrow = 0)
  dimnames(cdN_all) <- list(
    S = paste0("S",1:s), feature = feature_names, i = 1:N)
  cd <- cdN_all %>% 
    apply(MARGIN = c(1,2), FUN = function(x){ 
      c(mean(x), quantile(x, probs = p)) 
    })
  for (i in 1:d) {cd2 <- rbind(cd2, cd[,,i])}
  for (i in 1:N) {cdN <- rbind(cdN, cdN_all[,,i])}
  cdN <- cdN %>% 
    data.frame() %>%
    cbind(i = rep(1:N,each=s), S = paste0("S",1:s), .)
  
  cd2 <- cd2 %>% 
    data.frame() %>% 
    cbind(feature = rep(feature_names,each=3), CI = 1:3) %>% 
    pivot_longer(all_of(paste0("S",1:s)), names_to = "S")
  
  cd2 <- cd2 %>% mutate(type=recode(S, S1="ADL", S3="ADP", S5="ADR")) %>% 
    mutate(facet=recode(S, S1="F1", S3="F1", S5="F2")) %>%
    select(-S)
  cd3 <- cd2 %>% 
    filter(type %in% c("ADL","ADP") & CI == 1) %>% 
    pivot_wider(names_from = type, values_from = value)
  cdCI <- cd2 %>% 
    filter(type %in% c("ADL","ADP","ADR")) %>% 
    pivot_wider(names_from = CI, values_from = value, names_prefix = "CI")
  cdN2ADLP <- cdN %>% 
    mutate(type=recode(S, S1="ADL", S3="ADP", S5="ADR")) %>% 
    mutate(facet=recode(S, S1="F1", S3="F1", S5="F2")) %>%
    select(-S) %>% 
    filter(type %in% c("ADL","ADP","ADR")) %>% 
    pivot_longer(all_of(feature_names), names_to="feature")
  
  # The flossy dumbbell plot
  ggplot(data=cdN2ADLP) +
    geom_segment(data=cd3, aes(x=feature, 
                               xend=feature, 
                               y=ADL, 
                               yend=ADP), color="grey75", size=2) +
    geom_segment(data=filter(cdCI,type=="ADR"), 
                 aes(x=feature, 
                     xend=feature, 
                     yend=CI1, 
                     y=0), color="indianred4", size=2) +
    theme_set(theme_minimal()) +
    xlab("") +
    theme(strip.text.y = element_blank()) +
    theme(strip.text.x = element_blank()) +
    facet_grid(facet~.) +
    ylab("Shapley value") +
    scale_x_discrete(labels=leg_labs) +
    scale_fill_manual(values=colpal) +
    scale_colour_manual(values=colpal) +
    scale_shape_manual(values=c(15,16,17)) +
    geom_jitter(alpha=0.4,width=0.14,size=0.75,
                aes(x=feature,y=value,colour=type,shape=type)) +
    geom_point(data=cdCI, aes(x=feature, y=CI1, shape=type),  size=1) +
    geom_crossbar(data=cdCI, fatten=1, alpha=0.3, width=0.3,linetype=0,
                  aes(y=CI1, ymin=CI2, ymax=CI3, x=feature,
                      colour=type, fill=type)) +
    geom_hline(data=data.frame(facet="F2"),
               aes(yintercept=0), colour="grey",linetype=1)
}

compare_DARRRP_N_gender_4way <- function(
  sdat4way, xgb, sample_size = 1000, 
  N = 100, features, feature_names) {
  xgbt <- basic_xgb_test2(xgb, sdat4way)
  cdN <- compare_DARRP_N2(sdat4way, xgbt, features = features, 
                          feature_names = feature_names,
                          sample_size = sample_size, N = N)
  return(list(cdN = cdN, xgb = xgb, xgbt = xgbt))
}

split_dat_gender_4way <- function(dat, gender = "sex_isFemale", gender_M = F) {
  n <- nrow(dat)
  male_index <- which(dat[[gender]] == gender_M)
  nm <- length(male_index)
  X_m <- dat[male_index,-1]
  X_f <- dat[-male_index,-1]
  y_m <- dat[male_index,1]
  y_f <- dat[-male_index,1]
  train_m <- sample(1:nrow(X_m), as.integer(nm/2))
  train_f <- sample(1:nrow(X_f), as.integer(nm/2))
  x_train <- rbind(X_m[train_m,], X_f[train_f,])
  y_train <- c(y_m[train_m], y_f[train_f])
  x_males <- X_m[-train_m,]
  y_males <- y_m[-train_m]
  x_females <- X_f[-train_f,]
  y_females <- y_f[-train_f]
  nmt <- nrow(x_males)
  nft <- nrow(x_females)
  test_f <- sample(1:nft, nmt)
  x_test <- rbind(x_males, x_females[test_f,])
  y_test <- c(y_males, y_females[test_f])
  
  return(list(y_train = as.matrix(y_train), 
              y_test = as.matrix(y_test),
              y_males = as.matrix(y_males),
              y_females = as.matrix(y_females),
              x_train = as.matrix(x_train), 
              x_test = as.matrix(x_test),
              x_males = as.matrix(x_males),
              x_females = as.matrix(x_females)))
}

plot_compare_DARRP_N_4way <- function(
  cdN_all, p = c(0.025,0.975),
  d = dim(cdN_all)[2],
  y_name = "Shapley value",
  colpal=c("#CC79A7", "#0072B2", "#D55E00")) {
  
  p = c(0.025,0.975)
  d = dim(cdN_all)[2]
  y_name = "Shapley value"
  colpal=c("#CC79A7", "#0072B2", "#D55E00")
  
  s <- dim(cdN_all)[1]
  d <- dim(cdN_all)[2]
  N <- dim(cdN_all)[3]
  cd2 <- matrix(ncol=s, nrow=0)
  cdN <- matrix(ncol=d, nrow=0)
  S <- dimnames(cdN_all)[[1]]
  feature_names <- dimnames(cdN_all)[[2]]
  i <- dimnames(cdN_all)[[3]]
  cd <- cdN_all %>% 
    apply(MARGIN = c(1,2), FUN = function(x){ 
      c(mean(x), quantile(x, probs=p)) 
    })
  for (i in 1:d) {cd2 <- rbind(cd2, cd[,,i])}
  for (i in 1:N) {cdN <- rbind(cdN, cdN_all[,,i])}
  cdN <- cdN %>% 
    data.frame() %>%
    cbind(i = rep(1:N,each=s), S=paste0("S",1:s), .)
  
  cd2 <- cd2 %>% 
    data.frame() %>% 
    cbind(feature = rep(feature_names,each=3), CI=1:3) %>% 
    pivot_longer(all_of(paste0("S",1:s)), names_to="S")
  
  cd2 <- cd2 %>% 
    mutate(type=recode(S, 
                       S1="ADL", S2="ADL", S3="ADL",
                       S4="ADP", S5="ADP", S6="ADP",
                       S7="ADR", S8="ADR", S9="ADR")) %>% 
    select(-S) %>% 
    mutate(facet=recode(type, ADL="F1", ADP="F1", ADR="F2"))
  sets <- c("balanced","all male","all female")
  cd2$set <- factor(rep(sets, nrow(cd2)/3), levels=sets)
  
  cdN <- cdN %>% 
    mutate(type=recode(S, 
                       S1="ADL", S2="ADL", S3="ADL",
                       S4="ADP", S5="ADP", S6="ADP",
                       S7="ADR", S8="ADR", S9="ADR")) %>% 
    select(-S) %>% 
    mutate(facet=recode(type, ADL="F1", ADP="F1", ADR="F2"))
  cdN$set <- factor(rep(sets, nrow(cdN)/3), levels=sets)
  
  cd3 <- cd2 %>% 
    filter(type %in% c("ADL","ADP") & CI == 1) %>% 
    pivot_wider(names_from=type, values_from=value)
  cdCI <- cd2 %>% 
    filter(type %in% c("ADL","ADP","ADR")) %>% 
    pivot_wider(names_from=CI, values_from=value, names_prefix = "CI")
  cdN2 <- cdN %>% 
    filter(type %in% c("ADL","ADP","ADR")) %>% 
    pivot_longer(all_of(feature_names), names_to="feature")
  
  ggplot(data=cdN2) +
    geom_segment(data=cd3, aes(x=feature, 
                               xend=feature, 
                               y=ADL, 
                               yend=ADP), color="grey75", size=2) +
    geom_segment(data=filter(cdCI,type=="ADR"), 
                 aes(x=feature, 
                     xend=feature, 
                     yend=CI1, 
                     y=0), color="indianred4", size=2) +
    theme_set(theme_minimal()) +
    xlab("") +
    facet_grid(facet~set) +
    ylab("Shapley value") +
    scale_fill_manual(values=colpal) +
    scale_colour_manual(values=colpal) +
    scale_shape_manual(values=c(15,16,17)) +
    geom_jitter(alpha=0.4,width=0.14,size=0.75,
                aes(x=feature,y=value,colour=type,shape=type)) +
    theme(strip.text.y = element_blank()) +
    geom_point(data=cdCI, aes(x=feature, y=CI1, shape=type),  size=1) +
    geom_crossbar(data=cdCI, fatten=1, alpha=0.3, width=0.3,linetype=0,
                  aes(y=CI1, ymin=CI2, ymax=CI3, x=feature,
                      colour=type, fill=type)) +
    geom_hline(data=data.frame(facet="F2"),
               aes(yintercept=0), colour="grey",linetype=1)
}

basic_xgb_test2 <- function(bst, sdat4way) {
  pred_test <- predict(bst, sdat4way$x_test)
  pred_males <- predict(bst, sdat4way$x_males)
  pred_females <- predict(bst, sdat4way$x_females)
  residuals_test <- sdat4way$y_test - pred_test
  residuals_males <- sdat4way$y_males - pred_males
  residuals_females <- sdat4way$y_females - pred_females
  return(list(pred_test = pred_test,
              pred_males = pred_males,
              pred_females = pred_females,
              residuals_test = residuals_test,
              residuals_males = residuals_males,
              residuals_females = residuals_females))
}


plot_compare_DARRP_N_interact_ADR <- function(
  cdN_all, p = c(0.025,0.975),
  d = dim(cdN_all)[2],
  feature_names = paste0("X",1:d),
  y_name = "Shapley value",
  leg_labs = unname(TeX(paste0("$X_",1:d)))) {
  
  # p = c(0.025,0.975)
  # d = dim(cdN_all)[2]
  # feature_names = paste0("X",1:d)
  # y_name = "Shapley value"
  # leg_labs = unname(TeX(paste0("$X_",1:d)))
  
  s <- dim(cdN_all)[1]
  N <- dim(cdN_all)[3]
  cd2 <- matrix(ncol = s, nrow = 0)
  cdN <- matrix(ncol = d, nrow = 0)
  dimnames(cdN_all) <- list(
    S = paste0("S",1:s), feature = feature_names, i = 1:N)
  cd <- cdN_all %>% 
    apply(MARGIN = c(1,2), FUN = function(x){ 
      c(mean(x), quantile(x, probs = p)) 
    })
  for (i in 1:d) {cd2 <- rbind(cd2, cd[,,i])}
  for (i in 1:N) {cdN <- rbind(cdN, cdN_all[,,i])}
  cdN <- cdN %>% 
    data.frame() %>%
    cbind(i = rep(1:N,each=s), S = paste0("S",1:s), .)
  
  cd2 <- cd2 %>% 
    data.frame() %>% 
    cbind(feature = rep(feature_names,each=3), CI = 1:3) %>% 
    pivot_longer(all_of(paste0("S",1:s)), names_to = "S")
  
  cd2 <- cd2 %>% mutate(type=recode(S, S5 = "ADR")) %>% 
    select(-S)
  cd3 <- cd2 %>% 
    filter(type == "ADR") %>% 
    pivot_wider(names_from = CI, values_from = value, names_prefix = "CI")
  cdN2 <- cdN %>% 
    mutate(type=recode(S, S5="ADR")) %>% 
    select(-S) %>% 
    filter(type == "ADR") %>% 
    pivot_longer(all_of(feature_names), names_to="feature")
  
  ggplot(cd3, aes(x=feature, y=CI1)) +
    geom_segment(aes(x=feature, 
                     xend=feature, 
                     yend=CI1, 
                     y=0), color="indianred4", size=2) +
    geom_crossbar(fatten=1, alpha=0.3, width=0.3,linetype=0,
                  aes(y=CI1, ymin=CI2, ymax=CI3, x=feature,
                      colour=type, fill=type)) +
    theme_minimal() +
    xlab("") +
    ylab("Shapley value") +
    geom_hline(aes(yintercept=0), colour="grey",linetype=1) +
    scale_x_discrete(labels=leg_labs) +
    geom_point(size=3,colour="indianred") +
    geom_jitter(data=cdN2,alpha=0.4,width=0.14,shape=18,
                aes(x=feature,y=value,colour=type))
}

plot_compare_DARRP_N_interact_ADL_ADP <- function(
  cdN_all, p = c(0.025,0.975),
  d = dim(cdN_all)[2],
  feature_names = paste0("X",1:d),
  y_name = "Shapley value",
  leg_labs = unname(TeX(paste0("$X_",1:d)))) {
  
  s <- dim(cdN_all)[1]
  d <- dim(cdN_all)[2]
  N <- dim(cdN_all)[3]
  cd2 <- matrix(ncol = s, nrow = 0)
  cdN <- matrix(ncol = d, nrow = 0)
  dimnames(cdN_all) <- list(
    S = paste0("S",1:s), feature = feature_names, i = 1:N)
  cd <- cdN_all %>% 
    apply(MARGIN = c(1,2), FUN = function(x){ 
      c(mean(x), quantile(x, probs = p)) 
    })
  for (i in 1:d) {cd2 <- rbind(cd2, cd[,,i])}
  for (i in 1:N) {cdN <- rbind(cdN, cdN_all[,,i])}
  cdN <- cdN %>% 
    data.frame() %>%
    cbind(i = rep(1:N,each=s), S = paste0("S",1:s), .)
  
  cd2 <- cd2 %>% 
    data.frame() %>% 
    cbind(feature = rep(feature_names,each=3), CI = 1:3) %>% 
    pivot_longer(all_of(paste0("S",1:s)), names_to = "S")
  
  cd2 <- cd2 %>% mutate(type=recode(S, S1="ADL", S3="ADP")) %>% 
    select(-S)
  cd3 <- cd2 %>% 
    filter(type %in% c("ADL","ADP") & CI == 1) %>% 
    pivot_wider(names_from = type, values_from = value)
  cdCI <- cd2 %>% 
    filter(type %in% c("ADL","ADP")) %>% 
    pivot_wider(names_from = CI, values_from = value, names_prefix = "CI")
  cdN2 <- cdN %>% 
    mutate(type=recode(S, S1="ADL", S3="ADP")) %>% 
    select(-S) %>% 
    filter(type %in% c("ADL","ADP")) %>% 
    pivot_longer(all_of(feature_names), names_to="feature")
  
  # The flossy dumbbell plot
  ggplot(cd3) +
    geom_segment( aes(x=feature, 
                      xend=feature, 
                      y=ADL, 
                      yend=ADP), color="grey75", size=2) +
    geom_point( aes(x=feature, y=ADL),  size=3 ) + #color=rgb(0.2,0.7,0.1,0.8),
    geom_point( aes(x=feature, y=ADP),  size=3 ) + #color=rgb(0.7,0.2,0.1,0.8),
    theme_set(theme_minimal())  +
    xlab("") +
    ylab("Shapley value") +
    scale_x_discrete(labels=leg_labs) +
    geom_crossbar(data=cdCI, fatten=1, alpha=0.3, width=0.3,linetype=0,
                  aes(y=CI1, ymin=CI2, ymax=CI3, x=feature,
                      colour=type, fill=type)) +
    geom_jitter(data=cdN2,alpha=0.4,width=0.14,shape=18,
                aes(x=feature,y=value,colour=type))
}

plot_compare_DARRP_N_drift <- function(
  cdN_all, p = c(0.025,0.975),
  d = dim(cdN_all)[2],
  feature_names = paste0("X",1:d), 
  shap_index = c(1,5), #only two allowed
  shap_type = c("ADL", "ADR"), 
  y_name = "Shapley value",
  leg_labs = unname(c(TeX("$X_1"),TeX("$X_2"),TeX("$X_3"),TeX("$X_4")))) {
  
  m <- dim(cdN_all)[4]
  cd2 <- matrix(ncol = d, nrow = 0)
  for (t in 1:m) {
    cdt <- cdN_all[shap_index,,,t] %>% 
      apply(MARGIN = c(1,2), FUN = function(x){ 
        c(mean(x), quantile(x, probs = p)) 
      })
    for (i in 1:length(shap_index)) {cd2 <- rbind(cd2, cdt[,i,])}
  }
  colnames(cd2) <- feature_names
  cd2 <- cd2 %>% 
    data.frame() %>% 
    cbind(S = rep(shap_type,each=3), 
          CI = 1:3, time = rep(0:(m-1), each = 6)) %>% 
    pivot_longer(all_of(feature_names), names_to = "feature") %>% 
    pivot_wider(names_from = CI, values_from = value, names_prefix = "CI")
  
  plt <- ggplot(data=cd2, aes(x=time, y=CI1, colour=feature, fill=feature,
                              shape=feature, linetype=feature)) + 
    geom_point() + geom_line() +
    geom_ribbon(aes(ymin=CI2, ymax=CI3), alpha=0.1) +
    scale_x_continuous(breaks = 0:m) +
    scale_y_continuous(name = y_name) +
    scale_colour_discrete(labels=leg_labs) +
    scale_shape_discrete(labels=leg_labs) +
    scale_fill_discrete(labels=leg_labs) +
    scale_linetype_discrete(labels=leg_labs) + 
    theme_set(theme_minimal())  +
    facet_grid(S ~ .) +
    theme(strip.text.y.right = element_text(angle = 0))
  return(plt)
}

plot_compare_DARRP_N <- function(cdN, p = c(0.025,0.975), main, valid = F,
                                 all_labs = T) {
   feature_names <- dimnames(cdN)[[2]]
   cd <- apply(cdN, FUN = mean, MARGIN = c(1,2))
   cd_L <- apply(cdN, FUN = quantile, MARGIN = c(1,2), probs = p[1])
   cd_U <- apply(cdN, FUN = quantile, MARGIN = c(1,2), probs = p[2])

   centers <- plot_compare_DARRP(cd, main = main, valid = valid, all_labs = all_labs)
   arrows(centers, cd_L, centers, cd_U, 
          lwd = 1.5, angle = 90, code = 3, length = 0.05)
   segments(centers, cd_L, centers, cd_U, lwd = 1.5)
}

plot_compare_DARRP <- function(s,
     leg_loc = "topright", main = "untitled",
     valid = F, all_labs = T) {
  if(!valid) {
    legend <- c("ERDA test","ERDA train",
      "PDA test", "PDA train", 
      "RDA test","RDA train")
    cols <- c("gray50","gray75",
              "blue", "lightblue",
              "darkred","red3") 
  } else {
    legend <- c("ERDA test", "ERDA train", "ERDA valid",
        "PDA test", "PDA train",  "PDA valid",
        "RDA test","RDA train", "RDA valid")
    cols <- c("gray50","gray75","gray100",
              "blue", "lightblue","lightsteelblue2",
              "darkred","red3","violetred3")
  }
  if (all_labs) {
    legend <- c("ERDA all", legend)
    cols <- c("gray25", cols)
  }
  centers <- barplot(s,
               xlab = "Feature",
               ylab = "Attribution",
               col = cols,
               beside = T,
               main = main)
  legend(x = leg_loc, legend = legend, 
         col = cols, pch = rep(15,7 + valid*3))
  centers
}

compare_DARRRP_N_gender <- function(
  dat, p1f, p2f, sample_size = 1000, 
  N = 100, features, feature_names) {
  
  sdat <- split_dat_gender(dat, p1f, p2f)
  xgb <- basic_xgb_fit(sdat)
  xgbt <- basic_xgb_test(xgb, sdat)
  cdN <- compare_DARRP_N(sdat, xgbt, features = features, 
                         feature_names = feature_names,
                         sample_size = sample_size, N = N)
  plot_compare_DARRP_N(cdN, main = paste0("p1f: ", p1f, ",  ",
                                          "p2f: ", p2f))
  return(list(cdN = cdN, xgb = xgb, xgbt = xgbt))
}

compare_DARRRP_N_gender_3way <- function(
  dat, sample_size = 1000, 
  N = 100, features, feature_names) {
  
  sdat <- split_dat_gender_3way(dat)
  xgb <- basic_xgb_fit(sdat)
  xgbt <- basic_xgb_test(xgb, sdat, valid = T)
  cdN <- compare_DARRP_N(sdat, xgbt, features = features, 
                         feature_names = feature_names,
                         sample_size = sample_size, N = N,
                         valid = T)
  plot_compare_DARRP_N(cdN, main = paste0("p1f: ", 0.5, ",  ",
                                          "p2f: ", 0, ", ",
                                          "p3f: ", 1), valid = T)
  return(list(cdN = cdN, xgb = xgb, xgbt = xgbt))
}


basic_lmodel_test <- function(lmodel, dat, plots = F) {
  pred_test <- predict(lmodel, dat$df_yx_test)
  pred_train <- predict(lmodel, dat$df_yx_train)
  residuals_test <- dat$y_test - pred_test
  residuals_train <- dat$y_train - pred_train
  if (plots) {plot(pred_test, dat$y_test)}
  return(list(pred_test = pred_test,
              pred_train = pred_train,
              residuals_test = residuals_test,
              residuals_train = residuals_train))
}

diagnostics <- function(sdat, xgbt, plot = "all", 
                        features = 1:ncol(sdat$x_test),
                        feature_names) {
  shap_res <- shapley(xgbt$residuals_test, 
                      sdat$x_test[,features], utility = DC)
  shap_lab <- shapley(sdat$y_test, sdat$x_test[,features], utility = DC)
  shap_pred <- shapley(xgbt$pred_test, sdat$x_test[,features], utility = DC)
  shap_lab_diff <- shap_lab - 
    shapley(sdat$y_train, sdat$x_train[,features], utility = DC)
  if (tolower(plot) == "all") {
    plot(xgbt$pred_test, xgbt$residuals_test,
         ylab = "Residuals", xlab = "Fitted values",
         main = "res vs fits")
    abline(h = 0, col = "red")
    barplot(shap_res,
            main = "residuals shap (test set)")
    barplot(shap_lab, 
            main = "labels shap (training set)")
    m <- rbind(shap_lab, shap_pred, shap_res)
    if (!missing(feature_names)) {colnames(m) <- feature_names}
    barplot(m,
            xlab = "Feature",
            ylab = "Attribution",
            col = c("black","gray","red"),
            beside = T)
    legend(x = "top", legend = c("labels","predictions","residuals"), 
           col = c("black","gray","red"), pch = c(15,15,15))
  }
  if (tolower(plot) == "rvf") {
    plot(xgbt$pred_test, xgbt$residuals_test,
         ylab = "Residuals", xlab = "Fitted values",
         main = "res vs fits")
    abline(h = 0, col = "red")
  }
  return(list(shap_lab = shap_lab, shap_res = shap_res, shap_pred = shap_pred,
              shap_lab_diff = shap_lab_diff))
}

# Default xgb with 10 rounds and 50/50 test split, returns model, preds and accuracy 
basic_xgb <- function(sdat, plots = F, obj = "reg:squarederror") {
  binary <- T
  if (length(unique(sdat$y_train)) > 2) {binary <- F}
  obj <- if (binary) {"binary:logistic"} else {obj}
  bst <- xgboost(
    data = sdat$x_train,
    label = sdat$y_train,
    nround = 20,
    verbose = FALSE,
    objective = obj
  )
  pred_test <- predict(bst, sdat$x_test)
  pred_train <- predict(bst, sdat$x_train)
  residuals_test <- sdat$y_test - pred_test
  residuals_train <- sdat$y_train - pred_train
  if (plots) {plot(pred_test, sdat$y_test)}
  if (binary) {
    pred_test <- as.numeric(pred_test > 0.5)
    acc <- sum(pred_test == sdat$y_test)/length(sdat$y_test)
    mse <- "not applicable (binary response)"
  } else {
    mse <- mean((pred_test - sdat$y_test)^2) 
    acc <- "not applicable (continuous response)"
  }
  test_mse <- mse
  test_acc <- acc
  attr(bst, "binary") <- binary
  return(list(bst = bst, 
              pred_test = pred_test,
              test_mse = test_mse,
              test_acc = test_acc,
              pred_train = pred_train,
              residuals_test = residuals_test,
              residuals_train = residuals_train))
}

### This uses hyperparameters taken from 
# https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html
nhanes_xgb_fit <- function(dat, nround=20, verbose=T) {
  params <- list(
    eta = 0.002,
    max_depth = 3,
    objective = "survival:cox",
    subsample = 0.5)
  bst <- xgboost(
    params = params,
    data = dat$x_train,
    label = dat$y_train,
    nround = nround,
    verbose = verbose
  )
}

basic_xgb_fit <- function(dat, obj = "reg:squarederror") {
  binary <- T
  if (length(unique(dat$y_train)) > 2) {binary <- F}
  obj <- if (binary) {"binary:logistic"} else {obj}
  bst <- xgboost(
    data = dat$x_train,
    label = dat$y_train,
    nround = 20,
    verbose = FALSE,
    objective = obj
  )
  attr(bst, "binary") <- binary
  return(bst)
}

basic_xgb_test <- function(bst, dat, plots = F, valid = F) {
  binary <- attr(bst, "binary")
  pred_test <- predict(bst, dat$x_test)
  pred_train <- predict(bst, dat$x_train)
  pred_valid <- if(valid) {predict(bst, dat$x_valid)} else {NULL}
  residuals_test <- dat$y_test - pred_test
  residuals_train <- dat$y_train - pred_train
  residuals_valid <- if(valid) {dat$y_valid - pred_valid} else {NULL} 
  if (plots) {plot(pred_test, dat$y_test)}
  if (binary) {
    pred <- as.numeric(pred_test > 0.5)
    acc <- sum(pred == dat$y_test)/length(dat$y_test)
    mse <- "not applicable (binary response)"
  } else {
    mse <- mean((pred_test - dat$y_test)^2) 
    acc <- "not applicable (continuous response)"
  }
  test_mse <- mse
  test_acc <- acc
  return(list(test_mse = test_mse,
              test_acc = test_acc,
              pred_test = pred_test,
              pred_train = pred_train,
              pred_valid = pred_valid,
              residuals_test = residuals_test,
              residuals_train = residuals_train,
              residuals_valid = residuals_valid))
}


## Utility of each feature alone, then utility of all features together
examine_utility <- function(dat, utility) {
  for (i in 2:ncol(dat)) { 
    cat(paste0("C({",i-1,"}): ", utility(dat[,1,drop=F], dat[,i,drop=F])),"\n") 
  }
  cat(paste0("C([d]): ", utility(dat[,1,drop=F], dat[,-1,drop=F])),"\n") 
}

# Use SHAPforxgboost library to get SHAP values
examine_SHAP <- function(bst, x_train, plots = F) {
  shap_values <- shap.values(xgb_model = bst, X_train = x_train)
  shap_long <- shap.prep(xgb_model = bst, X_train = x_train)
  return(list(shapm = shap_values$mean_shap_score, 
              shapp = shap_long))
}

# Run and print all the evaluations, returning the xgb model
## Parameters:
# data_gen: data generating function
# utility: utility function
# n: sample size
# ...: other agurments to data_gen
run_evaluations <- function(data_gen, utility, n = 1e3,  plots = F, ...) {
  dgp_name <- toupper(as.character(substitute(data_gen)))
  cat(paste0("\n-----",dgp_name,"-----\n"))
  dat <- data_gen(n = n, ...)
  dat <- split_dat(dat)
  xgb <- basic_xgb(dat, plots = plots)
  cat("---\n")
  imp <- xgb.importance(model = xgb$bst)
  print(imp)
  cat("---\n")
  cat(paste0("xgb acc: ", xgb$test_acc,"\n"))
  cat(paste0("xgb mse: ", xgb$test_mse,"\n"))
  cat("---\n")
  examine_utility(dat$dat, utility)
  cat("---\n")
  shaps_test <- shapley(cbind(dat$y_test, dat$x_test), utility = utility)
  shaps_train <- shapley(cbind(dat$y_train, dat$x_train), utility = utility)
  shaps_preds_test <- shapley(
    cbind(xgb$pred_test, dat$x_test), utility = utility)
  shaps_preds_train <- shapley(
    cbind(xgb$pred_train, dat$x_train), utility = utility)
  shaps_diff_test <- shaps_preds_test - shaps_test
  shaps_diff_train <- shaps_preds_train - shaps_train
  shaps_res_test <- shapley(xgb$residuals_test, dat$x_test, utility = DC)
  shaps_res_train <- shapley(xgb$residuals_train, dat$x_train, utility = DC)
  cat("Sunnies test: ", shaps_test, "\n")
  cat("Sunnies train: ", shaps_train, "\n")
  cat("Sunnies preds test: ", shaps_preds_test, "\n")
  cat("Sunnies preds train: ", shaps_preds_train, "\n")
  cat("Sunnies diffs (pred-lab) test: ", shaps_diff_test, "\n")
  cat("Sunnies diffs (pred-lab) train: ", shaps_diff_train, "\n")
  cat("Sunnies res test: ", shaps_res_test, "\n")
  cat("Sunnies res train: ", shaps_res_train, "\n")
  cat("---\n")
  SHAP_test <- examine_SHAP(xgb$bst, dat$x_test, plots = plots)
  SHAP_train <- examine_SHAP(xgb$bst, dat$x_train, plots = plots)
  if (plots) {
    barplot(imp$Gain, main = "xgb.importance", names.arg = imp$Feature)
    barplot(SHAP_train$shapm, main = "SHAP train")
    barplot(SHAP_test$shapm, main = "SHAP test")
    U <- as.character(substitute(utility))
    plot(xgb$pred_test, xgb$residuals_test,
         ylab = "Residuals", xlab = "Fitted values",
         main = "res vs fits")
    abline(h = 0, col = "red")
    barplot(shaps_test, main = paste0("sunnies test "))
    barplot(shaps_train, main = paste0("sunnies train "))
    barplot(shaps_preds_test, main = paste0("sunnies preds test "))
    barplot(shaps_preds_train, main = paste0("sunnies preds train "))
    barplot(shaps_diff_test, main = paste0("s(pred)-s(lab) test "))
    barplot(shaps_diff_train, main = paste0("s(pred)-s(lab) train "))
    barplot(shaps_res_test, main = "s(lab - pred) test ")
    barplot(shaps_res_train, main = "s(lab - pred) train ")
  }
  cat("SHAP train: ", SHAP_train$shapm, "\n")
  cat("SHAP test: ", SHAP_test$shapm, "\n")
  #if (plots) {shap.plot.summary(SHAP$shapp)}
  cat("---\n")
  cat("\n")
  return(list(xgb = xgb, 
              shapp_train = SHAP_train$shapp,
              shapp_test = SHAP_test$shapp,
              dat = dat))
}
