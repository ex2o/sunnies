###############################
dat_t <- function(n, d, t, max_t, extra_features = T, de = 46) {
  X <- matrix(rnorm(n*d,0,2), nrow = n, ncol = d)
  extras <- if (extra_features) {
    Xe <- matrix(rnorm(n*de,0,sqrt(0.05)), nrow = n, ncol = de)
    rowSums(Xe)
  } else {0}
  y <- X %*% c(rep(1,d-2), 1 + t/max_t, 1 - t/max_t) + extras
  
  if (extra_features) {return(cbind(y,X,Xe))}
  return(cbind(y,X))
}


###############################
dat_a_few_important <- function(n = 1e3, d1 = 4, d0 = 10,
                                A1 = 1:d1, A0 = rep(0.01,d0)) {
  X1 <- matrix(rnorm(n*d1), nrow = n, ncol = d1)
  X0 <- matrix(rnorm(n*d0), nrow = n, ncol = d0)
  X <- matrix(0, nrow = n, ncol = d1+d0)
  important <- sample(1:ncol(X), ncol(X1))
  X[,important] <- X1
  X[,-important] <- X0
  y <- X1 %*% A1 + X0 %*% A0
  dat <- cbind(y,X)
  attr(dat, "important") <- important
  return(dat)
}


###############################
dat_linear_interaction <- function(n = 1e3, d = 4) {
  X <- matrix(rnorm(n*d), nrow = n, ncol = d)
  y <- rowSums(X) + X[,d-1]*X[,d]
  dat <- cbind(y,X)
  return(dat)
}


##############################
# y = 0*X_1^2 + 2*X_2^2 + ... + (d-1)X_d^2 where X_i ~ unif(-1,1) 
dat_unif_squared <- function(d = 4, n = 100, A = (2*(0:(d-1))),
                             add_noise = F) {
  X <- matrix(runif(n*d,-1,1), n, d)
  y <- X^2 %*% A
  if (add_noise) {
    X[,1] <- X[,1] + rnorm(n) 
  }
  dat <- cbind(y,X)
  colnames(dat) <- c("y",paste0("x",1:d))
  return(dat)
}

##############################
# This is a modification of dat_unif_squared, where X is non-random
dat_nonrandom_squared <- function(d = 4, n = 100, A = (2*(0:(d-1)))) {
  X <- matrix(seq(-1,1,length.out = n*d),n,d)
  y <- X^2 %*% A
  dat <- cbind(y,X)
  colnames(dat) <- c("y",paste0("x",1:d))
  return(dat)
}

##############################
# This is almost non-random
dat_nonnoisy_squared <- function(d = 4, n = 100, sd = 0.01) {
  X <- matrix(seq(-1,1, length.out = n*d),n,d)
  for (i in 1:d) {X[,i] <- X[,i] + rnorm(n,0,sd)}
  y <- X^2 %*% rep(1,d)
  dat <- cbind(y,X)
  colnames(dat) <- c("y",paste0("x",1:d))
  return(dat)
}

##############################
# This is a modification of dat_unif_squared, where all
# the coefficients are set to be 1, and
# we set X_2 = X_1 + eps_1 and X_3 = X_1 + eps_2
# where eps_i ~ N(0, sigma^2)
dat_unif_squared_corr <- function(d = 4, n = 100, sigma = 0.2) {
  X <- matrix(runif(n*d,-1,1), n, d)
  X[,1] <- X[,2] + rnorm(n, sd = sigma)
  X[,3] <- X[,2] + rnorm(n, sd = sigma)
  X[,2] <- X[,2] + rnorm(n, sd = sigma)
  y <- X^2 %*% rep(1,d)
  dat <- cbind(y,X)
  colnames(dat) <- c("y",paste0("x",1:d))
  return(dat)
}

##############################
# There is no relationship, just uniform on all variables
dat_unif_independent <- function(d = 4, n = 100) {
  dat <- matrix(runif(n*(d+1),-1,1), n, d+1)
  colnames(dat) <- c("y",paste0("x",1:d))
  return(dat)
}

##############################
# A cosine symmetric about 0
dat_unif_cos <- function(d = 4, n = 100, A = (2*(0:(d-1)))) {
  X <- matrix(runif(n*d,-pi,pi), n, d)
  y <- cos(X) %*% A
  dat <- cbind(y,X)
  colnames(dat) <- c("y",paste0("x",1:d))
  return(dat)
}

##############################
# A step function symmetric about 0
dat_unif_step <- function(d = 4, n = 100, A = (2*(0:(d-1)))) {
  #X <- runif(n,-1,1)
  X <- matrix(runif(n*d,-1,1), n, d)
  y <- (-0.5 < X & X < 0.5) %*% A
  dat <- cbind(y,X)
  colnames(dat) <- c("y",paste0("x",1:d))
  return(dat)
}

###############################
# An XOR-like thing with continuity,
# Continuous features with continuous labels
dat_concon_XOR <- function(n = 100) {
  x1 <- runif(n,-1,1)
  x2 <- runif(n,-1,1)
  y <- x1*(x1 > 0 & x2 < 0) + x2*(x1 < 0 & x2 > 0) +
       x1*(x1 < 0 & x2 < 0) - x2*(x1 > 0 & x2 > 0)
  dat <- cbind(y,x1,x2)
  colnames(dat) <- c("y",paste0("x",1:2))
  return(dat)
}

###############################
# Discrete features with discrete labels XOR
dat_catcat_XOR <- function(n = 1e3) {
  x1 <- sample(0:1, n, replace = T)
  x2 <- sample(0:1, n, replace = T)
  y  <- as.integer(xor(x1,x2))
  dat <- cbind(y,x1,x2)
  colnames(dat) <- c("y",paste0("x",1:2))
  return(dat)
}

###############################
# Continuous features with discrete labels XOR
dat_concat_XOR <- function(n = 1e3) {
  x1 <- runif(n, -1, 1)
  x2 <- runif(n, -1, 1)
  y  <- as.integer(xor(x1 > 0, x2 > 0))
  #plot(x1,x2, col = y + 1, main = "XOR")
  dat <- cbind(y,x1,x2)
  colnames(dat) <- c("y",paste0("x",1:2))
  return(dat)
}

###############################
# Counterexamples in Probability and Statistics 2.12
dat_tricky_gaussians <- function(n = 1e3) {
  x1 <- rnorm(n, 0, 1)
  x2 <- rnorm(n, 0, 1)
  x3a <- rnorm(n, 0, 2)
  x3 <- abs(x3a)*sign(x1*x2)
  y <- x1 + x2 + x3
  #plot(x2,y)
  #plot(x1,y)
  #plot(x3,y)
  dat <- cbind(y,x1,x2,x3)
  colnames(dat) <- c("y",paste0("x",1:3))
  return(dat)
}
