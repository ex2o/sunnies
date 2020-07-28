# We first remove all columns with more than p missing proportion.
# Then we drop all patients that still have missing values.
# We will use this to plot the effect of p.
# The ncols parameter allows us to optionally specify cols directly
# instead of p.
remove_all_missing <- function(X, p, count_only = F, ncols) {
  n <- nrow(X)
  X <- cbind(1:n, X)
  missv <- naniar::miss_var_summary(X)
  if (missing(ncols)) {
    to_drop <- missv[(missv["pct_miss"][[1]] > p*100),][[1]]
  } else {
    to_drop <- missv[1:ncols,][[1]]
  }
  X <- select(X, -one_of(to_drop))
  X <- tidyr::drop_na(X)
  if (count_only) {
    return(list(ncols = length(to_drop), nrows = n - nrow(X)))
  } else {
    message(paste0("Dropped columns: ", paste0(to_drop, collapse = ", ")))
    message(paste0("Dropped rows: ", paste0(n - nrow(X), collapse = ", ")))
    keep <- X[,1]
    X <- X[,-1]
    attr(X, "keep") <- keep
    return(X)
  }
}

# We will plot dropped rows as a function of dropped columns
plot_all_drops <- function(X, max_cols = 25) {
  rows <- vector(mode = "numeric", length = max_cols); cols <- rows
  for (nc in 1:max_cols) {
    counts <- remove_all_missing(X, count_only = T, ncols = nc)
    cols[nc] <- counts$ncols
    rows[nc] <- counts$nrows
  }
  plot(cols, rows, type = 'b', 
       xlab = "# Dropped Columns",
       ylab = "# Dropped Rows",
       lab = c(15,5,7))
}

visualise_missing <- function(Xh) {
  plots <- list()
  
  # Many X columns have at least one missing value
  plots$p1 <- gg_miss_which(Xh)
  
  # Roughly half the patients have 15 missing values or more:
  missc <- miss_case_summary(Xh)
  cat("Patients missing 15+ vals:", sum(missc["n_miss"] >= 15) / nrow(Xh), "\n")
  plots$p2 <- gg_miss_case(Xh)
  plots$p3 <- gg_miss_case_cumsum(Xh, breaks = 500)
  
  # The missing values are concentrated in these 15 variables:
  missv <- miss_var_summary(Xh)
  num <- sum(missv["pct_miss"] >= 50)
  cat("Variables missing 50%+ cases:", num, "\n")
  plots$p4 <- gg_miss_var(Xh, show_pct = T, facet = sex_isFemale)
  plots$p5 <- gg_miss_var_cumsum(Xh)
  
  # Save the names of these 15 columns and visualise them
  high_miss <- missv[1:15,][[1]]; high_miss
  plots$p6 <- vis_miss(Xh[,high_miss])
  
  # We can also see the 15 columns overlapping in missingness quite a lot
  plots$p7 <- gg_miss_upset(Xh, nsets = 16)  
  return(plots)
}



