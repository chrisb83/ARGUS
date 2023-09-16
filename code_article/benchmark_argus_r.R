library(argus)

check_runtime <- function() {
  # -----------------------------------------------------------------------------
  # benchmark: generate ARGUS rvs by inversion using qgamma
  # -----------------------------------------------------------------------------
  
  print("benchmark: generate ARGUS rvs by inversion using qgamma")
  chi <- 1.0
  n_reps <- 21
  n <- 1.e6
  x <- vector("numeric", n_reps)
  
  for (i in seq_along(x)) {
    tic <- Sys.time()
    u <- runif(n)
    ub <- 0.5*chi**2
    my_c <- pgamma(ub, 1.5)
    r <- qgamma(my_c*u, 1.5)
    y <- sqrt(1 - r/ub)
    toc <- Sys.time() - tic
    x[i] <- toc
  }
  # drop first entry, it is distorted by the setup step
  x <- x[2:n_reps]
  print(c(min(x), median(x), mean(x), sd(x)))
  
  # check histogram
  # xv <- seq(0, 1, length.out = 100)
  # yv <- dargus(xv, chi)
  # hist(y, breaks = 20, freq = FALSE)
  # lines(xv, yv)
  
  # -----------------------------------------------------------------------------
  # varying parameter case for a large range of parameters
  # -----------------------------------------------------------------------------
  
  print("varying parameter case for a large range of parameters (0, 10)")
  chi_range <- c(0, 10)
  n_reps <- 51
  x <- vector("numeric", n_reps)
  for (i in seq_along(x)) {
    chiv <- chi_range[1] + (chi_range[2] - chi_range[1])*runif(n) # values in chi_range
    tic <- Sys.time()
    y <- rargus(chi=chiv, method='inversion')
    toc <- Sys.time() - tic
    x[i] = toc
  }
  x <- x[2:n_reps]
  print(c(min(x), median(x), mean(x), sd(x)))
  
  
  # -----------------------------------------------------------------------------
  # varying parameter case for various parameters (small range)
  # -----------------------------------------------------------------------------
  
  print("varying parameter case for various parameters ([0.99chi, 1.01chi])")
  chis <- c(1e-6, 0.0001, 0.005, 0.05, 0.5, 1.0, 2.5, 5, 10)
  n_reps <- 21
  for (chi in chis) {
    x <- vector("numeric", n_reps)
    print(chi)
    n <- 1.e6
    for (i in seq_along(x)) {
      chiv <- chi*(0.99 + 0.02*runif(n)) # values in (0.99*chi, 1.01*chi)
      tic <- Sys.time()
      y <- rargus(chi=chiv, method='inversion')
      toc <- Sys.time() - tic
      x[i] = toc
    }
    x <- x[2:n_reps]
    print(c(min(x), median(x), mean(x), sd(x)))
  }  
}

check_runtime()