library(RcppCNPy)
library(dtwclust)
library(tictoc)

setwd("C:/Users/Mike/Dropbox/01 Purdue Thesis work/04 Experimental Study for Paper 2")
set.seed(8675309)

rv = 5
inputmat <- npyLoad(paste("C:/Users/Mike/Dropbox/01 Purdue Thesis work/04 Experimental Study for Paper 2/data/split_master_trial_results_rv_", rv, ".npy", sep=""))


num_trials = 1800
distances_dtw <- matrix(NA, nrow=num_trials, ncol=num_trials)

tic("make dtw diss mtx")
for(i in 1:nrow(distances_dtw)) {
  print(i)
  for(j in 1:ncol(distances_dtw)) {
    if (j < i) {
      distances_dtw[i, j] = dtw_basic(inputmat[i, ], inputmat[j, ], norm="L2")
    }
  }
}
toc()

save(distances_dtw, file=paste("dtw_rv_", rv, ".Rdata", sep=""))




heatmap(outmat, Colv = NA, Rowv = NA, scale="column", na.rm=TRUE)
