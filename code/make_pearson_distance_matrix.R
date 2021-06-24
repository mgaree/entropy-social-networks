setwd("C:/Users/Mike/Dropbox/01 Purdue Thesis work/04 Experimental Study for Paper 2")
set.seed(8675309)

rv = 5
inputmat <- npyLoad(paste("C:/Users/Mike/Dropbox/01 Purdue Thesis work/04 Experimental Study for Paper 2/data/split_master_trial_results_rv_", rv, ".npy", sep=""))

num_trials = 1800
distances_pearson <- matrix(NA, nrow=num_trials, ncol=num_trials)


tic("make pearson diss mtx")
for(i in 1:nrow(distances_pearson)) {
  # print(i)
  for(j in 1:ncol(distances_pearson)) {
    if (j < i) {
      distances_pearson[i, j] = cor(inputmat[i, ], inputmat[j, ])
    }
  }
}

distances_pearson <- distances_pearson + 1

toc()


save(distances_pearson, file=paste("pearson_rv_", rv, ".Rdata", sep=""))

