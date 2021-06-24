setwd("C:/Users/Mike/Dropbox/01 Purdue Thesis work/04 Experimental Study for Paper 2")
library(NbClust)

lista.methods = c(
  "kl", "ch", "hartigan", "cindex", "db", "silhouette",
  "ratkowsky", "ball", "ptbiserial", "frey", "mcclain", "dunn",
  "sdindex", "sdbw"
  )

# methods that crashed - reason
# ccc, scott, marriot, trcovw, tracew, friedman, rubin - system is computationally singular: reciprocal condition number = 1.72585e-16
# duda, pseudot2, beale - Error in cutree(hc, k = best.nc) : object 'best.nc' not found

# the methods omitted by NbClust using the "alllong" option were omitted here as well (gamma, tau, ...)

rv = 0
mode = "pearson"

set.seed(867530900)

inputmat <- npyLoad(paste("C:/Users/Mike/Dropbox/01 Purdue Thesis work/04 Experimental Study for Paper 2/data/split_master_trial_results_rv_", rv, ".npy", sep=""))

# for DTW
load(paste(mode, "_rv_", rv, ".Rdata", sep=""))
diss_mat = distances_dtw  # the name 'distances_dtw' was saved in the file, except for rv 0?? :(

# for Pearson
load(paste(mode, "_rv_", rv, ".Rdata", sep=""))
diss_mat = distances_pearson

tabla = as.data.frame(matrix(ncol = 2, nrow = length(lista.methods)))


tic("NbClust loop")
for(i in 1:length(lista.methods)){
  # print(lista.methods[i])
  nb = NbClust::NbClust(inputmat, diss=as.dist(diss_mat), distance=NULL, method="ward.D", min.nc = 2, max.nc = 12,
                        index=lista.methods[i])

  tabla[i,2] = nb$Best.nc[1]
  tabla[i,1] = lista.methods[i]
}
tabla

# these are graphical index methods and i need to look at their output for knees
NbClust::NbClust(inputmat, diss=as.dist(diss_mat), distance=NULL, method="ward.D", min.nc = 2, max.nc = 12, index="hubert")
NbClust::NbClust(inputmat, diss=as.dist(diss_mat), distance=NULL, method="ward.D", min.nc = 2, max.nc = 12, index="dindex")
print("see Plots for hubert and dindex")

toc()

#############

# when i've viewed tabla and decided on a consensus number of clusters, use HAC and cut to the consensus number

k = 3  # consensus cut number

hclust_result <- hclust(as.dist(diss_mat), method="ward.D")
cut_result <- cutree(hclust_result, k=k)

# save sequence of cluster ids to file
write.table(as.data.frame(t(cut_result)),file=paste(mode, "_best_clusters_rv_", rv, ".csv", sep=""), quote=F,sep=",",row.names=F, col.names=F)

plot(hclust_result)
rect.hclust(hclust_result , k = k, border = 2:6, cluster=cut_result)

##############
# to plot clusters
for (i in unique(cut_result)) {
  matplot(t(inputmat[which(cut_result == i, arr.ind=TRUE),]), type="l")
}