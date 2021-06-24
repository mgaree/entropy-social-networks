import numpy as np
from numpy import log2
import networkx as nx

from data_processors import relative_entropy_for_replication as re_p
from mycfun.cdataproc import relative_entropy_for_replication as re_c

from data_processors import mutual_information_for_replication as mi_p
from mycfun.cdataproc import mutual_information_for_replication as mi_c

from data_processors import transfer_entropy_for_replication as te_p
from mycfun.cdataproc import transfer_entropy_for_replication as te_c

np.random.seed(4321)

popsize = 100
timesteps = 500


data = np.random.uniform(-1, 1, (popsize, timesteps))  #start with 100n
bins = np.array([-1.        , -0.66666667, -0.33333333,  0.        ,
    0.33333333, 0.66666667,  1.0000003 ]) # the +EPS is built into the final edge

num_bins = len(bins) - 1

digi = np.digitize(data, bins).astype(np.int32)
digi -= 1

N, t_max = data.shape

output_data_p = np.zeros((N, t_max, 1))
output_data_c = np.zeros((N, t_max, 1))

G = nx.barabasi_albert_graph(popsize, 3, seed=4321)
G = nx.to_directed(G)

edges_as_nparray = np.array(G.edges)  # convenience for passing edge list to C
outdegrees_array = np.zeros(N, dtype=np.int32)
for i, v in G.out_degree:
    outdegrees_array[i] = v

# ############
# te_c(digi, num_bins, output_data_c, 0, edges_as_nparray, outdegrees_array)
# # # print('C', output_data_c[0, :, 0], output_data_c.shape)
# #
# te_p(data, bins, output_data_p, 0, G, (-1, 1) )
# # # print('python', output_data_p[0, :, 0], output_data_p.shape)
# #
# print(np.isclose(output_data_p[:, :, 0], output_data_c[:, :, 0]).all(), "!!!")

# %timeit te_p(data, bins, output_data_p, 0, G, (-1, 1) )
# %timeit te_c(digi, bins, output_data_c, 0, edges_as_nparray, outdegrees_array)

#
# dd = np.where(~np.isclose(output_data_c, output_data_p))[0:2]
# dd[0].size  # number of mismatches
# x = output_data_c[dd[0], dd[1], 0]
# y = output_data_p[dd[0], dd[1], 0]
# print(np.max(np.abs(x-y)))  # to get maximum "error"

#
# # ############
# mi_c(digi, num_bins, output_data_c, 0, edges_as_nparray, outdegrees_array)
# # print('C', output_data_c[0, :, 0], output_data_c.shape)
# #
# mi_p(data, bins, output_data_p, 0, G, (-1, 1) )
# # print('python', output_data_p[0, :, 0], output_data_p.shape)
#
# print(np.isclose(output_data_p[:, :, 0], output_data_c[:, :, 0]).all(), "!!!")
#
# # # %timeit mi_p(data, bins, output_data_p, 0, G, (-1, 1) )
# # # %timeit mi_c(digi, bins, output_data_c, 0, edges_as_nparray, outdegrees_array)


############
re_p(data, bins, output_data_p, 0, (-1, 1) )
# print(output_data_p[:, :, 0], output_data_p.shape)
#
re_c(digi, num_bins, output_data_c, 0)
# print(output_data_c[:, :, 0], output_data_c.shape)

print(np.isclose(output_data_p[:, :, 0], output_data_c[:, :, 0]).all(), "!!!")

# %timeit re_p(data, bins, output_data_p, 0, (-1, 1) )
# %timeit re_c(digi, bins, output_data_c, 0)
