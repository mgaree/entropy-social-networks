# -*- coding: utf-8 -*-
"""Helper methods for preparing networks (graphs) for experimental design."""

from copy import deepcopy

import numpy as np
import networkx as nx


def to_directed_modified(G, rg):
    """Convert undirected network to directed in a non-default way.

    Some NetworkX generators make undirected graphs, so here we alter the native
    `to_directed` method: for each edge {u, v}, we uniformly at random replace
    it with one of (u, v), (v, u), or both (u, v) and (v, u). This contrasts with
    the native method which always replaces {u, v} with both (u, v) and (v, u).

    Args:
        G (networkx.Graph): undirected graph to be converted
        rg (RandomGenRandomInterface or similar): random number generator for selecting what to do with each
            existing edge; must have a .random(N) method

    """
    H = nx.DiGraph()
    H.graph.update(deepcopy(G.graph))
    H.add_nodes_from((n, deepcopy(d)) for n, d in G._node.items())

    edges = list(G.edges)
    new_edges = list()

    rands = rg.random(len(edges))  # generating array of random numbers faster than one at a time

    for i, (u, v) in enumerate(edges):
        r = rands[i]
        if r < 0.33:
            # keep the edge as-is (u, v)
            pass
        elif r < 0.66:
            # replace the edge with (v, u)
            edges[i] = (v, u)
        else:
            # keep (u, v) and create new (v, u)
            new_edges.append((v, u))

    H.add_edges_from(edges + new_edges)
    return H


def erdos_renyi_random(n, seed):
    """Return an ER-random graph.

    NetworkX has two algorithms for ER-random graphs: erdos_renyi_graph and fast_gnp_random_graph.
    The first one runs in O(n^2) time and the second runs in O(n + m) time, where n is number of nodes
    and m is number of edges. For low density graphs (low values of parameter p), fast_gnp_random_graph
    is faster.

    After encountering extremely long network construction times for trials with 10,000 agents using
    ER random network structure (~5 minutes per replication), we switched to fast_gnp_random_graph
    (~700ms per replication). Time savings were realized for the other population sizes, as well.

    """
    p = np.log(n) / n  # density chosen as percolation threshold; np.log = natural log
    # the directed flag causes all directed edges to be considered, not every included edge to be bi-directional
    G = nx.fast_gnp_random_graph(n, p, seed, directed=True)
    G.name = f"erdos_renyi({n})"  # including p in the name just makes things messy
    return G


def small_world(n, p, k, seed):
    """Return a small-world graph."""
    G = nx.watts_strogatz_graph(n, k, p, seed)
    G.name = f"small_world({n}, {p}, {k})"
    return G


def scale_free(n, m, seed):
    """Return a scale-free Barabasi-Albert graph (preferential attachment)."""
    G = nx.barabasi_albert_graph(n, m, seed)
    G.name = f"scale_free({n}, {m})"
    return G


def prepare_graph_for_trial(G, rg):
    """Given a graph, make it directed and free of self-loops, then return largest component.

    Args:
        G (networkx.Graph or DiGraph): graph to prepare
        rg (RandomGenRandomInterface or similar): random number generator with a .random(N) method
            to produce an array of uniformly distributed random numbers

    Returns:
        networkx.DiGraph

    """
    # Some generators don't support directed graphs as built-in, so I fix that
    if not nx.is_directed(G):
        G = to_directed_modified(G, rg)

    # Preserve only largest weakly connected component and remove self-loops
    G.remove_edges_from(G.selfloop_edges())

    largest_cc = max(nx.weakly_connected_components(G), key=len)
    GG = G.subgraph(largest_cc).copy()  # copy() makes it a graph instead of a graphView
    GG = nx.convert_node_labels_to_integers(G)  # want node labels to be [0..N] to align with agent unique_ids

    return GG
