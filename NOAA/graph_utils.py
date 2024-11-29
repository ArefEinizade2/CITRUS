import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.linalg import circulant
from scipy.sparse import kron
from scipy import sparse

zeroTolerance = 1e-9  # Values below this number are considered zero.

def build_time_graph(window: int, directed: bool, cyclic: bool):
    """
    Circulant matrix as in https://arxiv.org/pdf/1712.00468.pdf (eq. 7)
    """
    if window <= 1:
        raise Exception("Ehm..")
    adjacency = circulant([0, 1] + [0 for _ in range(window-2)])
    if not cyclic:
        adjacency[0, window-1] = 0
    if not directed:
        adjacency += adjacency.transpose()

    # return adjacency.transpose()
    return adjacency




def build_parametric_product_graph(S_0, S_1, h_00, h_01, h_10, h_11):
    I_0 = np.eye(S_0.shape[1])
    I_1 = np.eye(S_1.shape[1])

    S_kron_II = torch.from_numpy(np.kron(I_0, I_1))
    S_kron_SI = torch.from_numpy(np.kron(S_0, I_1))
    S_kron_IS = torch.from_numpy(np.kron(I_0, S_1))
    S_kron_SS = torch.from_numpy(np.kron(S_0, S_1)).double()

    S = h_00 * S_kron_II + \
        h_01 * S_kron_IS + \
        h_10 * S_kron_SI + \
        h_11 * S_kron_SS
    return S


def permutation_by_degree(S):
    """
    Function taken by Fernando Gama's repository: https://github.com/alelab-upenn/graph-neural-networks
    and slightly modified

    permDegree: determines the permutation by degree (nodes ordered from highest
        degree to lowest)

    Input:
        S (np.array): matrix

    Output:
        permS (np.array): matrix permuted
        order (list): list of indices to permute S to turn into permS.
    """
    assert len(S.shape) == 2
    assert S.shape[0] == S.shape[1]
    assert type(S) == np.ndarray

    # Compute the degree
    d = np.sum(S, axis=1)
    # Sort ascending order (from min degree to max degree)
    order = np.argsort(d)
    # Reverse sorting
    order = np.flip(order, 0)
    # And update S
    S = S[order, :][:, order]

    return S, order.tolist()


def computeNeighborhood(S, K, n_active_nodes_out, n_active_nodes_neighborhood, outputType):
    """
    Function taken by Fernando Gama's repository: https://github.com/alelab-upenn/graph-neural-networks
    and slightly modified

    computeNeighborhood: compute the K-hop neighborhood of a graph

        computeNeighborhood(W, K, n_active_nodes_out = 'all', n_active_nodes_neighborhood = 'all', outputType = 'list')

    Input:
        S (np.array): adjacency matrix
        K (int): K-hop neighborhood to compute the neighbors
        n_active_nodes_out (int or 'all'): how many nodes (from top) to compute the neighbors
            from (default: 'all').
        n_active_nodes_neighborhood (int or 'all'): how many nodes to consider valid when computing the
            neighborhood (i.e. nodes beyond n_active_nodes_neighborhood are not trimmed out of the
            neighborhood; note that nodes smaller than n_active_nodes_neighborhood that can be reached
            by nodes greater than n_active_nodes_neighborhood, are included. default: 'all')
        outputType ('list' or 'matrix'): choose if the output is given in the
            form of a list of arrays, or a matrix with zero-padding of neighbors
            with neighborhoods smaller than the maximum neighborhood
            (default: 'list')

    Output:
        neighborhood (np.array or list): contains the indices of the neighboring
            nodes following the order established by the adjacency matrix.
    """
    assert outputType == 'list' or outputType == 'matrix'
    assert len(S.shape) == 2

    # In this case, if it is a 2-D array, we do not need to add over the
    # edge dimension, so we just sparsify it
    assert S.shape[0] == S.shape[1]
    S = sparse.coo_matrix((S > zeroTolerance).astype(S.dtype))
    # Now, we finally have a sparse, binary matrix, with the connections.
    # Now check that K and n_active_nodes_out are correct inputs.
    # K is an int (target K-hop neighborhood)
    # n_active_nodes_out is either 'all' or an int determining how many rows
    assert K >= 0  # K = 0 is just the identity
    # Check how many nodes we want to obtain
    if n_active_nodes_out == 'all':
        n_active_nodes_out = S.shape[0]
    if n_active_nodes_neighborhood == 'all':
        n_active_nodes_neighborhood = S.shape[0]
    assert 0 <= n_active_nodes_out <= S.shape[0]  # Cannot return more nodes than there are
    assert 0 <= n_active_nodes_neighborhood <= S.shape[0]

    # All nodes are in their own neighborhood, so
    allNeighbors = [[n] for n in range(S.shape[0])]
    # Now, if K = 0, then these are all the neighborhoods we need.
    # And also keep track only about the nodes we care about
    neighbors = [[n] for n in range(n_active_nodes_out)]
    # But if K > 0
    if K > 0:
        # Let's start with the one-hop neighborhood of all nodes (we need this)
        nonzeroS = list(S.nonzero())
        # This is a tuple with two arrays, the first one containing the row
        # index of the nonzero elements, and the second one containing the
        # column index of the nonzero elements.
        # Now, we want the one-hop neighborhood of all nodes (and all nodes have
        # a one-hop neighborhood, since the graphs are connected)
        for n in range(len(nonzeroS[0])):
            # The list in index 0 is the nodes, the list in index 1 is the
            # corresponding neighbor
            allNeighbors[nonzeroS[0][n]].append(nonzeroS[1][n])
        # Now that we have the one-hop neighbors, we just need to do a depth
        # first search looking for the one-hop neighborhood of each neighbor
        # and so on.
        oneHopNeighbors = allNeighbors.copy()
        # We have already visited the nodes themselves, since we already
        # gathered the one-hop neighbors.
        visitedNodes = [[n] for n in range(n_active_nodes_out)]
        # Keep only the one-hop neighborhood of the ones we're interested in
        neighbors = [list(set(allNeighbors[n])) for n in range(n_active_nodes_out)]
        # For each hop
        for k in range(1, K):
            # For each of the nodes we care about
            for i in range(n_active_nodes_out):
                # Take each of the neighbors we already have
                node_neighbors = neighbors[i].copy()
                for j in node_neighbors:
                    # and if we haven't visited those neighbors yet
                    if j not in visitedNodes[i]:
                        # Just look for our neighbor's one-hop neighbors and
                        # add them to the neighborhood list
                        neighbors[i].extend(oneHopNeighbors[j])
                        # And don't forget to add the node to the visited ones
                        # (we already have its one-hope neighborhood)
                        visitedNodes[i].append(j)
                # And now that we have added all the new neighbors, we just
                # get rid of those that appear more than once
                neighbors[i] = list(set(neighbors[i]))

    # Now that all nodes have been collected, get rid of those beyond n_active_nodes_neighborhood
    for i in range(n_active_nodes_out):
        # Get the neighborhood
        thisNeighborhood = neighbors[i].copy()
        # And get rid of the excess nodes
        neighbors[i] = [j for j in thisNeighborhood if j < n_active_nodes_neighborhood]

    if outputType == 'matrix':
        # List containing all the neighborhood sizes
        neighborhoodSizes = [len(x) for x in neighbors]
        # Obtain max number of neighbors
        maxNeighborhoodSize = max(neighborhoodSizes)
        # then we have to check each neighborhood and find if we need to add
        # more nodes (itself) to pad it so we can build a matrix
        paddedNeighbors = []
        for n in range(n_active_nodes_out):
            paddedNeighbors += [np.concatenate(
                (neighbors[n],
                 n * np.ones(maxNeighborhoodSize - neighborhoodSizes[n]))
            )]
        # And now that every element in the list paddedNeighbors has the same
        # length, we can make it a matrix
        neighbors = np.array(paddedNeighbors, dtype=np.int32)

    return neighbors


def create_connected_undirected_sbm_graph(sizes: list, probs: np.array, verbose: bool = False) -> nx.Graph:
    G = nx.stochastic_block_model(sizes, probs)
    if verbose:
        print(f"G is connected: {nx.is_connected(G)}")
        print(f"G is undirected: {not nx.is_directed(G)}")
    while not nx.is_connected(G) and not nx.is_directed(G):
        G = nx.stochastic_block_model(sizes, probs)
        if verbose:
            print("Recomputing graph ...")
            print(f"G is connected: {nx.is_connected(G)}")
            print(f"G is undirected: {not nx.is_directed(G)}")
    return G


def plot_sbm_graph(G: nx.Graph, sizes: list, title: str) -> None:
    colors = []
    for i, _ in enumerate(sizes):
        colors += [(i + 1) / 10 for _ in range(sizes[i])]
    nx.draw_networkx(G, with_labels=True, node_color=colors)
    plt.title(title)
    plt.show()


def compute_normalized_adjacency_matrix(graph: nx.Graph) -> np.array:
    adj_mat = nx.adjacency_matrix(graph)
    adj_spectrum = nx.adjacency_spectrum(graph)

    gso = adj_mat / max(np.absolute(adj_spectrum))
    return gso
