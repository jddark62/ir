import numpy as np
import pandas as pd
from collections import defaultdict

# -----------------------
# Build adjacency structures
# -----------------------

def build_adj_from_edges(edges):
    """
    edges: list of (src, dst)
    Returns:
        nodes: sorted list of nodes
        adj_list: dict node -> list of outgoing neighbors
    """
    adj_list = defaultdict(list)
    nodes_set = set()
    for u, v in edges:
        adj_list[u].append(v)
        nodes_set.add(u)
        nodes_set.add(v)
    # Ensure all nodes appear in adj_list
    for n in list(nodes_set):
        adj_list.setdefault(n, [])
    nodes = sorted(nodes_set)
    return nodes, adj_list

def build_adj_matrix(nodes, adj_list):
    """
    nodes: list of nodes
    adj_list: dict node -> list of outgoing neighbors
    Returns:
        A: adjacency matrix (n x n), A[i, j] = 1 if node_i -> node_j
    """
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)
    for u, outs in adj_list.items():
        i = idx[u]
        for v in outs:
            j = idx[v]
            A[i, j] = 1.0
    return A, idx

# -----------------------
# PageRank via power iteration
# -----------------------

def pagerank(adj_list, damping=0.85, max_iter=100, tol=1e-6):
    """
    PageRank on adjacency list representation.
    adj_list: dict node -> list of outgoing neighbors
    Returns: dict node -> PageRank score
    """
    nodes = sorted(adj_list.keys())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    
    # Build column-stochastic matrix M
    M = np.zeros((n, n), dtype=float)  # M[j, i] = 1/outdeg(i) if edge i->j
    for u, outs in adj_list.items():
        i = idx[u]
        if len(outs) == 0:
            # Dangling node: distribute uniformly
            M[:, i] = 1.0 / n
        else:
            prob = 1.0 / len(outs)
            for v in outs:
                j = idx[v]
                M[j, i] = prob
    
    # Initialize ranks
    r = np.ones(n) / n
    teleport = np.ones(n) / n
    
    for _ in range(max_iter):
        r_new = damping * (M @ r) + (1 - damping) * teleport
        if np.linalg.norm(r_new - r, 1) < tol:
            r = r_new
            break
        r = r_new
    
    return {nodes[i]: float(r[i]) for i in range(n)}

# -----------------------
# HITS algorithm
# -----------------------

def hits(adj_list, max_iter=100, tol=1e-6):
    """
    HITS algorithm.
    adj_list: dict node -> list of outgoing neighbors
    Returns:
        auth_scores: dict node -> authority score
        hub_scores: dict node -> hub score
    """
    nodes = sorted(adj_list.keys())
    A, idx = build_adj_matrix(nodes, adj_list)  # A[i, j] = 1 if i->j
    n = len(nodes)
    
    h = np.ones(n)  # initial hubs
    a = np.ones(n)  # initial authorities
    
    for _ in range(max_iter):
        a_new = A.T @ h
        h_new = A @ a_new
        
        # Normalize
        if np.linalg.norm(a_new) > 0:
            a_new = a_new / np.linalg.norm(a_new)
        if np.linalg.norm(h_new) > 0:
            h_new = h_new / np.linalg.norm(h_new)
        
        if (np.linalg.norm(a_new - a, 1) < tol and
            np.linalg.norm(h_new - h, 1) < tol):
            a, h = a_new, h_new
            break
        
        a, h = a_new, h_new
    
    auth_scores = {nodes[i]: float(a[i]) for i in range(n)}
    hub_scores = {nodes[i]: float(h[i]) for i in range(n)}
    return auth_scores, hub_scores

# -----------------------
# Link analysis from in-memory edges
# -----------------------

def pagerank_hits_from_edges(edges, damping=0.85, max_iter=100, tol=1e-6):
    nodes, adj_list = build_adj_from_edges(edges)
    pr = pagerank(adj_list, damping=damping, max_iter=max_iter, tol=tol)
    auth, hub = hits(adj_list, max_iter=max_iter, tol=tol)
    return pr, auth, hub

# Example usage (NO CSV):
if __name__ == "__main__" and False:  # set to True to run
    edges = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'C'),
        ('C', 'A'),
        ('C', 'D'),
        ('D', 'C')
    ]
    pr, auth, hub = pagerank_hits_from_edges(edges)
    print("PageRank:", pr)
    print("Authority:", auth)
    print("Hub:", hub)

# -----------------------
# Link analysis from CSV
# -----------------------

def pagerank_hits_from_csv(csv_path, src_col="source", dst_col="target",
                           damping=0.85, max_iter=100, tol=1e-6):
    """
    csv_path: CSV with at least two columns: src_col, dst_col
    Each row represents a directed edge src -> dst.
    """
    df = pd.read_csv(csv_path)
    edges = list(zip(df[src_col], df[dst_col]))
    return pagerank_hits_from_edges(edges, damping=damping, max_iter=max_iter, tol=tol)

# Example usage (WITH CSV):
if __name__ == "__main__" and False:  # set to True to run
    pr, auth, hub = pagerank_hits_from_csv("graph_edges.csv", src_col="from", dst_col="to")
    print("PageRank:", pr)
    print("Authority:", auth)
    print("Hub:", hub)
