
from __future__ import division

import networkx as nx
try:
    import scipy
    from scipy.io import loadmat
    from scipy.sparse import lil_matrix
    import scipy.linalg
    SCIPY_LOADED = True
except:
    print "Scipy is missing from this system, this makes invariants [XYZ] unavailable"
    SCIPY_LOADED = False
import sys


def size(G):
    """Size of a graph G is just the number of edges"""
    return G.number_of_edges()

def weighted_size(G):
    """Weighted size is the sum of the weights on all the edges in the graph."""
    weighted_size = 0
    for edge in G.edges(data=True):
        weighted_size += edge[2]['weight']
    return weighted_size

def max_degree(G):
    """Return the degree of the vertex with the highest degree in the graph"""
    return max(G.degree().values())

def degree_distribution(G, return_dictionary=False):
    """Return the distribution of degrees of vertices in the graph.
    If return_dictionary is specified, return a dictionary indexed
    by vertex name instead of a list of values (the default).
    """
    if return_dictionary:
        return G.degree()
    else:
        return G.degree().values()

def vertex_adjacent_weight(v, G):
    """Returns a sum of the weights on the edges adjacent to v."""
    return sum([ e[2]['weight'] for e in G.edges(data=True)])

def weighted_max_degree(G):
    """Return the sum of the weights of the edges adjacent to v where v
    is the vertex with the most weight adjacent to it"""
    weighted_degrees = [vertex_adjacent_weight(v, G) for v in G.nodes()]
    return max(weighted_degrees)
            
def MADg(G):
    """Greedy approximation of Maximum Average Degree (MADg) is a lower bound on the
    true maximum average degree.
    For an explanation of this invariant, see Pao, Coppersmith, and Priebe 2011.
    """
    vertex_list = G.nodes()
    average_degrees = []
    while len(vertex_list) > 1:
        Gp = G.subgraph(vertex_list)
        degrees = Gp.degree()

        average_degree = sum(degrees.values()) / len(degrees)
        average_degrees.append( average_degree)
        
        #Sort the vertices according to degrees
        e = degrees.keys()
        e.sort(cmp=lambda a,b: cmp(degrees[a],degrees[b]))
        vertex_list.remove(e[0]) #vertex with smallest degree
    MADg = max(average_degrees)
    return MADg

def MAWg(G):
    """Greedy approximation of the maximum average weight. We don't yet know
    how this relates to the true MAW, but we conjecture it is a lower bound,
    since MADg is a lower bound on MAD, and the procedure is very similar. --GAC
    """
    vertex_list = G.nodes()
    average_weights = []
    while len(vertex_list) > 1:
        Gp = G.subgraph(vertex_list)
        adjacent_weights = {}
        for v in Gp.nodes():
            adjacent_weights[v] = vertex_adjacent_weight(v, Gp)

        average_weight = sum(adjacent_weights.values()) / len(adjacent_weights)
        average_weights.append( average_weight)
        
        #Sort the vertices according to degrees
        e = adjacent_weights.keys()
        e.sort(cmp=lambda a,b: cmp(adjacent_weights[a],adjacent_weights[b]))
        vertex_list.remove(e[0]) #vertex with smallest amount of weight adjacent to it
    MAWg = max(average_weights)
    return MAWg

def MADe(G):
    """Eigenvalue approximation of maximum average degree. This is an upper bound
    on MAD. For further explanation, see Pao, Coppersmith, and Priebe 2011.
    """
    mtx = nx.adj_matrix(G)
    #Must convert the adjaceny matrix to be binary-valued only.
    threshold_at = 0 #Only semi-principledly selected. A better biologically-inspired value may be worthwhile --GAC
    for row in range(len(mtx)):
        for col in range(len(mtx)):
            if mtx.item((row,col)) > threshold_at:
                mtx.itemset((row,col),1)
    eigenvalues, eigenvectors = scipy.linalg.eig( mtx )
    MADe = float(max(eigenvalues))
    return MADe

def MAWe(G):
    """Eigenvalue approximation of maximum average degree. This is conjectured
    to be an upper bound on MAW.
    """
    eigenvalues, eigenvectors = scipy.linalg.eig(nx.adj_matrix(G) )
    MAWe = float(max(eigenvalues))
    return MAWe

def scan1(G):
    """This is the maximum locality statistic of Priebe et al. 2005.
    Also see Pao, Coppersmith, and Priebe 2011 for further information."""
    max_scan_stat = -1
    for vertex in G.nodes():
        neighborhood = G.neighbors(vertex) + [vertex]
        induced_subgraph = G.subgraph( neighborhood )
        this_scan_size = induced_subgraph.number_of_edges()
        if this_scan_size > max_scan_stat:
            max_scan_stat = this_scan_size
    return max_scan_stat

def scan1_distribution(G, return_dictionary=False):
    """This returns the locality statistic of Priebe et al. 2005 for each
    vertex in graph G, as used in Borges, Coppersmith, Meyer, and Priebe 2011.
    If return_dictionary is specified, we return a dictionary indexed by
    vertex name, rather than just the values (as returned by default).
    """
    def locality_statistic(v):
        neighborhood = G.neighbors(v) + [v]
        induced_subgraph = G.subgraph( neighborhood )
        return induced_subgraph.number_of_edges()
    if return_dictionary:
        return_dict = {}
        for v in G.nodes():
            return_dict[v] = locality_statistic(v)
        return return_dict
    else:
        return map( locality_statistic, G.nodes())
    
def weighted_scan1(G):
    """This is a weighted variant of the maximum locality statistic of
    Priebe et al. 2005. We sum the weights of the edges contained in
    $\Omega(N_1[v])$, the induced subgraph of the 1-hop neighborhood
    of vertex v. We return the maximum value of this statistic over all v.
    """
    max_scan_stat = -1
    for vertex in G.nodes():
        neighborhood = G.neighbors(vertex) + [vertex]
        induced_subgraph_edges = G.subgraph( neighborhood ).edges(data=True)
        this_scan_weight = sum([ edge[2]['weight'] for edge in induced_subgraph_edges])
        if this_scan_weight > max_scan_stat:
            max_scan_stat = this_scan_weight
    return max_scan_stat

def weighted_scan1_distribution(G, return_dictionary=False):
    """This returns a distribution of the a weighted variant of the
    maximum locality statistic of Priebe et al. 2005. We sum the weights
    of the edges contained in $\Omega(N_1[v])$, the induced subgraph of
    the 1-hop neighborhood of vertex v.
    We return the distrubtion, amenable to applications similar to
    Borges, Coppersmith, Meyer, and Priebe 2011.
    If return_dictionary is specified, we return a dictionary indexed by
    vertex name, rather than just the values (as returned by default).
    """
    def weighted_locality_statistic(v):
        neighborhood = G.neighbors(v) + [v]
        induced_subgraph_edges = G.subgraph( neighborhood ).edges(data=True)
        return sum([ edge[2]['weight'] for edge in induced_subgraph_edges])
    if return_dictionary:
        return_dict = {}
        for v in G.nodes():
            return_dict[v] = weighted_locality_statistic(v)
        return return_dict
    else:
        return map( weighted_locality_statistic, G.nodes())
    
def triangles(G):
    """This counts the number of total triangles (paths of length three)
    present in graph G.
    """
    return sum(nx.triangles(G).values())

def triangles_distribution(G, return_dictionary=False):
    """This returns a distribution of the number of triangles each
    vertex in G is involved in, amenable to applications similar to
    Borges, Coppersmith, Meyer, and Priebe 2011.
    If return_dictionary is specified, we return a dictionary indexed by
    vertex name, rather than just the values (as returned by default).
    """
    if return_dictionary:
        return nx.triangles(G)
    else:
        return nx.triangles(G).values()
    
def clustering_coefficient(G):
    """Returns the average clustering coefficient over all vertices
    in graph G."""
    return nx.average_clustering( G )

def clustering_coefficient_distribution(G, return_dictionary=False):
    """Returns the distribution of clustering coefficients, amenable
    to applications similar to Borges, Coppersmith, Meyer, and Priebe 2011.
    If return_dictionary is specified, we return a dictionary indexed by
    vertex name, rather than just the values (as returned by default).
    """
    if return_dictionary:
        return nx.clustering(G)
    else:
        return nx.clustering(G).values()

def average_shortest_path_length(G):
    """Return the average shortest path length between all pairs of
    vertices (twice-averaged).
    """
    try:
        return nx.average_shortest_path_length( G )
    except nx.exception.NetworkXError:
        print "Graph is not connected, return 2*n"
        return 2*len(G.nodes())

def average_shortest_path_length_distribution(G, return_dictionary=False):
    """Return a distribution of the average shortest path lengths between
    all pairs of vertices (averaged only once).
    If return_dictionary is specified, we return a dictionary indexed by
    vertex name, rather than just the values (as returned by default).
    """
    shortest_path_length = nx.shortest_path_length(G)
    return_dict = {}
    #print "-->",shortest_path_length.values()
    num_vertices = len(G.nodes())
    for v,apls in shortest_path_length.items():
        return_dict[v] = sum(apls)/num_vertices
    if return_dictionary:
        return return_dict
    else:
        return return_dict.values()

def dijkstra_path(G):
    """Return the shortest Dijkstra (weighted) path between all pairs of
    vertices (averaged twice)
    """
    dijkstra_path = nx.all_pairs_dijkstra_path_length(G)
    summed_dpv = [sum(dpv.values()) for dpv in dijkstra_path.values()]
    average_dijkstra_path = sum(summed_dpv)/ (len(G.nodes())**2)
    return average_dijkstra_path

def dijkstra_path_distribution(G, return_dictionary=False):
    """Return a distribution of the shortest Dijkstra (weighted) path between
    all pairs of vertices (averaged only once).
    If return_dictionary is specified, we return a dictionary indexed by
    vertex name, rather than just the values (as returned by default).
    """
    dijkstra_paths = nx.all_pairs_dijkstra_path_length(G)
    return_dict = {}
    num_vertices = len(G.nodes())
    for v,dijkstras in dijkstra_paths.items():
        return_dict[v] = sum(dijkstras)/num_vertices
    if return_dictionary:
        return return_dict
    else:
        return return_dict.values()

def closeness_centrality(G):
    """Return average unweighted closeness centrality."""
    return sum(nx.closeness_centrality(G).values())/len(G.nodes())

def closeness_centrality_distribution(G, return_dictionary=False):
    """Return a distribution of unweighted closeness centralities, as used in
    Borges, Coppersmith, Meyer, and Priebe 2011.
    If return_dictionary is specified, we return a dictionary indexed by
    vertex name, rather than just the values (as returned by default).
    """
    if return_dictionary:
        return nx.closeness_centrality(G)
    else:
        return nx.closeness_centrality(G).values()

def betweenness_centrality(G):
    """Return average unweighted betweenness centrality."""
    return sum(nx.betweenness_centrality(G).values())/len(G.nodes())

def betweenness_centrality_distribution(G, return_dictionary=False):
    """Return a distribution of unweighted betweenness centralities, 
    as used in Borges, Coppersmith, Meyer, and Priebe 2011.
    If return_dictionary is specified, we return a dictionary indexed by
    vertex name, rather than just the values (as returned by default).
    """
    if return_dictionary:
        return nx.betweenness_centrality(G)
    else:
        return nx.betweenness_centrality(G).values()

def weighted_betweenness_centrality(G):
    """Return a weighted version of betweenness centrality."""
    return sum(nx.betweenness_centrality(G, weighted_edges=True).values())/len(G.nodes())

def weighted_betweenness_centrality_distribution(G, return_dictionary=False):
    """Return a distribution of weighted betweenness centralities.
    If return_dictionary is specified, we return a dictionary indexed by
    vertex name, rather than just the values (as returned by default).
    """
    if return_dictionary:
        return nx.betweenness_centrality(G, weighted_edges=True)
    else:
        return nx.betweenness_centrality(G, weighted_edges=True).values()

def get_single_valued_invariants():
    if SCIPY_LOADED:
        single_valued_invariants = [size, weighted_size, max_degree, weighted_max_degree,
                                    MADg, MAWg, MADe, MAWe, scan1, weighted_scan1,
                                    triangles, clustering_coefficient, average_shortest_path_length,
                                    dijkstra_path,closeness_centrality, betweenness_centrality,
                                    weighted_betweenness_centrality]
    else:
        single_valued_invariants = [size, weighted_size, max_degree, weighted_max_degree,
                                    MADg, MAWg, scan1, weighted_scan1,
                                    triangles, clustering_coefficient, average_shortest_path_length,
                                    dijkstra_path,closeness_centrality, betweenness_centrality,
                                    weighted_betweenness_centrality]
    return single_valued_invariants

def get_ditributional_invariants():
    if SCIPY_LOADED:
        distribution_invariants = [degree_distribution, scan1_distribution,
                                   weighted_scan1_distribution, triangles_distribution,
                                   clustering_coefficient_distribution,
                                   average_shortest_path_length_distribution,
                                   dijkstra_path_distribution,
                                   closeness_centrality_distribution,
                                   betweenness_centrality_distribution,
                                   weighted_betweenness_centrality_distribution]
    else:
        distribution_invariants = [degree_distribution, scan1_distribution,
                                   weighted_scan1_distribution, triangles_distribution,
                                   clustering_coefficient_distribution,
                                   average_shortest_path_length_distribution,
                                   dijkstra_path_distribution,
                                   closeness_centrality_distribution,
                                   betweenness_centrality_distribution,
                                   weighted_betweenness_centrality_distribution]
    return distributional_invariants
        
if __name__=='__main__':
    """Assumedly, if you're running this standalone, it's to test it..."""


    from erz_random_graphs import erdos_renyi_zipf_graph

    G = erdos_renyi_zipf_graph(50,0.1)

    for fun in single_valued_invariants:
        print fun, fun(G)

    for fun in distribution_invariants:
        temp = fun(G)
        #print temp
        print fun, len(temp), temp[:3]
        
