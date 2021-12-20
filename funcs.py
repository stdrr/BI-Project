from collections import defaultdict
from networkx.drawing.layout import _process_params
import pandas as pd
import numpy as np
import networkx as nx
import markov_clustering as mc
<<<<<<< HEAD
import os
import re
from sklearn.metrics import recall_score, precision_score, f1_score, ndcg_score
=======
import imported_code.diamond as diamond
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
>>>>>>> main


def human_data(file='data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt',
               save_file='data/homo_preprocess.tsv'):
    df = pd.read_csv(file, sep='\t', low_memory=False)
    our_df = df[(df['Organism ID Interactor A'] == 9606) & (df['Organism ID Interactor B'] == 9606)]
    our_df = our_df[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']]
    our_df = our_df.drop_duplicates()
    our_df = our_df[our_df['Entrez Gene Interactor A'] != our_df['Entrez Gene Interactor B']]
    interactome_g = nx.from_pandas_edgelist(our_df, source='Entrez Gene Interactor A',
                                            target='Entrez Gene Interactor B')
    list_con_comp = sorted(nx.connected_components(interactome_g), key=len, reverse=True)
    lcc = interactome_g.subgraph(list_con_comp[0])
    our_list = nx.to_edgelist(lcc)
    our_df = pd.DataFrame(our_list, columns=['Entrez Gene Interactor A', 'Entrez Gene Interactor B', 'rem'])
    our_df = our_df[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']]
    our_df.to_csv(save_file, sep='\t', index=False)


def disease_data(file='data/curated_gene_disease_associations.tsv',
                 disease_id='C0003873'):
    df = pd.read_csv(file, sep='\t')
    curated_df = df[df['diseaseId'].str.contains(disease_id)]
    curated_df.to_csv('data/disease'+disease_id+'.tsv', sep='\t', index=False)


def data_overview(human_file, disease_file):
    curated_df = pd.read_csv(disease_file, sep='\t')
    interactome_df = pd.read_csv(human_file, sep='\t')
    interactome_g = nx.from_pandas_edgelist(interactome_df, source='Entrez Gene Interactor A',
                                            target='Entrez Gene Interactor B')
    curated_g = interactome_g.subgraph(curated_df['geneId'].to_list())
    list_con_comp = sorted(nx.connected_components(curated_g), key=len, reverse=True)
    lcc = curated_g.subgraph(list_con_comp[0])
    print('Number of genes associated with the desease:', curated_df['geneId'].nunique())
    print('Classes of the desease:', curated_df['diseaseClass'].unique())
    print('Number of genes present in the interactome:', curated_g.number_of_nodes())
    print('Largest connected component:', lcc.number_of_nodes())


def MCL_hyper(human_file, start=18, end=27):
    interactome_df = pd.read_csv(human_file, sep='\t')
    interactome_g = nx.from_pandas_edgelist(interactome_df, source='Entrez Gene Interactor A',
                                            target='Entrez Gene Interactor B')
    matrix = nx.to_scipy_sparse_matrix(interactome_g)
    for inflation in [i / 10 for i in range(start, end)]:
        result = mc.run_mcl(matrix, inflation=inflation)
        clusters = mc.get_clusters(result)
        Q = mc.modularity(matrix=result, clusters=clusters)
        print("inflation:", inflation, "modularity:", Q)


<<<<<<< HEAD
def MCL(human_file, inflation=1.8):
    interactome_df = pd.read_csv(human_file, sep='\t')
    interactome_g = nx.from_pandas_edgelist(interactome_df, source='Entrez Gene Interactor A',
                                            target='Entrez Gene Interactor B')
    matrix = nx.to_scipy_sparse_matrix(interactome_g)
    result = mc.run_mcl(matrix, inflation=inflation)
    clusters = mc.get_clusters(result)

    return clusters


def diffusion_heat(path='diffusion_heat/', top=50):

    dict_ = {}

    for i in os.listdir(path):
        df = pd.read_csv(path + i)
        df = df[df['diffusion_input'] != 1.0]
        df.sort_values(by=['diffusion_output_rank'], inplace=True)
        dict_[re.findall(r'\d+', i)[0]] = df['name'][:top].tolist()

    return dict_


def metrics(test_set, ground_truth):

    metrics = {}

    metrics['recall'] = recall_score(ground_truth, test_set, average='samples')
    metrics['precision'] = precision_score(ground_truth, test_set, average='samples')
    metrics['f1score'] = f1_score(ground_truth, test_set, average='samples')
    metrics['ndcg'] = ndcg_score(ground_truth, test_set, average='samples')

    return metrics


=======
def DIAMOND(network_file, seed_file, n, alpha=1, out_file='data/first_n_added_nodes_weight_alpha.txt'):
    """
    Code taken and adapted from https://github.com/dinaghiassian/DIAMOnD.git
>>>>>>> main

    :param network_file: file providing the network in the form of an edgelist
    :param seed_file: file with the seed genes (if table contains more than one
                      column they must be tab-separated; the first column will be used only)
    :param n: number of desired iterations
    :param alpha: seeds weight (integer), default value is 1
    :param out_file: name for the results file
    """
    
    # infer n
    if n == -1:
        network_df = pd.read_csv(network_file, sep=',', header=None)
        n = pd.concat([network_df.iloc[:,0], network_df.iloc[:,1]]).nunique()
        del network_df

    # check if input style is correct
    input_list = [network_file, seed_file, n, alpha, out_file] 
    network_edgelist_file, seeds_file, max_number_of_added_nodes, alpha, outfile_name = diamond.check_input_style(input_list)

    # read the network and the seed genes:
    G_original, seed_genes = diamond.read_input(network_edgelist_file, seeds_file)

    # run DIAMOnD
    added_nodes = diamond.DIAMOnD(G_original,
                          seed_genes,
                          max_number_of_added_nodes, alpha,
                          outfile=outfile_name)

    print("\n results have been saved to '%s' \n" % outfile_name) 


# ########################################################
# #
# #            DIABLE ALGORITHM
# #
# ########################################################


def get_diable_universe(candidates, cluster_nodes, neighbors):
    """
    Compute the DiaBLE universe as the union of the seed genes, 
    the genes that have at least one link to the seed set (candidate genes) 
    and their first neighbors.

    :param candidates: candidate genes at iteration k
    :param cluster_nodes: seed genes at iteration k
    :param neighbors: precomputed dictionary of all the nodes in the graph and their neighbors

    :return DiaBLE universe at iteration k
    """
    candidate_neighbors = set()
    for candidate in candidates:
        candidate_neighbors |= set(neighbors[candidate])
    return cluster_nodes | candidates | candidate_neighbors


def diable_iteration_of_first_X_nodes(G, S, X, alpha):
    """
    Parameters:
    ----------
    - G:     graph
    - S:     seeds
    - X:     the number of iterations, i.e only the first X gened will be
             pulled in
    - alpha: seeds weight
    Returns:
    --------

    - added_nodes: ordered list of nodes in the order by which they
      are agglomerated. Each entry has 4 info:
      * name : dito
      * k    : degree of the node
      * kb   : number of +1 neighbors
      * p    : p-value at agglomeration
    """

    N = G.number_of_nodes()

    added_nodes = []

    # ------------------------------------------------------------------
    # Setting up dictionaries with all neighbor lists
    # and all degrees
    # ------------------------------------------------------------------
    neighbors, all_degrees = diamond.get_neighbors_and_degrees(G)

    # ------------------------------------------------------------------
    # Setting up initial set of nodes in cluster
    # ------------------------------------------------------------------

    cluster_nodes = set(S)
    not_in_cluster = set()
    s0 = len(cluster_nodes)

    s0 += (alpha - 1) * s0
    N += (alpha - 1) * s0

    # ------------------------------------------------------------------
    # precompute the logarithmic gamma functions
    # ------------------------------------------------------------------
    gamma_ln = diamond.compute_all_gamma_ln(N + 1)

    # ------------------------------------------------------------------
    # Setting initial set of nodes not in cluster
    # ------------------------------------------------------------------
    for node in cluster_nodes:
        not_in_cluster |= neighbors[node]
    not_in_cluster -= cluster_nodes

    # ------------------------------------------------------------------
    #
    # M A I N     L O O P
    #
    # ------------------------------------------------------------------

    all_p = {}

    while len(added_nodes) < X:

        # ------------------------------------------------------------------
        #
        # Going through all nodes that are not in the cluster yet and
        # record k, kb and p
        #
        # ------------------------------------------------------------------

        info = {}

        ##############################################################################
        #
        #                    DiaBLE Universe Computation
        #
        ##############################################################################
        diable_universe = get_diable_universe(not_in_cluster, cluster_nodes, neighbors)
        N = len(diable_universe)

        ##############################################################################

        pmin = 10
        next_node = 'nix'
        reduced_not_in_cluster = diamond.reduce_not_in_cluster_nodes(all_degrees,
                                                                     neighbors, G,
                                                                     not_in_cluster,
                                                                     cluster_nodes, alpha)

        for node, kbk in reduced_not_in_cluster.items():
            # Getting the p-value of this kb,k
            # combination and save it in all_p, so computing it only once!
            kb, k = kbk
            try:
                p = all_p[(k, kb, s0)]
            except KeyError:
                p = diamond.pvalue(kb, k, N, s0, gamma_ln)
                all_p[(k, kb, s0)] = p

            # recording the node with smallest p-value
            if p < pmin:
                pmin = p
                next_node = node

            info[node] = (k, kb, p)

        # ---------------------------------------------------------------------
        # Adding node with smallest p-value to the list of aaglomerated nodes
        # ---------------------------------------------------------------------
        added_nodes.append((next_node,
                            info[next_node][0],
                            info[next_node][1],
                            info[next_node][2]))

        # Updating the list of cluster nodes and s0
        cluster_nodes.add(next_node)
        s0 = len(cluster_nodes)
        not_in_cluster |= (neighbors[next_node] - cluster_nodes)
        not_in_cluster.remove(next_node)

    return added_nodes


def DiaBLE(G_original, seed_genes, max_number_of_added_nodes, alpha, outfile=None):
    """
    Compute the DiaBLE ranking of the first max_number_of_added_nodes putative genes

    :param G_original: interactome graph
    :param seed_genes: seed genes (set)
    :param max_number_of_added_nodes: number of iterations to perform; as for DIAMOnD, 200 is reasonable
    :param alpha: see DIAMOnD
    :param outfile: file in which save the results

    :return ranking of max_number_of_added_nodes genes
    """
    # 1. throwing away the seed genes that are not in the network
    all_genes_in_network = set(G_original.nodes())
    seed_genes = set(seed_genes)
    disease_genes = seed_genes & all_genes_in_network

    if len(disease_genes) != len(seed_genes):
        print("DiaBLE(): ignoring %s of %s seed genes that are not in the network" % (
            len(seed_genes - all_genes_in_network), len(seed_genes)))

    # 2. agglomeration algorithm.
    added_nodes = diable_iteration_of_first_X_nodes(G_original,
                                                    disease_genes,
                                                    max_number_of_added_nodes, alpha)
    # 3. saving the results
    node_ids = []
    with open(outfile, 'w') as fout:

        fout.write('\t'.join(['#rank', 'DiaBLE_node', 'p_hyper']) + '\n')
        rank = 0
        for DiaBLE_node_info in added_nodes:
            rank += 1
            DiaBLE_node = DiaBLE_node_info[0]
            p = float(DiaBLE_node_info[3])
            node_ids.append(DiaBLE_node)

            fout.write('\t'.join(map(str, ([rank, DiaBLE_node, p]))) + '\n')

    return node_ids


def DIABLE(network_file, seed_file, n, alpha=1, out_file='data/diable_results.txt'):
    """
    Implementation of the DiaBLE algorithm based on diamond_iteration_of_first_X_nodes()
    function from https://github.com/dinaghiassian/DIAMOnD.git

    :param network_file: file providing the network in the form of an edgelist
    :param seed_file: file with the seed genes (if table contains more than one
                      column they must be tab-separated; the first column will be used only)
    :param n: number of desired iterations
    :param alpha: seeds weight (integer), default value is 1
    :param out_file: name for the results file

    :return 
    """

    # infer n
    if n == -1:
        network_df = pd.read_csv(network_file, sep=',', header=None)
        n = pd.concat([network_df.iloc[:,0], network_df.iloc[:,1]]).nunique()
        del network_df

    # check if input style is correct
    input_list = [network_file, seed_file, n, alpha, out_file] 
    network_edgelist_file, seeds_file, max_number_of_added_nodes, alpha, outfile_name = diamond.check_input_style(input_list)

    # read the network and the seed genes:
    G_original, seed_genes = diamond.read_input(network_edgelist_file, seeds_file) # graph, set

    # run DiaBLE
    added_nodes = DiaBLE(G_original,
                         seed_genes,
                         max_number_of_added_nodes, alpha,
                         outfile=outfile_name)

    print("\n results have been saved to '%s' \n" % outfile_name) 



# #################################################################
# #
# #                 Random Walk with Restart 
# #
# #################################################################

def RANDOM_WALK_WITH_RESTART(network_file, seed_file, r, score_thr=0.4, tol=1e-6, out_file='data/random_walk_wr_results.txt'):
    """
    Implementation of the Random Walk With Restart algorithm described in Walking the Interactome
    for Prioritization of Candidate Disease Genes, Sebastian Kohler et al.,
    https://dx.doi.org/10.1016%2Fj.ajhg.2008.02.013

    :param network_file: interactom network file
    :param seed_file: disease genes' file
    :param r: probability of restarting
    :param score_thr: probability threshold under which truncate the output rank
    :param tol: tolerance threshold for convergence
    :param out_file: file in which save the results

    :return 
    """
    assert r >= 0 and r <= 1, 'Probability not valid'
    assert score_thr >= 0 and score_thr <= 1, 'Score threshold not valid'

    # read the network and the seed genes:
    G, seed_genes = read_input(network_file, seed_file)

    # 1. throwing away the seed genes that are not in the network
    all_genes_in_network = set(G.nodes())
    seed_genes = set(seed_genes)
    disease_genes = seed_genes & all_genes_in_network

    if len(disease_genes) != len(seed_genes):
        print("RANDOM_WALK_WITH_RESTART(): ignoring %s of %s seed genes that are not in the network" % (
            len(seed_genes - all_genes_in_network), len(seed_genes)))

    # 2. agglomeration algorithm.
    genes_rank = random_walk_wr(G, disease_genes, r, score_thr, tol)

    # 3. saving the results
    with open(out_file, 'w') as fout:

        fout.write('\t'.join(['#rank', 'Gene', 'p']) + '\n')
        rank = 0
        for entry in genes_rank:
            fout.write('\t'.join(map(str, entry)) + '\n')

    print("\n results have been saved to '%s' \n" % out_file)
    

def read_input(network_file, seed_file):
    """
    Read the interactom network and disease genes' files.

    :param network_file: interactome file
    :param seed_file: disease genes' file

    :return interactome graph, disease genes in the file (list)
    """
    G = nx.read_edgelist(network_file, delimiter=',')
    seed_genes = pd.read_csv(seed_file, header=None, dtype=str).iloc[:,0].tolist()
    return G, seed_genes


def random_walk_wr(G:nx.Graph, seed_genes, r, score_thr, tol):
    """
    Core routine for the Random Walk With Restart algorithm.

    :param G: interactome graph
    :param seed_genes: disease genes restricted to the interactome
    :param r: probability of restarting
    :param score_thr: probability threshold under which truncate the output rank
    :param tol: tolerance threshold for convergence; in the ref. paper set to 1e-6

    :return genes rank: [(rank, gene, probability), ...]
    """
    N = G.number_of_nodes()
    W = normalize(nx.adjacency_matrix(G), norm='l1', axis=0, copy=False)

    index_to_node = {i:node for i, node in enumerate(G.nodes())}
    node_to_index = {node:i for i, node in index_to_node.items()}

    p0 = np.zeros(shape=(N,1))
    p0[ [node_to_index[node] for node in seed_genes] ] = 1 / len(seed_genes)

    l1_change = 1
    pt = np.copy(p0)

    while(l1_change > tol):
        pt_1 = (1 - r) * W @ pt + r * p0
        
        l1_change = np.linalg.norm(pt_1 - pt, 1)

        pt = pt_1

    pt_1 = pt_1[pt_1 >= score_thr]
    
    sorted_idxs = np.argsort(pt_1)[::-1]
    sorted_nodes = [index_to_node[i] for i in sorted_idxs]
    rank = list( zip(range(1,len(sorted_nodes)+1), sorted_nodes, pt_1[sorted_idxs]) )

    return rank


def k_fold(func, metric_func, k=10, **kwargs):
    """
    """
    # network_file, seed_file
    network_file = kwargs['network_file']
    seed_file = kwargs['seed_file']
    G, seed_genes = read_input(network_file, seed_file)

    # 1. throwing away the seed genes that are not in the network
    all_genes_in_network = set(G.nodes())
    seed_genes = set(seed_genes)
    disease_genes = np.array(list(seed_genes & all_genes_in_network))
    n = len(disease_genes)

    kfolds = KFold(n_splits=k, shuffle=True) 

    metrics = defaultdict(list)

    for train_idx, test_idx in kfolds.split(disease_genes):
        out = np.array(func(G, disease_genes[train_idx], **kwargs['func_args']))
        gt = disease_genes[test_idx]
        metrics_current_split = metric_func(out, gt, n)
        for key in metrics_current_split.keys():
            metrics[key].append(metrics_current_split[key])
    
    for metric, values in metrics.items():
        metrics[metric] = (np.average(values), np.std(values))

    return metrics


def precision(pred:np.array, gt:np.array):
    """
    """
    rank_len = np.minimum(len(pred), len(gt))
    tp_len = len(set(iter(pred)) & set(iter(gt)))
    return tp_len / rank_len


def recall(pred:np.array, gt:np.array):
    """
    prediction: np.array
    ground_truth: np.array
    """
    tp = len(set(iter(pred)).intersection(set(iter(gt))))
    fn = len(set(iter(gt)).difference(set(iter(pred))))

    return tp / (tp+fn)


def ndcg(pred:np.array, gt:np.array):
    """
    """
    p = np.minimum(len(pred), len(gt))
    idcg = np.sum(1 / np.log2(np.arange(2, p+2))) # i + 1
    intersection = np.isin(pred, gt)
    dcg = np.sum(1 / np.log2(np.nonzero(intersection)[0] + 2)) # shift the idxs in from [0,M] to [1,M+1], then add +1
    return dcg / idcg


def compute_metrics(pred, gt, n):
    """
    """
    top_pos = (50, n//10, n//4, n//2, n)
    metrics = {}

    for X in top_pos:
        metrics[f'precision_at_{X}'] = p = precision(pred[:X], gt)
        metrics[f'recall_at_{X}'] = r = recall(pred[:X], gt)
        metrics[f'ndcg_at_k_{X}'] = ndcg(pred[:X], gt)
        metrics[f'F1-score_at_{X}'] = np.nan_to_num(2 * (p * r) / (p + r))

    return metrics