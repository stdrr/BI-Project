import pandas as pd
import numpy as np
import networkx as nx
import markov_clustering as mc


def human_data(file='data/BIOGRID-ORGANISM-Homo_sapiens-4.4.204.tab3.txt',
               save_file='data/homo_preprocess.tsv'):
    df = pd.read_csv(file, sep='\t', low_memory=False)
    our_df = df[(df['Organism ID Interactor A'] == 9606) & (df['Organism ID Interactor B'] == 9606)]
    our_df = our_df[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']]
    our_df = our_df.drop_duplicates()
    our_df = our_df[our_df['Entrez Gene Interactor A'] != our_df['Entrez Gene Interactor B']]
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


def MCL(human_file, start=18, end=27):
    interactome_df = pd.read_csv(human_file, sep='\t')
    interactome_g = nx.from_pandas_edgelist(interactome_df, source='Entrez Gene Interactor A',
                                            target='Entrez Gene Interactor B')
    matrix = nx.to_scipy_sparse_matrix(interactome_g)
    for inflation in [i / 10 for i in range(start, end)]:
        result = mc.run_mcl(matrix, inflation=inflation)
        clusters = mc.get_clusters(result)
        Q = mc.modularity(matrix=result, clusters=clusters)
        print("inflation:", inflation, "modularity:", Q)



