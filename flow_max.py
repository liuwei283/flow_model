"""CytoTour
Usage:
    flow_max.py <lr_pair> <pathwaydb> <st_file>... [--species=<sn>] [--out=<fn>]
    flow_max.py (-h | --help)
    flow_max.py --version

Options:
    -s --species=<sn>   Choose species Mouse or Human [default: Mouse].
    -o --out=<fn>   Outdir [default: .].
    -h --help   Show this screen.
    -v --version    Show version.
"""

import pandas as pd
import numpy as np
from docopt import docopt
import Cytograph
import multiprocessing as mp
import datetime
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from math import inf
from scipy import optimize



def initialize_common_pathway_alpha(pathway, common_pathway_indices, same_type_expression):
    common_pathways = pathway.loc[common_pathway_indices, :]
    for pidx in common_pathway_indices:
        row = common_pathways.loc[pidx]
        src_node = row['src']
        dest_node = row['dest']

        ### alpha for the output flow from the source node
        output_flow = common_pathways[common_pathways['src'] == src_node]
        output_nodes = output_flow['dest'].values
        alpha_output = same_type_expression.loc[src_node]['mean']/np.sum(same_type_expression.loc[output_nodes]['mean'].values)

        ### alpha for the input flow to the destination node
        input_flow = common_pathways[common_pathways['dest'] == dest_node]
        input_nodes = input_flow['src'].values
        alpha_input = same_type_expression.loc[dest_node]['mean']/np.sum(same_type_expression.loc[input_nodes]['mean'].values)

        common_pathways.loc[pidx, 'alpha_out'] = alpha_output
        common_pathways.loc[pidx, 'alpha_in'] = alpha_input
    print("Common pathways:")
    print(common_pathways)
    return common_pathways


def find_noncommon_pathways(cell_network_collection, common_pathways):
    common_pathway_indices = common_pathways.index.values
    for cell_network in cell_network_collection:
        cell_pathway_indices = cell_network.get_pathway_original_indices()
        non_common_pathway_indices = np.where(~np.in1d(cell_pathway_indices, common_pathway_indices))[0]
        cell_network.set_noncommon_pathway_indices(non_common_pathway_indices)

############ collect and convert alpha values stored in the cell network instances to a single vector
# order:
# 1. common pathways output alpha
# 2. common pathways input alpha
# 3. single cell pathways output alpha
# 4. single cell pathways input alpha
def process_alpha_vector(cell_network_collection, common_pathways):
    alpha_vector = np.concatenate((common_pathways['alpha_out'].values, common_pathways['alpha_in']))
    for cell_network in cell_network_collection:
        non_common_pathways = cell_network.get_noncommon_pathways()
        alpha_vector = np.concatenate((alpha_vector, non_common_pathways['alpha_out'].values, non_common_pathways['alpha_in'].values))
    return alpha_vector

############ using simplex method to update alpha values
# alpha order - out -> in common -> noncommon
def update_alpha_values(cell_network_collection, common_pathways, n_alpha, common_pathway_indices):

    c = np.full(n_alpha, 1) # coefficient for objective variables - minimize alpha values to maximize the flow values

    alpha_idx = len(common_pathway_indices) * 2
    cell_count = 0
    linprog_matrix = np.array([])
    b_values = np.array([])


    for cell_network in cell_network_collection:
        cell_linprog_matrix, cell_b_values, alpha_increment = cell_network.generate_linprog_matrix_for_alpha_update(common_pathway_indices, alpha_idx, n_alpha)
        alpha_idx += alpha_increment

        cell_count += 1
        if cell_count == 1:
            linprog_matrix = cell_linprog_matrix
            b_values = cell_b_values
        else:
            linprog_matrix = np.vstack((linprog_matrix, cell_linprog_matrix))
            b_values = np.concatenate((b_values, cell_b_values))

    # check whether there exists some
    check_zero_columns = np.sum(linprog_matrix, axis = 0)
    print("Checking whether there are zero columns")
    print(np.where(check_zero_columns==0))
 
    print("Verifying the parameters for linear programming:")
    print(c.shape)
    print(linprog_matrix.shape)
    print(b_values.shape)

    # simplex method to maximize the flow sum in the current cell network
    return optimize.linprog(-c, A_ub = linprog_matrix, b_ub = b_values)

def set_updated_alpha(alpha_vector, cell_network_collection, common_pathways, n_alpha, common_pathway_indices):
    # update alpha values for common pathways
    n_common = len(common_pathway_indices)
    common_pathways['alpha_out'] = alpha_vector[:n_common]
    common_pathways['alpha_in'] = alpha_vector[n_common: 2 * n_common]
    print("updated alpha values for common pathways:")
    print(common_pathways)

    # update alpha values for noncommon pathways
    alpha_idx = n_common * 2
    for cell_network in cell_network_collection:
        alpha_increment = cell_network.update_alpha_values(alpha_vector, alpha_idx)
        alpha_idx += alpha_increment


def main(arguments):

    ############ read arguments
    print(arguments)
    st_files = arguments.get("<st_file>")
    lr_pair = arguments.get("<lr_pair>")
    pathwaydb = arguments.get("<pathwaydb>")
    species = str(arguments.get("--species"))
    out_dir = str(arguments.get("--out"))
    starttime = datetime.datetime.now()

    print("start reading data...")
    if len(st_files)==2:
        if st_files[0].endswith(".csv"):
            st_meta = pd.read_csv(st_files[0])
            if 'cell_type' not in st_meta.columns:
                TypeError("There is no column named 'cell_type'")
        else:
            TypeError("st_file should be st_meta.csv and st_data.csv or xxx.h5ad")
        st_data = pd.read_csv(st_files[1],index_col=0)
    print("reading data done")


    ############ read data
    st_data = st_data[st_data.apply(np.sum,axis=1)!=0] # filter gene with 0 expression
    st_gene = st_data.index.values.tolist() # expression genes
    lr_pair = pd.read_csv(lr_pair)
    lr_pair = lr_pair[lr_pair['species'] == species][['ligand', 'receptor']] # read ligand receptor pairs data of input species
    # lr_gene = list(set(lr_pair['ligand'].values).union(set(lr_pair['receptor'].values)))
    pathway = pd.read_table(pathwaydb, delimiter='\t', encoding= 'unicode_escape') # read pathway data / edges of flow graph


    ############ filtering
    ### pathway
    print("------------------------")
    pathway = pathway[["src", "dest"]][pathway['species'] == species].drop_duplicates() # filter out invalid data
    pathway = pathway[(pathway['src'].isin(st_gene))&(pathway['dest'].isin(st_gene))] # pathway gene should have expression value on st data
    print("The number of valid pathways:")
    print(pathway.shape) # size of valid pathway pairs
    pathway_gene = list(set(pathway['src'].values).union(set(pathway['dest'].values))) # pathway genes - union of source and destination
    print("The nubmer of valid pathway genes:")
    print(len(pathway_gene))

    ### st_data
    print("------------------------")
    st_data = st_data.loc[pathway_gene, :] # we focus on gene expression related to pathways
    st_gene = st_data.index.values.tolist() # expression genes
    # st_data = preprocess_st(st_data, filtering)
    print("The size of filterd st expression data")
    print(st_data.shape)
    print("The nubmer of filtered st expression genes")
    print(len(st_gene))
    # st genes should be same as pathway genes


    ### ligand receptor data
    # it is discovered that lir pairs hold small similarity with gene pathways
    lr_pair = lr_pair.rename(columns={'ligand': 'src', 'receptor': 'dest'})
    ligand_genes = list(set(lr_pair['src'].values))
    receptor_genes = list(set(lr_pair['dest'].values))
    # lr_pair_in_pathway = lr_pair[lr_pair.set_index(['src','dest']).index.isin(pathway.set_index(['src','dest']).index)] # 这里可以发现lr pair和pathway有较小的重合度
    # # lr_pair = lr_pair[lr_pair['receptor'].isin(st_gene)&lr_pair['ligand'].isin(st_gene)]
    # print("------------------------")
    # print("The number of lr pairs that appear in gene pathways")
    # print(lr_pair_in_pathway.shape)
    

    ############ to do: filter invalid cells


    ############ get all cell types
    print("------------------------")
    cell_types = sorted(list(set(st_meta['cell_type'].values.tolist())))
    print("All cell types:")
    print(cell_types)
    print("The number of distinct cell types:")
    print(len(cell_types))


    cell_count = 0
    ############ check flow pattern for all cell types
    for cell_type in cell_types:
        cell_names = st_meta[st_meta['cell_type'] == cell_type]['cell']
        cell_network_collection = []

        ############ Initialization of Cell Networks
        for cell_name in cell_names:
            cell_count += 1
            ############ data processing
            cell_expression = st_data[cell_name]
            cell_expression = cell_expression[cell_expression > 0]
            cell_pathway_gene = cell_expression.index
            
            # only pathways with gene expression are considered valid
            cell_pathway = pathway[(pathway['src'].isin(cell_pathway_gene))&(pathway['dest'].isin(cell_pathway_gene))]
            cell_pathway_gene = set(cell_pathway['src'].values).union(set(cell_pathway['dest'].values))

            # find gene pathways that are also lr pairs if any
            # pathway_as_lr_pairs = lr_pair[lr_pair.set_index(['src','dest']).index.isin(cell_pathway.set_index(['src','dest']).index)]
            # print("The number of pathwys that are also ligand receptor pairs:")
            # print(pathway_as_lr_pairs.shape)

            # find input and output genes
            input_gene = cell_pathway_gene.intersection(receptor_genes)
            output_gene = cell_pathway_gene.intersection(ligand_genes)
            inner_gene = cell_pathway_gene - input_gene - output_gene

            if len((input_gene.intersection(output_gene))) > 0:
                print("Error: pathway gene cannot be ligand and receptor simultaneously.")

            input_gene = list(input_gene)
            output_gene = list(output_gene)
            inner_gene = list(inner_gene)

            ############ Using Markov chain to filter valid nodes and edges
            pathway_graph = Cytograph.PathGraph(cell_name, input_gene, output_gene, inner_gene, cell_expression)
            pathway_graph.build_adjacency_matrix(cell_pathway)
            valid_nodes, valid_pathway = pathway_graph.filter_invalid_pathways()

            if valid_pathway.shape[0] == 0 or len(valid_nodes) == 0:
                print("The network of " + cell_name + " is empty after filtering.")
                continue
            else:
                cell_network_collection.append(pathway_graph)

        print("Valid cell counts:")
        print(len(cell_network_collection))


        ############ find common pathways and initialize common alpha values
        same_type_expression = st_data[cell_names]
        same_type_expression['mean'] = np.mean(same_type_expression.values, axis = 1)
        pathway_stat = {}
        for cell_network in cell_network_collection:
            valid_pathway_indices = cell_network.get_pathway_original_indices()
            for idx in valid_pathway_indices:
                if idx in pathway_stat: pathway_stat[idx] += 1
                else: pathway_stat[idx] = 1

        n_cells = cell_names.size
        common_pathway_indices = [pidx for pidx, n_occ in pathway_stat.items() if n_occ > 0.2 * n_cells] # 0.2 is the changable ratio of the common pathway occurance
        common_pathway_indices.sort() # ascending order

        print("Indices for common pathways:")
        print(common_pathway_indices)

        # initialize alpha values for common pathways
        common_pathways = initialize_common_pathway_alpha(pathway, common_pathway_indices, same_type_expression) # to do: may changed to 1 for initializaiton
        # initialize alpha values for non-common pathways
        for cell_network in cell_network_collection:
            cell_network.initialize_alpha(common_pathways)

        find_noncommon_pathways(cell_network_collection, common_pathways)
        
        ############ Iterations to find best alpha and max flow values
        stop = False
        n_alpha = 0
        iter_count = 0
        while not stop:
            iter_count += 1
            print("Start new iteration...")
            # before alpha update
            pre_alpha_values = process_alpha_vector(cell_network_collection, common_pathways)
            n_alpha = len(pre_alpha_values)

            ### update flows in each cell
            for cell_network in cell_network_collection:
                cell_network.build_linprog()

            ### update alpha values integrating all cells of the current cell types
            linprog_res = update_alpha_values(cell_network_collection, common_pathways, n_alpha, common_pathway_indices)
            # print(linprog_res.x)
            if linprog_res.success == True:
                print("Completed one iteration for alpha iteration.")
                # set updated alpha values
                updated_alpha_vector = linprog_res.x
                # updated_alpha_vector[updated_alpha_vector < 0.001] = 0.001
                # print(updated_alpha_vector)
                set_updated_alpha(updated_alpha_vector, cell_network_collection, common_pathways, n_alpha, common_pathway_indices)

                # stop condition - differences of the alpha vector
                diff = np.linalg.norm(np.array(pre_alpha_values) - np.array(updated_alpha_vector))
                if diff < 0.1:
                    stop = True
                    print("Stop iterating")
                    # updated_alpha_vector[updated_alpha_vector < 0.001] = 0
                print("Difference of alpha vector:")
                print(diff)
            else:
                print("Error for this linear programming task - alpha update.")
                print(linprog_res.message)
                break

        print("The number of total iterations:")
        print(iter_count)

        ############ check the similarity of pathway graphs of the same cell type
        for cell_network in cell_network_collection:
            common_pathways = cell_network.get_common_pathway_flow(common_pathways)

        # print(common_pathways)
        common_pathways.to_csv("./common_path_flows_" + cell_type + ".csv")
        break

if __name__=="__main__":
    arguments = docopt(__doc__, version="flow_max 1.0.0")
    main(arguments)




