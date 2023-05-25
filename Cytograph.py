

import pandas as pd
import numpy as np
import sys
import networkx as nx
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
from scipy import optimize
import math

class PathGraph(object):

    def __init__(self, cell_name, input_nodes, output_nodes, inner_nodes, cell_expression):
        self.cell_name = cell_name
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.inner_nodes = inner_nodes

        self.n_input = len(input_nodes)
        self.n_output = len(output_nodes)
        self.n_inner = len(inner_nodes)
        self.n_edges = 0

        self.nodes = input_nodes + output_nodes + inner_nodes
        self.n_nodes = len(self.nodes)
        
        # Initialize the adjacency matrix
        # Create a matrix with `num_of_nodes` rows and columns
        self.ad_matrix = np.zeros((self.n_nodes, self.n_nodes))
        self.connect_matrix = np.zeros((self.n_nodes, self.n_nodes))

        self.cell_expression = cell_expression
    

    def build_adjacency_matrix(self, pathway):
        # print("Pathway shapes:")
        # print(pathway.shape)
        pathway = pathway[(~pathway['dest'].isin(self.input_nodes))&(~pathway['src'].isin(self.output_nodes))] # remove related pathways
        # print(pathway.shape)

        self.pathway = pathway
        self.ad_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for index, row in self.pathway.iterrows():
            src = row['src']
            dest = row['dest']
            i = self.nodes.index(src)
            j = self.nodes.index(dest)
            self.ad_matrix[i,j] = 1
        np.fill_diagonal(self.ad_matrix, 1)
        self.connect_matrix = np.copy(self.ad_matrix)
        self.n_edges = pathway.shape[0]

    ############ The final connectivity status should be static
    def build_connectivity(self):
        stop = False
        count = 0
        # print("Starting to build the connectivity matrix:")
        
        while stop != True and count < self.n_edges + 2:
            # print("----------------")
            pre_connect_matrix = np.copy(self.connect_matrix)
            self.connect_matrix = np.dot(self.connect_matrix, self.ad_matrix) # use Markov chain
            self.connect_matrix[self.connect_matrix > 0] = 1

            # print(self.connect_matrix)

            if np.array_equal(pre_connect_matrix, self.connect_matrix) == True:
                stop = True
            count += 1
            # print(count)

        if stop == False:
            print("Error: cannot build the connectivity matrix.")
        # print("Complete building the connectivity matrix: " + self.cell_name)
        
    
    def filter_invalid_pathways(self):
        self.build_connectivity()
        # nodes should be touched by input nodes as dest and output nodes as src
        input_connectivity = np.sum(self.connect_matrix[:self.n_input, :], axis = 0)
        output_connectivity = np.sum(self.connect_matrix[:, self.n_input: self.n_input + self.n_output], axis = 1)
        valid_node_index = []
        for i in range(self.n_nodes):
            if input_connectivity[i] <= 0 or output_connectivity[i] <= 0:
                gene_name = self.nodes[i]
                # self.nodes.pop(i) # remove this node
                if i < self.n_input: self.input_nodes.remove(gene_name)
                elif i >= self.n_input and i < self.n_input + self.n_output: self.output_nodes.remove(gene_name)
                else: self.inner_nodes.remove(gene_name)
                self.pathway = self.pathway[(self.pathway['src']!=gene_name)&(self.pathway['dest']!=gene_name)] # remove related pathways
            else:
                valid_node_index.append(i)

        
        # reset cytograph status
        self.ad_matrix = self.ad_matrix[valid_node_index, valid_node_index]
        self.connect_matrix = self.connect_matrix[valid_node_index, valid_node_index]
        self.nodes = self.input_nodes + self.output_nodes + self.inner_nodes
        self.n_input, self.n_output, self.n_inner = len(self.input_nodes), len(self.output_nodes), len(self.inner_nodes)
        self.n_nodes = len(self.nodes)

        self.pathway['original_index'] = self.pathway.index.values
        self.pathway.reset_index(inplace=True)

        return self.nodes, self.pathway

    # ############ Plot Network Graph
    # to do: more advanced, intuitive and fancy networks
    def plot_graph(self, cell_expression):

        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.pathway[['src', 'dest']].values)

        # set position
        graph_size = (200, 200)
        pos = {}
        for i in range(self.n_input):
            pos[self.input_nodes[i]] = [70, 20 + i * 150 / self.n_input]

        for i in range(self.n_output):
            pos[self.output_nodes[i]] = [170, 20 + i * 150 / self.n_output]

        for i in range(self.n_inner):
            pos[self.inner_nodes[i]] = [120, 20 + i * 150 / self.n_inner]

        # pos = nx.spring_layout(G)

        print(self.nodes)
        print(pos)
        
        node_sizes = np.multiply(cell_expression[self.nodes].values.squeeze(), 20)

        node_colors = ['red'] * self.n_input + ['blue'] * self.n_output + ['green'] * self.n_inner

        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
        edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrows = True, arrowsize=10, edge_color='black', width=self.pathway['max_flow'].values, connectionstyle="arc3,rad=0.1")

        ax = plt.gca()
        ax.set_axis_off()
        plt.show()
    
    # initialize the alpha value - to do: change the starting status of the alpha values
    def initialize_alpha(self, common_pathways):
        common_pathway_indices = common_pathways.index.values
        for index, row in self.pathway.iterrows():
            original_index = row['original_index']
            if original_index in common_pathway_indices:
                self.pathway.loc[index, 'alpha_out'] = common_pathways.loc[original_index, 'alpha_out']
                self.pathway.loc[index, 'alpha_in'] = common_pathways.loc[original_index, 'alpha_in']
            else:
                src_node = row['src']
                dest_node = row['dest']

                ### alpha for the output flow from the source node
                output_flow = self.pathway[self.pathway['src'] == src_node]
                output_nodes = output_flow['dest'].values
                alpha_output = self.cell_expression[src_node]/np.sum(self.cell_expression[output_nodes].values)

                ### alpha for the input flow to the destination node
                input_flow = self.pathway[self.pathway['dest'] == dest_node]
                input_nodes = input_flow['src'].values
                alpha_input = self.cell_expression[dest_node]/np.sum(self.cell_expression[input_nodes].values)

                self.pathway.loc[index, 'alpha_out'] = alpha_output
                self.pathway.loc[index, 'alpha_in'] = alpha_input

    def build_linprog(self):
        # variable order is important
        n_pathways = self.pathway.shape[0] # the number of decision variables
        n_constraints = self.n_input + 2 * self.n_inner + self.n_output # the number of constraints / the number of slack variable

        # print(self.pathway)
        # print("variables:")
        # print(n_pathways)
        # print("Constraints:")
        # print(n_constraints)

        # linprog_matrix = np.concatenate((np.full(n_pathways, 1), np.full(n_constraints + 1, 0)))
        linprog_matrix = np.array([])
        # col_num = linprog_matrix.shape
        # constraints of input nodes - only one side
        slack_vindex = 0

        c = np.full(n_pathways, 1)

        for input_node in self.input_nodes:

            output_flows = self.pathway[self.pathway['src'] == input_node]
            pathway_indices = output_flows.index.values
            # print("Debug:")
            # print(output_flows)
            flow_alphas = output_flows['alpha_out']

            constraint_equation = np.zeros(n_pathways)
            constraint_equation[pathway_indices] = flow_alphas
            if slack_vindex == 0:
                linprog_matrix = constraint_equation
            else:
                linprog_matrix = np.vstack((linprog_matrix, constraint_equation))
            slack_vindex += 1
        # constraints of inner nodes - both sides
        for inner_node in self.inner_nodes:
            # input constraints
            input_flows = self.pathway[self.pathway['dest'] == inner_node]
            input_pindices = input_flows.index.values
            input_alphas = input_flows['alpha_in']

            constraint_equation = np.zeros(n_pathways)
            constraint_equation[input_pindices] = input_alphas

            linprog_matrix = np.vstack((linprog_matrix, constraint_equation))
            slack_vindex += 1

            # output constraints
            output_flows = self.pathway[self.pathway['src'] == inner_node]
            output_pindices = output_flows.index.values
            output_alphas = output_flows['alpha_out']

            constraint_equation = np.zeros(n_pathways)
            constraint_equation[output_pindices] = output_alphas

            linprog_matrix = np.vstack((linprog_matrix, constraint_equation))
            slack_vindex += 1

        # constraints of output nodes - only one side
        for output_node in self.output_nodes:
            input_flows = self.pathway[self.pathway['dest'] == output_node]
            pathway_indices = input_flows.index.values
            flow_alphas = input_flows['alpha_in']

            constraint_equation = np.zeros(n_pathways)
            constraint_equation[pathway_indices] = flow_alphas

            linprog_matrix = np.vstack((linprog_matrix, constraint_equation))
            slack_vindex += 1

        b_values = np.concatenate((self.cell_expression[self.input_nodes].values, self.cell_expression[self.inner_nodes].values, self.cell_expression[self.inner_nodes].values, self.cell_expression[self.output_nodes].values))
        # print("Verifying the parameters for linear programming:")
        # print(c.shape)
        # print(linprog_matrix.shape)
        # print(b_values.shape)

        # simplex method to maximize the flow sum in the current cell network
        linprog_res = optimize.linprog(-c, A_ub = linprog_matrix, b_ub = b_values)
        if linprog_res.success == True:
            flow_res = linprog_res.x
            flow_res[flow_res < 0.1] = 0.1
            # print(flow_res)
            self.pathway['max_flow'] = flow_res
            # flow sum max of all pathways
            self.max_flow = np.sum(self.pathway['max_flow'].values)
        else:
            print("Error for this linear programming task - flow update.")
            print(linprog_res.message)
    
    def get_common_pathway_flow(self, common_pathways):
        for index, row in common_pathways.iterrows():
            if index in self.pathway['original_index'].values:
                common_pathways.loc[index, self.cell_name] = self.pathway[self.pathway['original_index'] == index]['max_flow'].values
            else:
                common_pathways.loc[index, self.cell_name] = 0

        return common_pathways

    def generate_linprog_matrix_for_alpha_update(self, common_pathway_indices, alpha_idx, n_alpha):
        linprog_matrix = np.array([])
        slack_vindex = 0
        n_common_pathway = len(common_pathway_indices) # the number of all common pathways instead of just the current cell
        n_noncommon_pathway = len(self.noncommon_pathway_indices) # the number of non-common pathways of the current cell

        alpha_increment = n_noncommon_pathway * 2

        for cur_node in self.nodes:
            if cur_node in self.input_nodes or cur_node in self.inner_nodes:
                output_flows = self.pathway[self.pathway['src'] == cur_node] # output flow from this node

                common_output_flows = output_flows[output_flows['original_index'].isin(common_pathway_indices)]
                noncommon_output_flows = output_flows[~(output_flows['original_index'].isin(common_pathway_indices))]# output_flows_indices_ori[~np.isin(output_flows_indices_ori, common_pathway_indices)]

                constraint_equation = np.zeros(n_alpha)

                flow_values_noncommon = noncommon_output_flows['max_flow']
                noncommon_output_flows_indices = np.where(np.in1d(self.noncommon_pathway_indices, noncommon_output_flows.index.values))[0]
                constraint_equation[noncommon_output_flows_indices + alpha_idx] = flow_values_noncommon

                if common_output_flows.shape[0] > 0:
                    flow_values_common = common_output_flows['max_flow']
                    common_output_flows_indices_in_common = np.where(np.in1d(common_pathway_indices, common_output_flows['original_index'].values))[0]
                    constraint_equation[common_output_flows_indices_in_common] = flow_values_common

                if slack_vindex == 0:
                    linprog_matrix = constraint_equation
                else:
                    linprog_matrix = np.vstack((linprog_matrix, constraint_equation))
                slack_vindex += 1

            if cur_node in self.inner_nodes or cur_node in self.output_nodes:
                input_flows = self.pathway[self.pathway['dest'] == cur_node] # output flow from this node

                common_input_flows = input_flows[input_flows['original_index'].isin(common_pathway_indices)]
                noncommon_input_flows = input_flows[~(input_flows['original_index'].isin(common_pathway_indices))]# output_flows_indices_ori[~np.isin(output_flows_indices_ori, common_pathway_indices)]

                constraint_equation = np.zeros(n_alpha)

                flow_values_noncommon = noncommon_input_flows['max_flow']
                noncommon_input_flows_indices = np.where(np.in1d(self.noncommon_pathway_indices, noncommon_input_flows.index.values))[0]
                constraint_equation[noncommon_input_flows_indices + alpha_idx + n_noncommon_pathway] = flow_values_noncommon

                if common_input_flows.shape[0] > 0:
                    flow_values_common = common_input_flows['max_flow']
                    common_input_flows_indices_in_common = np.where(np.in1d(common_pathway_indices, common_input_flows['original_index'].values))[0]
                    constraint_equation[common_input_flows_indices_in_common + n_noncommon_pathway] = flow_values_common

                if slack_vindex == 0:
                    linprog_matrix = constraint_equation
                else:
                    linprog_matrix = np.vstack((linprog_matrix, constraint_equation))
                slack_vindex += 1
                
        
        b_values = np.concatenate((self.cell_expression[self.input_nodes].values, self.cell_expression[self.output_nodes].values, self.cell_expression[self.inner_nodes].values, self.cell_expression[self.inner_nodes].values))
        
        return linprog_matrix, b_values, alpha_increment
    
    def update_alpha_values(self, alpha_vector, alpha_idx):
        n_noncommon = len(self.noncommon_pathway_indices)
        self.pathway.loc[self.noncommon_pathway_indices, 'alpha_out'] = alpha_vector[alpha_idx:alpha_idx + n_noncommon]
        self.pathway.loc[self.noncommon_pathway_indices, 'alpha_in'] = alpha_vector[alpha_idx + n_noncommon:alpha_idx + 2*n_noncommon]
        return 2*n_noncommon
    
    # reset version
    def set_noncommon_pathway_indices(self, noncommon_pathway_indices):
        self.noncommon_pathway_indices = noncommon_pathway_indices
        print("The number of noncommon way of " + self.cell_name + ":")
        print(len(self.noncommon_pathway_indices))
    
    def get_noncommon_pathways(self):
        return self.pathway.iloc[self.noncommon_pathway_indices, :]
    
    def get_pathways(self):
        return self.pathway
    
    def get_pathway_original_indices(self):
        return self.pathway['original_index'].values

