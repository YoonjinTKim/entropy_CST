import os
import networkx as nx 
import numpy as np
import math
import matplotlib.pyplot as plt
import random 
import scipy.stats as stats

def network_entropy(G):
    temp_sum = 0
    edge_list = G.edges()
    prob_sum = graph_prob_sum(G)
    for e in edge_list:
        temp_sum = temp_sum + (edge_p(e, G)/prob_sum) * math.log(edge_p(e, G)/prob_sum)
    return -1 * temp_sum

def non_edges_bipartite(G):
    non_edges = nx.non_edges(G)
    top_nodes = {n for n,d in G.nodes(data=True) if d["bipartite"] == 0}
    bottom_nodes ={n for n,d in G.nodes(data=True) if d["bipartite"] == 1}
    
    res = []
    for edge in non_edges:
        if edge[0] in top_nodes and edge[1] in bottom_nodes:
            res.append((edge[0], edge[1]))
        if edge[0] in bottom_nodes and edge[1] in top_nodes:
            res.append((edge[1], edge[0]))
    return res
# Edge probability
# edge e in graph G
def edge_p(e, G):
    # probability = (degree of start node) * (degree of target node)
    start_node_deg  = G.degree[e[0]]
    target_node_deg = G.degree[e[1]]


    return start_node_deg * target_node_deg

def graph_prob_sum(G):
    temp_sum = 0
    for edge in G.edges():
        temp_sum = temp_sum + edge_p(edge, G)

    return temp_sum


def edge_prob_dict(G):
    #TODO only get the bottom n 
    edge_prob = {}
    for edge in non_edges_bipartite(G):
        #if (len(edge_prob) > n and edge_p(edge, G) > max(edge_prob.values())):
        #    continue
        edge_prob[edge] = edge_p(edge, G)
    
    return {k: v for k,v in sorted(edge_prob.items(), key=lambda item:item[1])}

def greedy_edge(G):
    return min(edge_prob_dict(G), key = edge_prob_dict(G).get)


def greedy_algorithm(G, n):
    greedy = []
    # edge to be removed
    for i in range(n):
        temp_edge = greedy_edge(G)
        G.add_edge(temp_edge[0], temp_edge[1])
        greedy.append(network_entropy(G))
    print("sanity check for greedy")
    print(len(G.edges()))
    return greedy

def all_at_once(G, n):
    # remove the all n bottom edges from 
    sorted_list =  edge_prob_dict(G)
    return {k: sorted_list[k] for k in list(sorted_list)[:n]}
    #Get n number of least possible edges
    
  
def all_at_once_algorithm(G, n):
    #edges to be removed 
    adding_edge_list = all_at_once(G, n)
    G.add_edges_from(adding_edge_list.keys())
    #print("sanity check")
    #print(len(G.edges()))
    return network_entropy(G)


def k_at_once_algorithm(G, k):
    adding_edge_list = all_at_once(G, k)
    G.add_edges_from(adding_edge_list.keys())
    return network_entropy(G)

def random_consecutive_algorithm(G, n):
    #Randomly selecting and add 
    missing_edges = non_edges_bipartite(G)
    res = []
    for i in range(1, n+1):
        key = random.choice(missing_edges)
        missing_edges.remove(key)
        G.add_edge(key[0], key[1])
        res.append(network_entropy(G))
    return res

def graph_entropy_dict(G):
    res = {}
    for edge in non_edges_bipartite(G):
        G_temp = G.copy()
        G_temp.add_edge(edge[0], edge[1])
        res[edge] = network_entropy(G_temp) 
    return {k: v for k, v in sorted(res.items(), key =lambda item:item[1])}

def change_plot_in_list(raw_val):
    change = []
    for i in range(len(raw_val) - 1):
        change.append(raw_val[i+1] - raw_val[i])
    return change

def all_bottom_ties_at_once(G, n):
    sorted_list = edge_prob_dict(G)
    x_axis = []
    score = []
    edge_count = 0
    # Run until all the edges are added 
    while edge_count < n:
        #Get the least possible edge
        least_possible_value = min(sorted_list.values())
        #Get all the bottom ties 
        bottom_ties = [k for k,v in sorted_list.items() if v == least_possible_value]
        #x_axis.append(len(bottom_ties))
        edge_count = edge_count + len(bottom_ties)
        x_axis.append(edge_count)
        G.add_edges_from(bottom_ties)
        score.append(network_entropy(G))

        #Recalculate
        sorted_list = edge_prob_dict(G)

    return x_axis, score

def simmulated_annealing(G, bounds, n_iter, step_size, temp):
    sorted_list = edge_prob_dict(G)
    best = sorted_list.keys()[0]
    G_temp = G.copy()
    best_eval = network_entropy(G.temp)



def main():
    filename = 'clean_CLOVER_species.pickle'
    G = nx.read_gpickle(filename)
    #number_of_edges = n 
    n = 500

    iteration = 1000

    non_edges = non_edges_bipartite(G)
    x_axis = []
    score = []
    for i in range(iteration):
        G_temp = G.copy()
        #Select n number 
        edge_to_be_added = random.choices(non_edges, k=n)
        G_temp.add_edges_from(edge_to_be_added)
        score.append(network_entropy(G_temp))
        x_axis.append(i)
        print(i)
        
    plt.plot(x_axis, score, 'o')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Graph entropy')
    plt.savefig(filename.split(".")[0] + "_multiplerun_" + str(n) + "_it" + str(iteration) + ".png")
    plt.close()

    mu, std = stats.norm.fit(score)
    dist = stats.norm(mu, std)
    print(mu)
    print(std)
    x = np.linspace(min(score), max(score), 1000)
    pdf = dist.pdf(x)
    plt.hist(score)
    plt.plot(x, pdf, 'r', label='Fitted Normal Distribution')
    plt.legend()
    plt.ylabel('Graph entropy')
    plt.savefig(filename.split(".")[0] + "_histo_" + str(n) + "_it" + str(iteration) + ".png")
    plt.close()

    
   
  
    
     


    


  
main()
