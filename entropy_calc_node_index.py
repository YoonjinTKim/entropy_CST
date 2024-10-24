import os
import networkx as nx 
import numpy as np
import math
import matplotlib.pyplot as plt
import random 
from networkx import bipartite
import argparse
from threading import Thread
import pandas as pd
#import cProfile
def network_entropy(G):
    temp_sum = 0
    edge_list = G.edges()
    prob_sum = graph_prob_sum(G)

    for e in edge_list:
        temp_sum = temp_sum + (edge_p(e, G)/prob_sum) * math.log(edge_p(e, G)/prob_sum)
    return -1 * temp_sum


#Update the entropy of the graph
def update_entropy(f_ajc, virus_degree_dict, host_degree_dict, host_node, virus_node, original_entropy, original_f_sum):
    #How much total sum is going to increase? node1's row sum + node2's column sum 
    #print("update entropy")
    #print("host is " + host_node)
    #print("virus is " + virus_node) 
    host_degree = host_degree_dict[host_node]
    # node 2's degree 
    virus_degree = virus_degree_dict[virus_node]

    df = f_ajc.copy()
    df[host_node][virus_node] = host_degree * virus_degree
    # update host's neighbors
    df[host_node] = df[host_node] / host_degree * (host_degree + 1)
    sum_change = sum(df[host_node] / (host_degree + 1))
    # update virus' neighbors
    #print("using loc")
    #print(virus_node)
    df.loc[virus_node] = df.loc[virus_node] / virus_degree*(virus_degree + 1)
    for host in list(df.columns.values):
        if df[host][virus_node] != 0: 
            df[host][virus_node] = df[host][virus_node] / virus_degree * (virus_degree + 1)
    #df[virus_node] = new_virus
    sum_change = sum_change + sum(df.loc[virus_node] / (virus_degree + 1))
    #print(df)
    # Update adjacency matrix
    # if the time is still too big 
    sum_tot = host_degree * virus_degree + original_f_sum + sum_change 
    #sanity check
    #print("sanity check to see if the sum is correct sum")
    #print(sum_tot)
    #print(df.to_numpy().sum())
    # Calculate 
    t = df* (1/sum_tot)
  
    log_p = np.log(t, out=np.zeros_like(t), where = (t != 0))
    #print(log_p)
    p_sum = -1 * (t.mul(log_p).to_numpy().sum())
    return (p_sum - original_entropy)

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
    #biadjacency_matrix = biadjacency_matrix(G)
    #n_
    return temp_sum

def network_entropy_h(G):
    top_nodes = [node for node in G.nodes() if G.nodes()[node]['bipartite']==0]
    bottom_nodes = [node for node in G.nodes() if G.nodes()[node]['bipartite']==1]

    ajc = bipartite.biadjacency_matrix(G, top_nodes, bottom_nodes).todense()
    V_1_degree = np.sum(ajc, axis=1).tolist()
    V_2_degree = np.sum(ajc, axis=0).tolist()
    prod = np.matmul(V_1_degree, V_2_degree)
    prod_sum_score = np.concatenate(np.multiply(prod, ajc)).sum()
    #return (np.concatenate(prod_sum_score).sum())

    temp_mat = prod * (1/prod_sum_score)

    return -1 * (np.concatenate(np.multiply(np.multiply(temp_mat, np.log(temp_mat)), ajc)).sum())



def information_gain_dict(G):
    ig_dict = {}
    og_entropy = network_entropy_h(G)
    top_nodes = {n for n,d in G.nodes(data=True) if d["bipartite"] == 0}
    bottom_nodes ={n for n,d in G.nodes(data=True) if d["bipartite"] == 1}
    
    ajc = bipartite.biadjacency_matrix(G, top_nodes, bottom_nodes) 
    print(ajc) 
    V_1_degree = np.sum(ajc, axis=1).tolist()
    V_2_degree = np.sum(ajc, axis=0).tolist()
    print(V_1_degree)
    print(V_2_degree)
    


    return ig_dict

def edge_prob_dict_threaded(G, edge, og_entropy):
    print(edge)
    G_temp = G.copy()
    G_temp.add_edge(edge[0], edge[1])
    change = network_entropy(G_temp) - og_entropy
    return change
    #return edge_prob

def edge_prob_dict_parallel(G):
    #TODO only get the bottom n 
    edge_prob = {}
    og_entropy = network_entropy(G)
    non_edges = non_edges_bipartite(G)
    with concurrent.futures.ThreadPoolExecutor as ex:
        for edge, change in zip(non_edges, ex.map):
            edge_prob[edge] = ex.submit(edge_prob_dict_threaded(G, edge, og_entropy))
        
    #number of nodes
    print(edge_prob)
    return {k: v for k,v in (sorted(edge_prob.items(), key=lambda item:item[1]))}

def update_entropy_parallel(f_ajc, virus_degree_dict, host_degree_dict, host_node, virus_node, original_entropy, original_f_sum):
    #How much total sum is going to increase? node1's row sum + node2's column sum 
    #print("update entropy")
    #print("host is " + host_node)
    #print("virus is " + virus_node) 
    host_degree = host_degree_dict[host_node]
    virus_degree = virus_degree_dict[virus_node]

    df = f_ajc.copy()
    df[host_node][virus_node] = host_degree * virus_degree
    # update host's neighbors
    df[host_node] = df[host_node] / host_degree * (host_degree + 1)
    sum_change = sum(df[host_node] / (host_degree + 1))
    # update virus' neighbors
    #print("using loc")
    #print(virus_node)
    for host in list(df.columns.values):
        if df[host][virus_node] != 0:
            df[host][virus_node] = df[host][virus_node] / virus_degree * (virus_degree + 1)
    sum_change = sum_change + sum(df.loc[virus_node] / (virus_degree + 1))
    #print(df)
    # Update adjacency matrix
    # if the time is still too big 
    sum_tot = host_degree * virus_degree + original_f_sum + sum_change
    # Calculate 
    t = df* (1/sum_tot)

    log_p = np.log(t, out=np.zeros_like(t), where = (t != 0))
    #print(log_p)
    p_sum = -1 * (t.mul(log_p).to_numpy().sum())
    return (p_sum - original_entropy)


    
def update_entropy_edge_prob_dict_parallel(G, host_nodes, virus_nodes, non_edges):
    ajc = bipartite.biadjacency_matrix(G, host_nodes, virus_nodes).todense()
    df = pd.DataFrame(ajc, columns = virus_nodes, index=host_nodes)
    host_degree = np.sum(ajc, axis=1).tolist()
    virus_degree = np.sum(ajc, axis=0).tolist()
    host_degree_dict = df.sum(numeric_only=True, axis= 0).to_dict()
    virus_degree_dict = df.sum(numeric_only=True, axis= 1).to_dict()
    prod = np.matmul(host_degree, virus_degree)
    f_score_ajc = pd.DataFrame(np.multiply(prod, ajc), columns = virus_nodes, index=host_nodes)
    original_entropy = network_entropy_h(G)
    original_sum = f_score_ajc.to_numpy().sum()

    res_dict = {}
    count = 0
    with concurrent.futures.ThreadPoolExecutor as ex:
        for edge, change in zip(non_edges, ex.map):
            print("count " + str(count))
            count = count + 1
            #check if the first node is host
            if edge[0] in host_nodes:
                res_dict[(edge[0], edge[1])] = ex.submit(update_entropy_parallel(f_score_ajc, virus_degree_dict, host_degree_dict, edge[1], edge[0], original_entropy, original_sum))
            else:
                res_dict[(edge[1], edge[0])] = ex.submit(update_entropy_parallel(f_score_ajc, virus_degree_dict, host_degree_dict, edge[0], edge[1], original_entropy, original_sum))

    #update_entropy(f_score_ajc, virus_degree_dict, host_degree_dict, virus_nodes[1], host_nodes[1], original_entropy, original_sum)
    return res_dict

def edge_prob_dict(G, host_nodes, virus_nodes, non_edges):
    #host_nodes = [n for n,d in G.nodes(data=True) if d["bipartite"] == 1]
    #virus_nodes = [n for n,d in G.nodes(data=True) if d["bipartite"] == 0]
    ajc = bipartite.biadjacency_matrix(G, host_nodes, virus_nodes).todense()
    df = pd.DataFrame(ajc, columns = virus_nodes, index=host_nodes)
    host_degree = np.sum(ajc, axis=1).tolist()
    virus_degree = np.sum(ajc, axis=0).tolist()
    host_degree_dict = df.sum(numeric_only=True, axis= 0).to_dict()
    virus_degree_dict = df.sum(numeric_only=True, axis= 1).to_dict()
    prod = np.matmul(host_degree, virus_degree)
    f_score_ajc = pd.DataFrame(np.multiply(prod, ajc), columns = virus_nodes, index=host_nodes)
    original_entropy = network_entropy_h(G)
    original_sum = f_score_ajc.to_numpy().sum()

    res_dict = {}
    count = 0 
    for edge in non_edges:
        print("count " + str(count))
        count = count + 1
        if count == 100:
            #exit()
            return res_dict
        #check if the first node is host
        if edge[0] in host_nodes:
            res_dict[(edge[0], edge[1])] = update_entropy(f_score_ajc, virus_degree_dict, host_degree_dict, edge[1], edge[0], original_entropy, original_sum)  
        else: 
            res_dict[(edge[1], edge[0])] = update_entropy(f_score_ajc, virus_degree_dict, host_degree_dict, edge[0], edge[1], original_entropy, original_sum)

    #update_entropy(f_score_ajc, virus_degree_dict, host_degree_dict, virus_nodes[1], host_nodes[1], original_entropy, original_sum)
    return res_dict

def greedy_edge(score_dict):
    return max(score_dict, key = score_dict.get)

def edge_p_est_1(G, edge):
    return G.degree(edge[0]) * G.degree(edge[1])

def edge_p_est_2(G, edge):
    #sum of edge[0]'s neighbor's degree
    edge_0_n = sum([G.degree(d) for d in [n for n in G[edge[0]]]])
    #sum of edge[1]'s neighbors' degree
    edge_1_n = sum([G.degree(d) for d in [n for n in G[edge[1]]]])

    return G.degree(edge[0]) * G.degree(edge[1]) * edge_0_n * edge_1_n

def est_1_dict(G, non_edges):
    res = {}
    for edge in non_edges:
        res[edge] = edge_p_est_1(G, edge)
    return res

def est_2_dict(G, non_edges):
    res = {}
    for edge in non_edges:
        res[edge] = edge_p_est_2(G, edge)
    return res

def greedy_algorithm_est(G, n, method):
    greedy = []
    non_edges = non_edges_bipartite(G)

    for i in range(n):
        print("greedy est")
        print(i)
        temp_edge = [k for k,v in sorted(method(G, non_edges).items(), key=lambda item:item[1])][0]
        G.add_edge(temp_edge[0], temp_edge[1])
        non_edges.remove(temp_edge)
        greedy.append(network_entropy_h(G))
    return greedy



def greedy_algorithm(G, n):
    greedy = []
    host_nodes = [n for n,d in G.nodes(data=True) if d["bipartite"] == 1]
    virus_nodes = [n for n,d in G.nodes(data=True) if d["bipartite"] == 0]
    non_edges = non_edges_bipartite(G) 
    # edge to be removed
    for i in range(n):
        print("greedy")
        print(i)
        temp_edge = greedy_edge(edge_prob_dict(G, host_nodes, virus_nodes, non_edges))
        G.add_edge(temp_edge[0], temp_edge[1])
        #non_edges.remove((temp_edge[0], temp_edge[1]))
        greedy.append(network_entropy_h(G))
        #print(network_entropy(G))
        #print(network_entropy_h(G))
    print("sanity check for greedy")
    print(len(G.edges()))
    return greedy

def all_at_once(n, temp_dict):
    # remove the all n bottom edges from 
    #sorted_list =  edge_prob_dict(G)
    #print(temp_dict)
    return {k: temp_dict[k] for k in list(temp_dict.keys())[:n]}
    #Get n number of least possible edges
    
  
def all_at_once_algorithm(G, n, temp_dict):
    #edges to be removed 
    adding_edge_list = all_at_once(n, temp_dict)
    G.add_edges_from(adding_edge_list.keys())
    #print("sanity check")
    #print(len(G.edges()))
    #print(network_entropy(G))
    #print(network_entropy_h(G))
    return network_entropy_h(G)


def k_at_once_algorithm(G, k):
    adding_edge_list = all_at_once(G, k)
    G.add_edges_from(adding_edge_list.keys())
    return network_entropy_h(G)

def random_consecutive_algorithm(G, n):
    #Randomly selecting and add 
    missing_edges = non_edges_bipartite(G)
    res = []
    for i in range(1, n+1):
        key = random.choice(missing_edges)
        missing_edges.remove(key)
        G.add_edge(key[0], key[1])
        res.append(network_entropy_h(G))
    print("Random sanity check")
    print(len(G.edges()))
    return res
def graph_entropy_for_taxa(G):
    res = {}
    non_edges = non_edges_bipartite(G)
    for edge in non_edges:
        #print(edge)
        G_temp = G.copy()
        G_temp.add_edge(edge[0], edge[1])
        res[edge] = network_entropy_h(G_temp)
    return {k: v for k, v in sorted(res.items(), key =lambda item:item[1])}

def graph_entropy_dict(G):
    res = {}
    non_edges = non_edges_bipartite(G)
    for edge in non_edges:
        G_temp = G.copy()
        G_temp.add_edge(edge[0], edge[1])
        res[edge] = network_entropy_h(G_temp) 
        #print(edge)
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
        score.append(network_entropy_h(G))

        #Recalculate
        sorted_list = edge_prob_dict(G)

    return x_axis, score

def simmulated_annealing(G, bounds, n_iter, step_size, temp):
    sorted_list = edge_prob_dict(G)
    best = sorted_list.keys()[0]
    G_temp = G.copy()
    best_eval = network_entropy_h(G.temp)

def alternate_prob(G):
    top_nodes = {n for n,d in G.nodes(data=True) if d["bipartite"] == 0}
    bottom_nodes ={n for n,d in G.nodes(data=True) if d["bipartite"] == 1}
    prob_res = {}
    for e in G.edges():
        prob_res[e] = (len(top_nodes) - G.degree(e[0])) * (len(bottom_nodes) - G.degree(e[1]))

    total_sum = sum(prob_res.values())
    return prob_res

def alternate_entropy(G):
    res = {}
    for edge in non_edges_bipartite(G):
        G_temp = G.copy()
        G_temp.add_edge(edge[0], edge[1])
        res[edge] = network_entropy_alternate(G_temp)
    return {k: v for k, v in sorted(res.items(), key =lambda item:item[1])}






def network_entropy_alternate(G):
    dictionary = alternate_prob(G)
    print(dictionary)
    total_sum = sum(dictionary.values())
    temp_dict  = {k: float(v)/total_sum * np.log(float(v)/total_sum) for k,v in dictionary}
    return -1 * sum(temp_dict.values())

def greedy_edge_alternate(G, score_dict):
    min_value = min(score_dict.values())
    res = [key for key in score_dict if score_dict[key] == min_value]
    print(res)
    return res[0]


def greedy_algorithm_alternate(G, n):
    greedy = []
    # edge to be removed
    for i in range(n):
        print("greedy")
        print(i)
        temp_edge = greedy_edge_alternate(G, alternate_prob(G))
        print("temp_edge")
        print(temp_edge[0])
        print(temp_edge[1])
        G.add_edge(temp_edge[0], temp_edge[1])
        greedy.append(network_entropy_alternate(G))
        #print(network_entropy(G))
        #print(network_entropy_h(G))
    print("sanity check for greedy alt")
    print(len(G.edges()))
    print(greedy)
    return greedy



def main():

    
    parser = argparse.ArgumentParser(description="Input argument for plotting")
    parser.add_argument('--n', type=int, help="Number of edges to be added")
    parser.add_argument('--filename', type=str, help="networkx pickle file for input network")
    parser.add_argument('--texa', type=str, help="networkx pickle file for texanomic network")
    parser.add_argument('--est', type=bool, help="estiamtion of entropy")
    #Testing update entropy
    '''
    G = bipartite.gnmk_random_graph(5,6,10)
    top_nodes = [node for node in G.nodes() if G.nodes()[node]['bipartite']==0]
    bottom_nodes = [node for node in G.nodes() if G.nodes()[node]['bipartite']==1]

    ajc = bipartite.biadjacency_matrix(G, top_nodes, bottom_nodes).todense()
    V_1_degree = np.sum(ajc, axis=1).tolist()
    V_2_degree = np.sum(ajc, axis=0).tolist()
    print(V_1_degree)
    print(V_2_degree)
    prod = np.matmul(V_1_degree, V_2_degree)
    print(prod)
    print(ajc)
    prod_sum_score = np.multiply(prod, ajc)
    print(np.concatenate(prod_sum_score).sum())
    print(graph_prob_sum(G))
    update_entropy(ajc, prod, prod_sum_score, 0, 0)
    exit()
    '''
    args = parser.parse_args()
    filename = args.filename
    n = args.n

    if args.est:
        G = nx.read_gpickle(filename)
        G_est_1 = G.copy()
        G_est_2 = G.copy()

        est_1_greedy = greedy_algorithm_est(G_est_1, n, est_1_dict)
        est_2_greedy = greedy_algorithm_est(G_est_2, n, est_2_dict)
        print("here")
        print(est_1_greedy) 
        print(est_2_greedy)
        x_axis = np.linspace(1, n, n)

        plt.plot(x_axis, est_1_greedy, label='Edge prob estimation 1')
        plt.plot(x_axis, est_2_greedy, label='Edge prob estiamtion 2')
        plt.legend()
        #plt.title(filename)
        plt.xlabel('number of added edges')
        plt.ylabel('Graph entropy')
        plt.savefig("species_"+str(n)+"_est.png")
        plt.close()

        exit()
    if args.filename is None and  args.n is None:
        print("Insufficient input parameters. Must input graph and number of edges to be added")
        exit()
    elif args.filename is None:
        G = nx.Graph()
        G.add_nodes_from(['A', 'B', 'C', 'D'], bipartite=0)
        G.add_nodes_from(['E', 'F', 'G', 'H'], bipartite=1)
        G.add_edges_from([('A', 'E'), ('A', 'F'), ('A', 'G'), ('B', 'E'), ('B', 'F'), ('C', 'E'), ('C', 'F'), ('D', 'E'), ('D', 'H')])
        
    else:
        G = nx.read_gpickle(filename)

    host_nodes = [n for n,d in G.nodes(data=True) if d["bipartite"] == 1]
    virus_nodes = [n for n,d in G.nodes(data=True) if d["bipartite"] == 0]
    ajc = bipartite.biadjacency_matrix(G, host_nodes, virus_nodes).todense()
    df = pd.DataFrame(ajc, columns = virus_nodes, index=host_nodes)
    host_degree = np.sum(ajc, axis=1).tolist()
    virus_degree = np.sum(ajc, axis=0).tolist()
    host_degree_dict = df.sum(numeric_only=True, axis= 0).to_dict()
    virus_degree_dict = df.sum(numeric_only=True, axis= 1).to_dict()
    prod = np.matmul(host_degree, virus_degree)
    f_score_ajc = pd.DataFrame(np.multiply(prod, ajc), columns = virus_nodes, index=host_nodes)
    original_entropy = network_entropy_h(G)
    original_sum = f_score_ajc.to_numpy().sum()
    update_entropy(f_score_ajc, virus_degree_dict, host_degree_dict, virus_nodes[1], host_nodes[1], original_entropy, original_sum)

    #if (args.taxa is True):
    #    G_taxa = G.copy()
        
    #t_dict = graph_entropy_for_taxa(G)
    #print(t_dict)
    #return bottom 3 and top 3
    
    #information_gain_dict(G)
    #exit()
    
    #G_copy_all_at_once = G.copy()
    G_copy_greedy = G.copy()
    G_copy_greedy_alt = G.copy()
    G_copy_random = G.copy()
    G_copy_bottom_all = G.copy()
    all_at_once_score = []
    #all_at_once_dict = edge_prob_dict(G_copy_all_at_once)
    #for i in range(1, n+1):
    #    print("all_at_once")
    #    print(i)
    #    G_copy_all_at_once = G.copy()
    #    all_at_once_score.append(all_at_once_algorithm(G_copy_all_at_once, i, all_at_once_dict))
    

    '''
    #for k at once
    #G_copy_2 = G.copy()
    #G_copy_3 = G.copy()
    #G_copy_5 = G.copy()
    #G_copy_10 = G.copy()
 
    #k_2_at_once_score = []
    #k_3_at_once_score = []
    #k_5_at_once_score = []
    #k_10_at_once_score = [] 

    #k_2_x_axis = []
    k_3_x_axis = []
    k_5_x_axis = []
    k_10_x_axis = [] 
    #for i in range(513594):
    #    if (i%2 == 0):
    #        k_2_x_axis.append(i + 2)
    #        k_2_at_once_score.append(k_at_once_algorithm(G_copy_2, 2))
    #    if (i%3 == 0):
    #        k_3_x_axis.append(i+3)
    #        k_3_at_once_score.append(k_at_once_algorithm(G_copy_3, 3))
    #    if (i%5 == 0):
    #        k_5_x_axis.append(i+5)
    #        k_5_at_once_score.append(k_at_once_algorithm(G_copy_5, 5))
    #    if (i%10 == 0): 
    #        k_10_x_axis.append(i+10)
    #        k_10_at_once_score.append(k_at_once_algorithm(G_copy_10, 10))
    '''
    #filename = filename.split(".")[0]

    #cProfile.run(greedy_algorithm(G_copy_greedy, n))
    greedy_score = greedy_algorithm(G_copy_greedy, n)
    #greedy_score_alt = greedy_algorithm_alternate(G_copy_greedy_alt, n)
    #random_score = random_consecutive_algorithm(G_copy_random, n)
    
    '''
    bottom_x_axis, bottom_all_score = all_bottom_ties_at_once(G_copy_bottom_all, n)
    x_axis = np.linspace(1, n, n)
    #xmin = min(x_axis)
    #xmax = max(x_axis)
    #plt.xlim(xmin, xmax)
    
    #Testing why the scale is off
    plt.plot(x_axis, x_axis)
    plt.close()    



    #plt.plot(x_axis, all_at_once_score, label = 'all at once score')
    #plt.plot(x_axis, greedy_score, label = 'greedy score')
    plt.plot(x_axis, greedy_score_alt, label='greedy score alt')
    #plt.plot(x_axis, random_score, label = 'random consecutive score')
    #plt.plot(bottom_x_axis, bottom_all_score, label='all least possible ties at once score')
    #plt.plot(k_2_x_axis, k_2_at_once_score, label ='2 at once')
    #plt.plot(k_3_x_axis, k_3_at_once_score, label = '3 at once')
    #plt.plot(k_5_x_axis, k_5_at_once_score, label = '5 at once')
    #plt.plot(k_10_x_axis, k_10_at_once_score, label = '10 at once')
    plt.legend()
    plt.title(filename)
    plt.xlabel('number of added edges')
    plt.ylabel('Graph entropy')
    plt.savefig(filename+"_"+str(n)+".png")
    plt.close()

    
    #differences 
    greedy_diff = change_plot_in_list(greedy_score)
    greedy_diff_alt = change_plot_in_list(greedy_score_alt)
    # random_diff = change_plot_in_list(random_score)
    all_at_once_diff = change_plot_in_list(all_at_once_score)
    x_change_axis = np.linspace(1, (n-1), (n-1))
    #plt.plot(x_change_axis, random_diff, label = 'random consecutive change')
    plt.plot(x_change_axis, all_at_once_diff, label = 'all at once change')
    plt.plot(x_change_axis, greedy_diff, label = 'greedy change')
    #plt.plot(x_change_axis, greedy_diff_alt, label ='greedy score alt')
    plt.legend()
    plt.title("Clean CLOVER CST")
    plt.xlabel('Iteration')
    plt.ylabel('Graph entropy change')
    plt.savefig(filename+"_"+str(n) + '_diff.png')
    #plt.show()
    plt.close()
    '''

  
#cProfile.run('main()', sort='cumtime')
main()
