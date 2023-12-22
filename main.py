import RUN
import argparse
import torch
import pandas as pd
import numpy as np
import networkx as nx
import copy
import matplotlib.pyplot as plt
import os
import sys
import tqdm
from models.utils import pearson_correlation, breaktie

def Run(datafile):
    df_data = pd.read_csv(datafile)
    edges = dict()
    
    df_data.drop("time", axis=1, inplace=True) #synthetic don't need this one
    columns = list(df_data)

    for c in columns: 
        idx = df_data.columns.get_loc(c)
        edge = RUN.GraphConstruct(c, cuda=cuda, epochs=nrepochs, 
        lr=learningrate, optimizername=optimizername, file=datafile, args=args)
        edges.update(edge)
    return edges, columns

def CreateGraph(edge, columns):
    G = nx.DiGraph()
    for c in columns:
        G.add_node(c)
    for pair in edge:
        p1,p2 = pair
        G.add_edge(columns[p2], columns[p1])
    return G

def main(datafiles):
    edge_pair, columns = Run(datafiles) 
    pruning = pd.read_csv(args.root_path + '/' + args.data_path)   
    G = CreateGraph(edge_pair, columns)

    while not nx.is_directed_acyclic_graph(G):
        edge_cor = []
        edges = G.edges()
        for edge in edges:
            source, target = edge
            edge_cor.append(pearson_correlation(pruning[source], pruning[target]))
        tmp = np.array(edge_cor)
        tmp_idx = np.argsort(tmp)
        edges = list(edges)
        source, target= edges[tmp_idx[0]][0], edges[tmp_idx[0]][1]

        G.remove_edge(source, target)
 
    dangling_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]
    personalization = {}
    for node in G.nodes():
        if node in dangling_nodes:
            personalization[node] = 1.0
        else:
            personalization[node] = 0.5
    pagerank = nx.pagerank(G, personalization=personalization)
    pagerank = dict(sorted(pagerank.items(), key=lambda x: x[1], reverse=True))

    pagerank = breaktie(pagerank, G, trigger_point)

    if trigger_point != "None":
        for i, node in enumerate(pagerank):
            if node == root_cause:
                if i < 2:
                    print(root_cause, "is in top-1")
                elif i < 4:
                    print(root_cause, "is in top-3")
                elif i < 6:
                    print(root_cause, "is in top-5")
                print(root_cause, "is at", i)
    else:
        previous_score = 0
        to_break = 0
        num_group = 0
        for node, rank in pagerank.items():
            if previous_score == rank:
                num_group += 1
            if previous_score != rank:
                to_break += 1
                to_break += num_group
                num_group = 0
            if node == root_cause:
                if to_break < 2:
                    print(root_cause, "is in top-1")
                elif to_break < 4:
                    print(root_cause, "is in top-3")
                elif to_break < 6:
                    print(root_cause, "is in top-5")
                print(root_cause, "is at", to_break)
            previous_score = rank
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN')

    parser.add_argument('--cuda', type=str, default="cuda:0")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--trigger_point', type=str, default='None', help='Calculate the distance between node and trigger point')
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--num_workers', type=float, default=10)
    parser.add_argument('--root_cause', type=str)

    args = parser.parse_args()

    nrepochs = args.epochs
    learningrate = args.learning_rate
    optimizername = args.optimizer
    cuda=args.cuda
    trigger_point = args.trigger_point
    root_cause = args.root_cause
    datafiles = args.root_path + '/' + args.data_path

    main(datafiles)
