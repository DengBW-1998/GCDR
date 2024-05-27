import networkx as nx
import codes.graph
import random
from networkx.algorithms import bipartite as bi
import numpy as np
from codes.lsh import get_negs_by_lsh
from io import open
import os
import itertools

class GraphUtils(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.G = nx.Graph()

        self.edge_dict_u = {}
        self.edge_dict_v = {}
        self.edge_list = []
        self.node_u = []
        self.node_v = []
        self.u_nodes = 0
        self.v_nodes = 0
        self.n_nodes = 0 

    def construct_training_graph(self, filename=None):
        if filename is None:
            filename = os.path.join(self.model_path, "ratings_train.dat")
        edge_list_u_v = []
        edge_list_v_u = []
        with open(filename, encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rating = line.strip().split("\t")
                if self.edge_dict_u.get(user) is None:
                    self.edge_dict_u[user] = {}
                if self.edge_dict_v.get(item) is None:
                    self.edge_dict_v[item] = {}
                edge_list_u_v.append((user, item, float(rating)))
                self.edge_dict_u[user][item] = float(rating)
                self.edge_dict_v[item][user] = float(rating)
                edge_list_v_u.append((item, user, float(rating)))
                line = fin.readline()
        # create bipartite graph
        self.node_u = self.edge_dict_u.keys()
        self.node_v = self.edge_dict_v.keys()
        sorted(self.node_u)
        sorted(self.node_v)
        self.G.add_nodes_from(self.node_u, bipartite=0)
        self.G.add_nodes_from(self.node_v, bipartite=1)
        self.G.add_weighted_edges_from(edge_list_u_v+edge_list_v_u)
        self.edge_list = edge_list_u_v
        self.u_nodes=(int)(list(self.node_u)[-1][1:])+1
        self.v_nodes=1600
        self.n_nodes=self.u_nodes+self.v_nodes