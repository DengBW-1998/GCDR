import sys
import numpy as np
from sklearn import preprocessing
from deeprobust.graph.data import Dataset
from codes.data_utils import DataUtils
from codes.graph_utils import GraphUtils
import random
import math
import os
import scipy.sparse as sp

class BineModel(object):
    def __init__(self,dataset):   
        if dataset=='wiki':
            self.train_data=r'data/wiki/rating_train.dat'
            self.test_data=r'data/wiki/rating_test.dat'
            self.model_name='wiki'
            self.vectors_u=r'data/wiki/vectors_u.dat'
            self.vectors_v=r'data/wiki/vectors_v.dat'
            self.case_train=r'data/wiki/case_train.dat'
            self.case_test=r'data/wiki/case_test.dat' 
            self.dataset='wiki'
        
        if dataset=='dblp':
            self.train_data=r'data/dblp/rating_train.dat'
            self.test_data=r'data/dblp/rating_test.dat'
            self.model_name='dblp'
            self.vectors_u=r'data/dblp/vectors_u.dat'
            self.vectors_v=r'data/dblp/vectors_v.dat'
            self.dataset='dblp'

def adj_2_adjnn(adj,u,v,th):
    n=u+v
    adj_uv=adj[:u,u:]
    adj_vu=adj[u:,:u]
    adj_uv=np.array(adj_uv)
    adj_vu=np.array(adj_vu)

    adj_uv=sp.csr_matrix(adj_uv)
    adj_vu=sp.csr_matrix(adj_vu)
    adj=sp.csr_matrix(adj)
    adj_uu=adj_uv.dot(adj_vu)
    adj_vv=adj_vu.dot(adj_uv)
    
    
    adj_uu=adj_uu.todense()
    adj_vv=adj_vv.todense()
    adj_uu-=th
    adj_uu[adj_uu>=0]=1
    adj_uu[adj_uu<0]=0
    adj_vv-=th
    adj_vv[adj_vv>=0]=1
    adj_vv[adj_vv<0]=0

    adj_nn=adj.copy()
    adj_nn[:u,:u]=adj_uu
    adj_nn[u:,u:]=adj_vv 
    return adj_nn

def adjsp_2_adjnn(adj,u,v,th):
    n=u+v
    adj_uv=adj[:u,u:]
    adj_vu=adj[u:,:u]
    
    adj_uu=adj_uv.dot(adj_vu)
    adj_vv=adj_vu.dot(adj_uv)
        
    adj_uu=adj_uu.todense()
    adj_vv=adj_vv.todense()
    adj_uu-=th
    adj_uu[adj_uu>=0]=1
    adj_uu[adj_uu<0]=0
    adj_vv-=th
    adj_vv[adj_vv>=0]=1
    adj_vv[adj_vv<0]=0

    adj_nn=adj.copy()
    adj_nn[:u,:u]=adj_uu
    adj_nn[u:,u:]=adj_vv   
    return adj_nn

def getAdj(th,dataset,rate):
    
    if(dataset=='dblp' or dataset=='wiki'):
        bine=BineModel(dataset)

        model_path = os.path.join('./', bine.model_name)
        if os.path.exists(model_path) is False:
            os.makedirs(model_path)

        dul = DataUtils(model_path)
        gul = GraphUtils(model_path)
        gul.construct_training_graph(bine.train_data)

        #(1/rate) represents the proportion of nodes in the selected subgraph
        gul.u_nodes = gul.u_nodes//rate
        gul.v_nodes = gul.v_nodes//rate
        gul.n_nodes = gul.n_nodes//rate
        u,v,n=gul.u_nodes,gul.v_nodes,gul.n_nodes
        adj=[[0]*n for _ in range(n)] 

        for i in range(len(gul.edge_list)):
            u_index = (int)(gul.edge_list[i][0][1:])
            v_index = (int)(gul.edge_list[i][1][1:])
            if u>u_index and v>v_index:
                adj[u_index][v_index+u]=1
                adj[v_index+u][u_index]=1
        adj=np.array(adj)
      
        adj_nn=adj_2_adjnn(adj,u,v,th)
        adj=sp.csr_matrix(adj)

        return adj_nn,adj,u,v,None
    
    if (dataset=='citeseer' or dataset=='pubmed'):
        data = Dataset(root='./data/', name=dataset, setting='prognn')
        adj = np.array(data.adj.todense())
        #cora 2485 citeseer 2110  pubmed 19717
        #print(adj.shape)
        if dataset=='citeseer':
            u=1407
            v=703
        if dataset=='pubmed':
            u=13145
            v=6572
        labels = data.labels[:u + v]
        adj=adj[:u+v,:u+v]
        adj[:u,:u]=0
        adj[u:,u:]=0
        adj[adj>0]=(int)(1)

        # cora 4728//2  citeseer 2840//2  pubmed 38782//2
        #print(sum(sum(adj)))

        adj_nn=adj_2_adjnn(adj,u,v,th)
        adj=sp.csr_matrix(adj)

        return adj_nn,adj,u,v,labels
