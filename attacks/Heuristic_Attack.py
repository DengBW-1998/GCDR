from deeprobust.graph.data import Dataset
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.linalg import eigh
import time
import deeprobust.graph.defense as dr
from utils import Bu2Adj
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


def get_train_val_test(idx, train_size, val_size, test_size, stratify):
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, sparse=False):
    if preprocess_adj == True:
        adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

    if preprocess_feature:
        features = normalize_f(features)

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
        labels = sparse_mx_to_torch_sparse_tensor(labels)
    else:
        labels = torch.LongTensor(np.array(labels))
        features = torch.FloatTensor(np.array(features.todense()))
        adj = torch.FloatTensor(adj.todense())

    return adj, features, labels


def compute_lambda(adj, idx_train, idx_test):
    num_all = adj.sum().item() / 2
    train_train = adj[idx_train][:, idx_train].sum().item() / 2
    test_test = adj[idx_test][:, idx_test].sum().item() / 2
    train_test = num_all - train_train - test_test
    return train_train / num_all, train_test / num_all, test_test / num_all


def heuristic_attack(adj, n_perturbations, idx_train, idx_unlabeled, lambda_1, lambda_2, vlabels):
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_unlabeled)
    degree = adj.sum(dim=1).to('cpu')
    canditate_train_idx = idx_train[degree[idx_train] < (int(degree.mean()) + 1)]
    candidate_test_idx = idx_test[degree[idx_test] < (int(degree.mean()) + 1)]
    #     candidate_test_idx = idx_test
    perturbed_adj = adj.clone()
    cnt = 0
    train_ratio = lambda_1 / (lambda_1 + lambda_2)
    n_train = int(n_perturbations * train_ratio)
    n_test = n_perturbations - n_train
    while cnt < n_train:
        node_1 = np.random.choice(canditate_train_idx, 1)
        node_2 = np.random.choice(canditate_train_idx, 1)
        if vlabels[node_1] != vlabels[node_2] and adj[node_1, node_2] == 0:
            perturbed_adj[node_1, node_2] = 1
            perturbed_adj[node_2, node_1] = 1
            cnt += 1

    cnt = 0
    while cnt < n_test:
        node_1 = np.random.choice(canditate_train_idx, 1)
        node_2 = np.random.choice(candidate_test_idx, 1)
        if vlabels[node_1] != vlabels[node_2] and perturbed_adj[node_1, node_2] == 0:
            perturbed_adj[node_1, node_2] = 1
            perturbed_adj[node_2, node_1] = 1
            cnt += 1
    return perturbed_adj

def Heuristic_Attack(Bu, n_flips):
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    u=Bu.shape[0]
    v=Bu.shape[1]
    vlabels = np.zeros((u + v))
    vlabels[u:] = 1

    val_size = 0.1
    test_size = 0.8
    train_size = 1 - test_size - val_size

    adj = Bu2Adj(Bu)
    idx = np.arange(adj.shape[0])
    idx_train, idx_val, idx_test = get_train_val_test(idx, train_size, val_size, test_size, stratify=vlabels)
    idx_unlabeled = np.union1d(idx_val, idx_test)

    features = np.ones((adj.shape[0], 32))
    features = sp.csr_matrix(features)

    device = 'cpu'

    adj = adj.todense()

    idx_unlabeled = np.union1d(idx_val, idx_test)
    # Generate perturbations
    lambda_1, lambda_2, lambda_3 = compute_lambda(adj, idx_train, idx_unlabeled)

    adj_matrix_flipped = heuristic_attack(torch.from_numpy(adj).float(), n_flips, idx_train, idx_unlabeled, lambda_1,
                                          lambda_2,vlabels)

    adj_matrix_flipped = sp.csr_matrix(adj_matrix_flipped)
    return adj_matrix_flipped
